"""Frontier exploration: a coverage grid from every sweep, and where to go next.

Every sweep marks the space the sound crossed as free and the cells where it
echoed as occupied; everything else stays unknown. Frontiers are free cells
next to unknown space: the edges of what has been mapped. Each stop the car
plans a path on the grid, kept a car's width away from obstacles, to the most
worthwhile reachable frontier, preferring ones that need right turns so it
sweeps round a room consistently, and takes one step along it. When no
reachable frontier is left, the area is explored.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from robocar.surfaces import SONAR_MAX_RANGE_CM, direction, sonar_origin

CELL_CM = 5.0
FREE_LOG_ODDS = -0.4
OCCUPIED_LOG_ODDS = 0.9
LOG_ODDS_LIMIT = 4.0
FREE_BELOW = -0.3
OCCUPIED_ABOVE = 0.5
ROBOT_RADIUS_CM = 16.0
FRONTIER_MIN_CELLS = 4
WAYPOINT_DISTANCE_CM = 40.0
# Cost of a frontier: path length plus turning, with left turns costing more
# so the car works round a room clockwise instead of dithering.
TURN_COST_PER_DEGREE = 0.4
LEFT_TURN_PENALTY = 25.0
FRONTIER_SIZE_REWARD = 1.5
GRID_MARGIN_CELLS = 20
# Keep heading for the same frontier unless another is clearly better, so a
# fresh sweep reshuffling the costs does not make the car flip-flop.
TARGET_STICK_RADIUS_CM = 35.0
# The forward sonar never sees right beside or behind the car, so unknown
# cells this close to anywhere it has stood are blind spots, not frontiers.
NEAR_FIELD_CM = 35.0
TARGET_SWITCH_MARGIN = 60.0


@dataclass(frozen=True)
class Frontier:
    centroid: tuple[float, float]
    target: tuple[float, float]
    cells: int
    path_cm: float


@dataclass(frozen=True)
class Waypoint:
    bearing_deg: float  # counterclockwise from the car's heading
    distance_cm: float
    frontier: Frontier


class CoverageGrid:
    """Log-odds occupancy over a grid that grows to fit what has been swept."""

    def __init__(self, cell_cm: float = CELL_CM) -> None:
        self.cell = cell_cm
        self.origin = np.array([0.0, 0.0])  # world position of cell (0, 0)'s corner
        self.log_odds = np.zeros((0, 0), dtype=np.float32)
        self.stood: list[tuple[float, float]] = []

    def _ensure(self, points: np.ndarray) -> None:
        low = points.min(axis=0)
        high = points.max(axis=0)
        if self.log_odds.size:
            current_low = self.origin
            current_high = self.origin + np.array(self.log_odds.shape[::-1]) * self.cell
            if (low >= current_low).all() and (high < current_high).all():
                return
            low = np.minimum(low, current_low)
            high = np.maximum(high, current_high)
        margin = GRID_MARGIN_CELLS * self.cell
        new_origin = np.floor((low - margin) / self.cell) * self.cell
        size = np.ceil((high + margin - new_origin) / self.cell).astype(int)
        grown = np.zeros((size[1], size[0]), dtype=np.float32)
        if self.log_odds.size:
            offset = np.round((self.origin - new_origin) / self.cell).astype(int)
            rows, columns = self.log_odds.shape
            grown[offset[1] : offset[1] + rows, offset[0] : offset[0] + columns] = self.log_odds
        self.origin, self.log_odds = new_origin, grown

    def cell_of(self, point: Sequence[float]) -> tuple[int, int]:
        """(row, column) of a world point."""
        column = int(math.floor((point[0] - self.origin[0]) / self.cell))
        row = int(math.floor((point[1] - self.origin[1]) / self.cell))
        return row, column

    def centre_of(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, column = cell
        return (
            float(self.origin[0] + (column + 0.5) * self.cell),
            float(self.origin[1] + (row + 0.5) * self.cell),
        )

    def integrate(self, scan: Sequence[Mapping[str, object]], pose: Sequence[float], beam_width_deg: float) -> None:
        origin = sonar_origin(pose)
        # (direction, distance cleared, distance of the echo or None)
        rays: list[tuple[np.ndarray, float, float | None]] = []
        for item in scan:
            distance = item.get("distance_cm")
            if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
                continue
            distance = float(distance)
            echo = distance if distance <= SONAR_MAX_RANGE_CM else None
            reach = min(distance, SONAR_MAX_RANGE_CM)
            # The beam's core clears space; only the centre line marks the echo.
            for offset in (-beam_width_deg / 4, 0.0, beam_width_deg / 4):
                ray = direction(float(pose[2]) + float(item["angle_deg"]) + offset)
                rays.append((ray, reach, echo if offset == 0.0 else None))
        footprint = np.array(pose[:2], dtype=float)
        ends = [origin + ray * (echo if echo is not None else reach) for ray, reach, echo in rays]
        ends += [footprint + ROBOT_RADIUS_CM, footprint - ROBOT_RADIUS_CM, origin]
        self._ensure(np.array(ends))
        free = np.zeros(self.log_odds.shape, dtype=bool)
        occupied = np.zeros(self.log_odds.shape, dtype=bool)
        for ray, reach, echo in rays:
            clear = reach - (5.0 + 0.03 * reach)
            for step in np.arange(0.0, max(clear, 0.0), self.cell / 2):
                row, column = self.cell_of(origin + ray * step)
                free[row, column] = True
            if echo is not None:
                row, column = self.cell_of(origin + ray * echo)
                occupied[row, column] = True
        self.stood.append((float(footprint[0]), float(footprint[1])))
        # The car is standing here, so its footprint is free.
        row, column = self.cell_of(footprint)
        radius = int(math.ceil(ROBOT_RADIUS_CM / self.cell))
        rows = slice(max(row - radius, 0), row + radius + 1)
        columns = slice(max(column - radius, 0), column + radius + 1)
        free[rows, columns] = True
        free &= ~occupied
        self.log_odds[free] += FREE_LOG_ODDS
        self.log_odds[occupied] += OCCUPIED_LOG_ODDS
        np.clip(self.log_odds, -LOG_ODDS_LIMIT, LOG_ODDS_LIMIT, out=self.log_odds)

    def free(self) -> np.ndarray:
        return self.log_odds < FREE_BELOW

    def occupied(self) -> np.ndarray:
        return self.log_odds > OCCUPIED_ABOVE

    def unknown(self) -> np.ndarray:
        return ~self.free() & ~self.occupied()

    def traversable(self) -> np.ndarray:
        """Free cells at least a car's radius from anything occupied."""
        radius = int(math.ceil(ROBOT_RADIUS_CM / self.cell))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        blocked = cv2.dilate(self.occupied().astype(np.uint8), kernel) > 0
        return self.free() & ~blocked

    def frontier_cells(self) -> np.ndarray:
        unknown = self.unknown()
        next_to_unknown = np.zeros_like(unknown)
        next_to_unknown[1:, :] |= unknown[:-1, :]
        next_to_unknown[:-1, :] |= unknown[1:, :]
        next_to_unknown[:, 1:] |= unknown[:, :-1]
        next_to_unknown[:, :-1] |= unknown[:, 1:]
        blind = np.zeros(unknown.shape, dtype=np.uint8)
        radius = int(round(NEAR_FIELD_CM / self.cell))
        for position in self.stood:
            row, column = self.cell_of(position)
            cv2.circle(blind, (column, row), radius, 1, -1)
        return self.free() & next_to_unknown & (blind == 0)

    def explored_fraction(self) -> float:
        known = ~self.unknown()
        return float(known.mean()) if known.size else 0.0

    def _distances(self, start: tuple[int, int]) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
        """Dijkstra over traversable cells from the car's cell."""
        traversable = self.traversable()
        rows, columns = traversable.shape
        distance = np.full(traversable.shape, np.inf)
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        if not (0 <= start[0] < rows and 0 <= start[1] < columns):
            return distance, parents
        distance[start] = 0.0
        queue = [(0.0, start)]
        steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)] + [
            (dr, dc, math.sqrt(2)) for dr in (-1, 1) for dc in (-1, 1)
        ]
        while queue:
            cost, (row, column) = heapq.heappop(queue)
            if cost > distance[row, column]:
                continue
            for dr, dc, length in steps:
                nr, nc = row + dr, column + dc
                if not (0 <= nr < rows and 0 <= nc < columns) or not traversable[nr, nc]:
                    continue
                candidate = cost + length * self.cell
                if candidate < distance[nr, nc]:
                    distance[nr, nc] = candidate
                    parents[(nr, nc)] = (row, column)
                    heapq.heappush(queue, (candidate, (nr, nc)))
        return distance, parents

    def frontiers(self, pose: Sequence[float]) -> tuple[list[Frontier], dict, tuple[int, int]]:
        if not self.log_odds.size:
            return [], {}, (0, 0)
        cells = self.frontier_cells()
        start = self.cell_of(pose[:2])
        distance, parents = self._distances(start)
        labels_count, labels = cv2.connectedComponents(cells.astype(np.uint8), connectivity=8)
        found: list[Frontier] = []
        for label in range(1, labels_count):
            members = np.argwhere(labels == label)
            if len(members) < FRONTIER_MIN_CELLS:
                continue
            reachable = [tuple(member) for member in members if math.isfinite(distance[tuple(member)])]
            if not reachable:
                continue
            target = min(reachable, key=lambda cell: distance[cell])
            centroid = self.centre_of(tuple(np.round(members.mean(axis=0)).astype(int)))
            found.append(Frontier(centroid, self.centre_of(target), len(members), float(distance[target])))
        return found, parents, start

    def next_waypoint(
        self,
        pose: Sequence[float],
        avoid_within_deg: float | None = None,
        previous_target: Sequence[float] | None = None,
    ) -> Waypoint | None:
        """The next point to head for on the way to the best frontier.

        With `avoid_within_deg`, frontiers lying within that angle of straight
        ahead are skipped (the sonar says the way ahead is blocked). A frontier
        near `previous_target` is kept unless another is clearly cheaper."""
        frontiers, parents, start = self.frontiers(pose)
        here = np.array(pose[:2], dtype=float)

        def turn_to(point: Sequence[float]) -> float:
            offset = np.array(point) - here
            bearing = math.degrees(math.atan2(-offset[0], offset[1]))
            return (bearing - float(pose[2]) + 180.0) % 360.0 - 180.0

        def cost(frontier: Frontier) -> float:
            turn = turn_to(frontier.target)
            return (
                frontier.path_cm
                + TURN_COST_PER_DEGREE * abs(turn)
                + (LEFT_TURN_PENALTY if turn > 0 else 0.0)
                - FRONTIER_SIZE_REWARD * min(frontier.cells, 40)
            )

        if avoid_within_deg is not None:
            frontiers = [frontier for frontier in frontiers if abs(turn_to(frontier.target)) > avoid_within_deg]
        if not frontiers:
            return None
        best = min(frontiers, key=cost)
        if previous_target is not None:
            kept = [
                frontier
                for frontier in frontiers
                if math.hypot(frontier.target[0] - previous_target[0], frontier.target[1] - previous_target[1])
                <= TARGET_STICK_RADIUS_CM
            ]
            if kept:
                same = min(kept, key=cost)
                if cost(same) <= cost(best) + TARGET_SWITCH_MARGIN:
                    best = same
        # Walk back along the path to the point about a waypoint's distance out.
        path = [self.cell_of(best.target)]
        while path[-1] != start and path[-1] in parents:
            path.append(parents[path[-1]])
        path.reverse()
        waypoint = self.centre_of(path[-1])
        for cell in path[1:]:
            point = self.centre_of(cell)
            if math.hypot(point[0] - here[0], point[1] - here[1]) >= WAYPOINT_DISTANCE_CM:
                waypoint = point
                break
        return Waypoint(turn_to(waypoint), math.hypot(waypoint[0] - here[0], waypoint[1] - here[1]), best)

    def to_image(self) -> np.ndarray:
        """Free dark teal, occupied light, unknown transparent (BGRA, north up)."""
        image = np.zeros((*self.log_odds.shape, 4), dtype=np.uint8)
        image[self.free()] = (70, 90, 30, 150)
        image[self.occupied()] = (215, 232, 236, 230)
        image[self.frontier_cells()] = (93, 179, 242, 255)  # gold (OpenCV order is B, G, R)
        return np.flipud(image)

    def bounds(self) -> list[float]:
        height, width = self.log_odds.shape
        return [
            float(self.origin[0]),
            float(self.origin[1]),
            float(self.origin[0] + width * self.cell),
            float(self.origin[1] + height * self.cell),
        ]
