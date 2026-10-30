"""Frontier exploration: a coverage grid from every sweep, and where to go next.

Every sweep marks the space the sound crossed as free and the cells where it
echoed as occupied; everything else stays unknown. Frontiers are free cells
next to unknown space: the edges of what has been mapped.

Frontiers are judged by what going there could show. Rays cast from each one
through the grid count the unknown floor it would bring into view, stopping
at walls: a long reading off to one side leaves a big unknown area past its
end and scores high, while a sliver in a corner scores almost nothing. Each
frontier is also classed as open space, a doorway or corridor (walls close on
both sides), or an ordinary edge. An ExplorationLearner turns that into a
choice, and learns from every run: how much each kind of frontier really
reveals compared with the prediction, and how much new floor a centimetre of
driving (or a degree of turning) tends to buy, which prices the effort of
getting there. Paths keep away from walls where there is room, so corridors
are followed down the middle. When no frontier is worth a trip, the area is
explored.
"""

import heapq
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from robocar.surfaces import SONAR_FORWARD_OFFSET_CM, SONAR_MAX_RANGE_CM, direction, sonar_origin

logger = logging.getLogger(__name__)

CELL_CM = 5.0
FREE_LOG_ODDS = -0.4
OCCUPIED_LOG_ODDS = 0.9
LOG_ODDS_LIMIT = 4.0
FREE_BELOW = -0.3
OCCUPIED_ABOVE = 0.5
# Walls confirmed by more than one sweep. Echoes from a doorway's edges land
# across the opening, but move as the car does, so they rarely build up.
CONFIRMED_ABOVE = 1.5
# Gaps in mapped walls narrower than this are closed when judging what a
# frontier could show: corners and glancing stretches rarely echo.
WALL_GAP_CLOSE_CM = 30.0
# Share of the view a wall cell seen only once lets through to gain rays.
TENTATIVE_WALL_TRANSPARENCY = 0.6
# A ray passing head-on through a mapped wall cell would have echoed there if
# the wall were real; such cells lose this much on top of ordinary free space.
CONTRADICTED_LOG_ODDS = -1.2
ROBOT_RADIUS_CM = 16.0
# A missing echo is open space, or sound glancing off a wall it met more than
# about this far from head-on and bouncing away. Runs of echoes in a sweep are
# taken as walls continuing this far past their ends; a missing echo that
# would cross one at a glancing angle stops there (a corridor's side walls),
# while one crossing nearly head-on would have echoed, so it is a real gap (a
# doorway). Missing echoes never clear space past a wall already on the map.
SPECULAR_DEGREES = 40.0
# A missing echo is a hint, not proof: it clears only this far, and the
# unknown left beyond makes a high-value frontier that draws the car closer.
NO_ECHO_FREE_CM = 50.0
WALL_EXTEND_CM = 150.0
RUN_JUMP_CM = 6.0
RUN_JUMP_FRACTION = 0.1
FRONTIER_MIN_CELLS = 4
# Long frontiers are judged in pieces this size: one running along a wall can
# include a doorway that only shows from beside it.
FRONTIER_PIECE_CM = 50.0
# The next waypoint is the furthest point along the path, up to this far,
# that the car can reach in a straight line.
LEG_MAX_CM = 60.0
GRID_MARGIN_CELLS = 20
# Keep heading for the same frontier unless another promises clearly more, so
# a fresh sweep reshuffling the numbers does not make the car flip-flop.
TARGET_STICK_RADIUS_CM = 80.0
TARGET_SWITCH_FACTOR = 1.5
# The forward sonar never sees right beside or behind the car, so unknown
# cells this close to anywhere it has stood are blind spots, not frontiers.
NEAR_FIELD_CM = 35.0
# The car only has to get this close to a frontier to look past it; edges
# often lie in the band beside walls it may not drive into.
FRONTIER_REACH_CM = 30.0
# Cells just behind an echo are looked at but stay unknown: the space behind
# a wall. Once looked at this often they are unseeable, not worth a trip.
LOOK_BEHIND_ECHO_CM = 30.0
UNSEEABLE_AFTER_LOOKS = 3
# What a frontier could reveal: rays every 3 degrees out to sonar range,
# across what a sweep sees (the sweep plus half the beam each side) when the
# car arrives facing the way it travelled.
GAIN_RAY_STEP_DEGREES = 3.0
GAIN_HALF_VIEW_DEGREES = 72.0
# Frontiers that could show less than this many cells (0.05 m2) are slivers
# and blind spots, not worth a trip.
GAIN_MIN_CELLS = 20.0
# A frontier this much could reveal counts as open space.
OPEN_GAIN_CELLS = 300.0
# Walls on both sides closer together than this make a doorway or corridor.
NARROW_OPENING_CM = 110.0
# Paths cost more within this distance of a wall, up to (1 + penalty) times.
CLEARANCE_COMFORT_CM = 45.0
CLEARANCE_PENALTY = 2.0
# A degree of turning costs as much effort as this much driving: turns are
# less accurate and each needs its own sweep.
TURN_EFFORT_CM_PER_DEGREE = 1.2
# Turning straight back after a turn undoes it; it costs this much extra
# effort, so the car only reverses when the other way is clearly better.
REVERSAL_EFFORT_CM = 120.0
# Every trip costs at least a sweep, worth about this much driving.
TRIP_OVERHEAD_CM = 60.0
# Exploring is over when the best frontier promises less than this share of
# the new floor per centimetre that moves have been buying on average.
STOP_FRACTION_OF_RATE = 0.05
KINDS = ("open", "doorway", "edge", "lead")
# Matching a sweep to the coverage map: echoes within MATCH_HIT_CM of mapped
# obstacles count as hits. A match is good when at least MATCH_GOOD_SCORE of
# the echoes landing on mapped ground are hits, and at least MATCH_MIN_HITS.
MATCH_HIT_CM = 10.0
MATCH_SPREAD_CM = 5.0
# Rays across the sonar's beam (it answers with the nearest thing anywhere in it).
MATCH_BEAM_RAYS_DEG = (-9.0, -4.5, 0.0, 4.5, 9.0)
MATCH_STEP_CM = 4.0
MATCH_STEP_DEG = 2.0
MATCH_MIN_POINTS = 10
MATCH_MIN_HITS = 12
MATCH_GOOD_SCORE = 0.6
# ...and only when nothing farther than MATCH_UNIQUE_CM / MATCH_UNIQUE_DEG
# from it scores within MATCH_MIN_MARGIN of it.
MATCH_UNIQUE_CM = 10.0
MATCH_UNIQUE_DEG = 6.0
MATCH_MIN_MARGIN = 0.015
# Leads: openings seen directly in a sweep, where readings run at least
# LEAD_DEEPER_CM deeper than on both sides (or do not echo) for at least
# LEAD_MIN_DEGREES. Each is worth going to until the floor LEAD_BEHIND_CM
# past its mouth is mapped, or the car has stood within LEAD_VISITED_CM of it.
LEAD_DEEPER_CM = 25.0
LEAD_SIDE_DEGREES = 18.0
LEAD_MIN_DEGREES = 6.0
LEAD_BEHIND_CM = 60.0
LEAD_VISITED_CM = 35.0
LEAD_MIN_GAIN_CELLS = 400.0
LEAD_MERGE_CM = 40.0
# Corridors: walls on both sides, roughly parallel, within this width, seen
# in mapped walls within CORRIDOR_LOOK_CM of the car.
CORRIDOR_LOOK_CM = 100.0
CORRIDOR_WIDTH_CM = (40.0, 130.0)
CORRIDOR_MIN_WALL_CELLS = 4
CORRIDOR_STRAIGHTNESS = 6.0


@dataclass(frozen=True)
class Frontier:
    centroid: tuple[float, float]
    target: tuple[float, float]
    cells: int
    path_cm: float
    gain: float = 0.0  # unknown cells it could bring into view
    kind: str = "edge"  # "open", "doorway" or "edge"
    width_cm: float | None = None  # across a doorway or corridor
    middle: tuple[float, float] | None = None  # the part of the frontier to look at


@dataclass(frozen=True)
class ScanMatch:
    """Where a sweep fits the coverage map best, and how well."""

    pose: tuple[float, float, float]
    score: float  # share of judged echoes landing on something mapped, by closeness
    hits: int
    judged: int  # hits plus echoes landing on floor mapped as empty
    prior_score: float  # the same share at the pose it started from
    margin: float  # how much better it fits than the best pose well away from it

    @property
    def good(self) -> bool:
        return self.margin >= MATCH_MIN_MARGIN and self.score >= MATCH_GOOD_SCORE and self.hits >= MATCH_MIN_HITS

    def summary(self) -> dict[str, object]:
        return {
            "pose": [round(value, 1) for value in self.pose],
            "score": round(self.score, 2),
            "hits": self.hits,
            "judged": self.judged,
            "prior_score": round(self.prior_score, 2),
            "margin": round(self.margin, 3),
            "good": self.good,
        }


@dataclass(frozen=True)
class Corridor:
    axis_deg: float  # heading along the corridor (either way along it)
    centre_offset_cm: float  # how far the centre line is to the left (+) of the car
    width_cm: float


@dataclass(frozen=True)
class Waypoint:
    bearing_deg: float  # counterclockwise from the car's heading
    distance_cm: float
    frontier: Frontier


# Learning limits: how far a kind's weight and the effort price may move.
WEIGHT_LIMITS = (0.15, 2.0)
# A pursuit counts as arrived within this much of reaching distance.
ARRIVAL_SLACK_CM = 20.0
RATE_LIMITS_CELLS_PER_CM = (0.5, 40.0)
LEARNING_RATE = 0.3


@dataclass
class ExplorationLearner:
    """Chooses frontiers by expected new floor minus the effort to get there,
    and learns both from what actually happens.

    `weights[kind]` is the share of a frontier's predicted gain that pursuing
    that kind really revealed (rays are optimistic, and doorways often lead
    to whole new rooms). A frontier's value is that expected new floor per
    centimetre of effort to get there. `rate` is the new floor a centimetre
    of effort has actually been buying; when even the best frontier promises
    only a small share of that, exploring has stopped paying.
    """

    weights: dict[str, float] = field(default_factory=lambda: {"open": 0.6, "doorway": 0.8, "edge": 0.4, "lead": 0.8})
    rate: float = 4.0
    pursuits: int = 0
    # The turn just made, if the car has not driven since.
    last_turn: float = field(default=0.0, repr=False)
    _last: tuple[tuple[float, float, float], int] | None = field(default=None, repr=False)
    # (kind, predicted gain, known cells at the start, target)
    _pursuit: tuple[str, float, int, tuple[float, float]] | None = field(default=None, repr=False)

    def expected(self, frontier: Frontier) -> float:
        return self.weights.get(frontier.kind, 0.5) * frontier.gain

    def utility(self, frontier: Frontier, turn_deg: float) -> float:
        """Expected new floor (cells) per centimetre of effort."""
        effort = TRIP_OVERHEAD_CM + frontier.path_cm + TURN_EFFORT_CM_PER_DEGREE * abs(turn_deg)
        if self.last_turn * turn_deg < 0 and abs(turn_deg) > 15.0:
            effort += REVERSAL_EFFORT_CM
        return self.expected(frontier) / effort

    def worth_it(self, value: float) -> bool:
        return value >= STOP_FRACTION_OF_RATE * self.rate

    def observe(self, pose: Sequence[float], known_cells: int) -> None:
        """After each sweep: how much new floor the last move bought."""
        if self._last is not None:
            (x, y, heading), known = self._last
            effort = math.hypot(pose[0] - x, pose[1] - y) + TURN_EFFORT_CM_PER_DEGREE * abs(
                (float(pose[2]) - heading + 180.0) % 360.0 - 180.0
            )
            if effort > 2.0:
                bought = max(0, known_cells - known) / effort
                low, high = RATE_LIMITS_CELLS_PER_CM
                self.rate = min(high, max(low, (1 - LEARNING_RATE) * self.rate + LEARNING_RATE * bought))
        self._last = ((float(pose[0]), float(pose[1]), float(pose[2])), known_cells)

    def pursue(self, frontier: Frontier, known_cells: int) -> None:
        """Note the frontier chosen; finishing one pursuit scores its kind."""
        if self._pursuit is not None:
            target = self._pursuit[3]
            if math.hypot(frontier.target[0] - target[0], frontier.target[1] - target[1]) <= TARGET_STICK_RADIUS_CM:
                return
            self.finish(known_cells)
        self._pursuit = (frontier.kind, frontier.gain, known_cells, frontier.target)

    def finish(self, known_cells: int) -> None:
        """Score a pursuit's kind, but only if the car got there: one given
        up halfway says nothing about what that kind of frontier shows."""
        if self._pursuit is None:
            return
        kind, predicted, start, target = self._pursuit
        self._pursuit = None
        if predicted < GAIN_MIN_CELLS or self._last is None:
            return
        (x, y, _), _ = self._last
        if math.hypot(target[0] - x, target[1] - y) > FRONTIER_REACH_CM + ARRIVAL_SLACK_CM:
            return
        ratio = max(0, known_cells - start) / predicted
        low, high = WEIGHT_LIMITS
        current = self.weights.get(kind, 0.5)
        self.weights[kind] = min(high, max(low, (1 - LEARNING_RATE) * current + LEARNING_RATE * ratio))
        self.pursuits += 1

    def start_run(self) -> None:
        self._last = None
        self._pursuit = None
        self.last_turn = 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "weights": {kind: round(value, 3) for kind, value in self.weights.items()},
            "rate_cells_per_cm": round(self.rate, 3),
            "pursuits": self.pursuits,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "ExplorationLearner":
        learner = cls()
        weights = data.get("weights")
        if isinstance(weights, Mapping):
            for kind in KINDS:
                value = weights.get(kind)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    learner.weights[kind] = min(WEIGHT_LIMITS[1], max(WEIGHT_LIMITS[0], float(value)))
        rate = data.get("rate_cells_per_cm")
        if isinstance(rate, (int, float)) and math.isfinite(rate):
            learner.rate = min(RATE_LIMITS_CELLS_PER_CM[1], max(RATE_LIMITS_CELLS_PER_CM[0], float(rate)))
        pursuits = data.get("pursuits")
        if isinstance(pursuits, int):
            learner.pursuits = pursuits
        return learner


def _echo_runs(
    readings: Sequence[tuple[float, float]], pose: Sequence[float], origin: np.ndarray, beam_width_deg: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walls implied by runs of neighbouring echoes at similar ranges, each
    extended WALL_EXTEND_CM past its ends, as (start, end) segments.

    A wide beam reports the nearest point of a wall anywhere in it, so the
    echoes do not lie on the wall along their centre lines. Each run is fitted
    as the beam sees a flat wall instead: at sweep angle a, a wall at
    distance d whose normal points at angle n reads d / cos(max(0, |a - n| - half
    the beam width))."""
    echoes = sorted((angle, distance) for angle, distance in readings if distance <= SONAR_MAX_RANGE_CM)
    angles = sorted(angle for angle, _ in readings)
    step = min((b - a for a, b in zip(angles, angles[1:]) if b > a), default=3.0)
    half = beam_width_deg / 2
    runs: list[list[tuple[float, float]]] = []
    for angle, distance in echoes:
        if runs:
            last_angle, last_distance = runs[-1][-1]
            if angle - last_angle <= 1.5 * step and abs(distance - last_distance) <= RUN_JUMP_CM + RUN_JUMP_FRACTION * min(
                distance, last_distance
            ):
                runs[-1].append((angle, distance))
                continue
        runs.append([(angle, distance)])
    segments = []
    for run in runs:
        if len(run) < 2:
            continue
        run_angles = np.array([angle for angle, _ in run])
        ranges = np.array([distance for _, distance in run])
        best = None
        for normal in np.arange(run_angles.min() - 60.0, run_angles.max() + 60.0 + 1e-9, 1.0):
            factor = np.cos(np.radians(np.maximum(0.0, np.abs(run_angles - normal) - half)))
            distance = float(np.median(ranges * factor))
            error = float(np.sum((ranges - distance / factor) ** 2))
            if best is None or error < best[0]:
                best = (error, normal, distance)
        _, normal, distance = best
        facing = direction(float(pose[2]) + normal)
        foot = origin + facing * distance
        along = np.array([-facing[1], facing[0]])
        # The stretch of wall the run saw: each reading reflects where the
        # beam's edge nearest the perpendicular meets the wall.
        offsets = run_angles - normal
        beyond = np.radians(offsets - np.clip(offsets, -half, half))
        spans = distance * np.tan(beyond)
        low, high = float(spans.min()), float(spans.max())
        segments.append((foot + along * (low - WALL_EXTEND_CM), foot + along * (high + WALL_EXTEND_CM)))
    return segments


def _openings(
    readings: Sequence[tuple[float, float]], pose: Sequence[float], origin: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Openings in a sweep: stretches reading clearly deeper than readings on
    both sides within LEAD_SIDE_DEGREES (the beam smears an opening's edges,
    so they deepen gradually). Returns each opening's mouth (at the nearer
    sides' distance, in the middle of the stretch) and the direction into it."""
    ordered = sorted(readings)
    depth = [min(distance, SONAR_MAX_RANGE_CM + 50.0) for _, distance in ordered]
    angles = [angle for angle, _ in ordered]
    mouths: list[float | None] = []
    for index, (angle, here) in enumerate(zip(angles, depth)):
        left = [d for a, d in zip(angles, depth) if angle - LEAD_SIDE_DEGREES <= a < angle]
        right = [d for a, d in zip(angles, depth) if angle < a <= angle + LEAD_SIDE_DEGREES]
        if left and right and here >= min(left) + LEAD_DEEPER_CM and here >= min(right) + LEAD_DEEPER_CM:
            mouths.append(max(min(left), min(right)))
        else:
            mouths.append(None)
    found = []
    index = 0
    while index < len(ordered):
        if mouths[index] is None:
            index += 1
            continue
        end = index
        while end + 1 < len(ordered) and mouths[end + 1] is not None:
            end += 1
        if angles[end] - angles[index] >= LEAD_MIN_DEGREES - 1e-6:
            middle = (index + end) // 2
            into = direction(float(pose[2]) + (angles[index] + angles[end]) / 2)
            found.append((origin + into * float(mouths[middle]), into))
        index = end + 1
    return found


def _glancing_stop(
    origin: np.ndarray, ray: np.ndarray, reach: float, walls: Sequence[tuple[np.ndarray, np.ndarray]]
) -> float:
    """How far a missing echo can be trusted: up to the first implied wall it
    meets at a glancing angle, which would have bounced the sound away."""
    limit = math.cos(math.radians(SPECULAR_DEGREES))
    for start, end in walls:
        edge = end - start
        matrix = np.column_stack([ray, -edge])
        if abs(np.linalg.det(matrix)) < 1e-9:
            continue
        distance, fraction = np.linalg.solve(matrix, start - origin)
        if not (0 < distance < reach and 0 <= fraction <= 1):
            continue
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        if abs(float(ray @ normal)) < limit:
            reach = max(0.0, distance - 5.0)
    return reach


def load_learner(path: Path | None) -> ExplorationLearner:
    """What earlier runs learned, from a JSON file beside the map database."""
    if path is None or not path.exists():
        return ExplorationLearner()
    try:
        return ExplorationLearner.from_json(json.loads(path.read_text()))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Ignoring unreadable exploration learning %s: %s", path, exc)
        return ExplorationLearner()


def save_learner(path: Path | None, learner: ExplorationLearner) -> None:
    if path is None:
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(learner.to_json(), indent=1))
    temporary.replace(path)


class CoverageGrid:
    """Log-odds occupancy over a grid that grows to fit what has been swept."""

    def __init__(self, cell_cm: float = CELL_CM) -> None:
        self.cell = cell_cm
        self.origin = np.array([0.0, 0.0])  # world position of cell (0, 0)'s corner
        self.log_odds = np.zeros((0, 0), dtype=np.float32)
        self.stood: list[tuple[float, float]] = []
        self.looks = np.zeros((0, 0), dtype=np.uint8)
        # Openings seen in sweeps: (mouth point, direction into it).
        self.leads: list[tuple[np.ndarray, np.ndarray]] = []

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
        looks = np.zeros((size[1], size[0]), dtype=np.uint8)
        if self.log_odds.size:
            offset = np.round((self.origin - new_origin) / self.cell).astype(int)
            rows, columns = self.log_odds.shape
            grown[offset[1] : offset[1] + rows, offset[0] : offset[0] + columns] = self.log_odds
            looks[offset[1] : offset[1] + rows, offset[0] : offset[0] + columns] = self.looks
        self.origin, self.log_odds, self.looks = new_origin, grown, looks

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
        readings: list[tuple[float, float]] = []
        for item in scan:
            distance = item.get("distance_cm")
            if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
                continue
            readings.append((float(item["angle_deg"]), float(distance)))
        walls_seen = _echo_runs(readings, pose, origin, beam_width_deg)
        for mouth, into in _openings(readings, pose, origin):
            if all(np.linalg.norm(mouth - other) > LEAD_MERGE_CM for other, _ in self.leads):
                self.leads.append((mouth, into))
        # (direction, distance cleared, distance of the echo or None)
        rays: list[tuple[np.ndarray, float, float | None]] = []
        for angle, distance in readings:
            echo = distance if distance <= SONAR_MAX_RANGE_CM else None
            reach = min(distance, SONAR_MAX_RANGE_CM)
            # The beam's core clears space; only the centre line marks the echo.
            for offset in (-beam_width_deg / 4, 0.0, beam_width_deg / 4):
                ray = direction(float(pose[2]) + angle + offset)
                ray_reach = _glancing_stop(
                    origin, ray, reach if echo is not None else min(reach, NO_ECHO_FREE_CM), walls_seen
                )
                rays.append((ray, ray_reach, echo if offset == 0.0 else None))
        footprint = np.array(pose[:2], dtype=float)
        ends = [origin + ray * (echo + LOOK_BEHIND_ECHO_CM if echo is not None else reach) for ray, reach, echo in rays]
        ends += [footprint + ROBOT_RADIUS_CM, footprint - ROBOT_RADIUS_CM, origin]
        self._ensure(np.array(ends))
        free = np.zeros(self.log_odds.shape, dtype=bool)
        occupied = np.zeros(self.log_odds.shape, dtype=bool)
        looked = np.zeros(self.log_odds.shape, dtype=bool)
        for ray, reach, echo in rays:
            if echo is not None:
                row, column = self.cell_of(origin + ray * echo)
                occupied[row, column] = True
                for step in np.arange(echo + self.cell, echo + LOOK_BEHIND_ECHO_CM, self.cell / 2):
                    looked[self.cell_of(origin + ray * step)] = True
        # Walls known before or found in this sweep, thickened so a ray cannot
        # slip between diagonal cells. A ray meeting one at a glancing angle
        # stops (the wall would have bounced the sound away silently); one
        # meeting it head-on would have echoed there, so that wall cell is
        # misplaced and the ray carries on.
        wall_cells = self.occupied() | occupied
        walls = cv2.dilate(wall_cells.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        contradicted = np.zeros(self.log_odds.shape, dtype=bool)
        for ray, reach, echo in rays:
            clear = reach - (5.0 + 0.03 * reach)
            for step in np.arange(0.0, max(clear, 0.0), self.cell / 2):
                row, column = self.cell_of(origin + ray * step)
                if step > ROBOT_RADIUS_CM and walls[row, column]:
                    if self._glancing(wall_cells, (row, column), ray, echo is None):
                        break
                    if wall_cells[row, column]:
                        contradicted[row, column] = True
                free[row, column] = True
        self.stood.append((float(footprint[0]), float(footprint[1])))
        # The car is standing here, so its footprint is free.
        row, column = self.cell_of(footprint)
        radius = int(math.ceil(ROBOT_RADIUS_CM / self.cell))
        rows = slice(max(row - radius, 0), row + radius + 1)
        columns = slice(max(column - radius, 0), column + radius + 1)
        free[rows, columns] = True
        free &= ~occupied
        looked &= ~free & ~occupied
        self.looks[looked] = np.minimum(self.looks[looked] + 1, 255)
        self.log_odds[free] += FREE_LOG_ODDS
        self.log_odds[contradicted & ~occupied] += CONTRADICTED_LOG_ODDS
        self.log_odds[occupied] += OCCUPIED_LOG_ODDS
        np.clip(self.log_odds, -LOG_ODDS_LIMIT, LOG_ODDS_LIMIT, out=self.log_odds)

    def _glancing(self, wall_cells: np.ndarray, cell: tuple[int, int], ray: np.ndarray, no_echo: bool) -> bool:
        """Whether a ray meets the wall around a cell at a glancing angle. With
        too few wall cells to tell its direction, a missing echo is stopped
        (it may be glancing) and a real echo carries on."""
        row, column = cell
        window = 3
        rows = slice(max(row - window, 0), row + window + 1)
        columns = slice(max(column - window, 0), column + window + 1)
        points = np.argwhere(wall_cells[rows, columns])
        if len(points) < 3:
            return no_echo
        xy = points[:, ::-1].astype(float)
        centred = xy - xy.mean(axis=0)
        values, axes = np.linalg.eigh(centred.T @ centred)
        if values[1] < 2.0 * max(values[0], 1e-6):
            return no_echo  # a blob, not a line
        along = axes[:, 1]
        normal = np.array([-along[1], along[0]])
        return abs(float(ray @ normal)) < math.cos(math.radians(SPECULAR_DEGREES))

    def free(self) -> np.ndarray:
        return self.log_odds < FREE_BELOW

    def occupied(self) -> np.ndarray:
        return self.log_odds > OCCUPIED_ABOVE

    def unknown(self) -> np.ndarray:
        return ~self.free() & ~self.occupied()

    def worth_seeing(self) -> np.ndarray:
        return self.unknown() & (self.looks < UNSEEABLE_AFTER_LOOKS)

    def traversable(self) -> np.ndarray:
        """Free cells at least a car's radius from anything occupied."""
        radius = int(math.ceil(ROBOT_RADIUS_CM / self.cell))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        blocked = cv2.dilate(self.occupied().astype(np.uint8), kernel) > 0
        return self.free() & ~blocked

    def frontier_cells(self) -> np.ndarray:
        unknown = self.worth_seeing()
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

    def known_count(self) -> int:
        return int((~self.unknown()).sum()) if self.log_odds.size else 0

    def _blocked(self) -> np.ndarray:
        """How much of the view each cell lets through, for gain rays: 0 at
        confirmed walls (gaps narrower than a doorway closed, since corners
        rarely echo), TENTATIVE_WALL_TRANSPARENCY at walls seen only once
        (they may be stray echoes from a doorway's edges), 1 elsewhere. Walls
        are thickened by one so rays cannot slip between diagonal cells."""
        size = 2 * int(round(WALL_GAP_CLOSE_CM / self.cell / 2)) + 1
        closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        confirmed = cv2.morphologyEx((self.log_odds > CONFIRMED_ABOVE).astype(np.uint8), cv2.MORPH_CLOSE, closing)
        confirmed = cv2.dilate(confirmed, np.ones((3, 3), np.uint8)) > 0
        tentative = cv2.dilate(self.occupied().astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        transparency = np.ones(self.log_odds.shape, dtype=np.float32)
        transparency[tentative] = TENTATIVE_WALL_TRANSPARENCY
        transparency[confirmed] = 0.0
        return transparency

    def information_gain(
        self, point: Sequence[float], heading_deg: float, unknown: np.ndarray, blocked: np.ndarray
    ) -> float:
        """Unknown cells a sweep facing `heading_deg` from a point would see.

        Rays stop at walls; space off the grid has never been seen, so it
        counts as unknown. Each ray sample stands for the arc of floor around
        it, so the sum approximates an area in cells."""
        step = math.radians(GAIN_RAY_STEP_DEGREES)
        half = math.radians(GAIN_HALF_VIEW_DEGREES)
        angles = math.radians(heading_deg) + np.arange(-half, half + 1e-9, step)
        steps = np.arange(self.cell, SONAR_MAX_RANGE_CM, self.cell)
        xs = point[0] + np.outer(-np.sin(angles), steps)
        ys = point[1] + np.outer(np.cos(angles), steps)
        columns = np.floor((xs - self.origin[0]) / self.cell).astype(int)
        rows = np.floor((ys - self.origin[1]) / self.cell).astype(int)
        inside = (rows >= 0) & (rows < unknown.shape[0]) & (columns >= 0) & (columns < unknown.shape[1])
        safe_rows, safe_columns = np.clip(rows, 0, unknown.shape[0] - 1), np.clip(columns, 0, unknown.shape[1] - 1)
        through = np.where(inside, blocked[safe_rows, safe_columns], 1.0)
        hidden = np.where(inside, unknown[safe_rows, safe_columns], True)
        # Each sample sees past everything before it, wall by wall.
        visible = np.cumprod(np.concatenate([np.ones((len(angles), 1)), through[:, :-1]], axis=1), axis=1)
        arc_cells = steps * step / self.cell
        return float((hidden * visible * arc_cells).sum())

    def opening_width(self, point: Sequence[float], unknown: np.ndarray, occupied: np.ndarray) -> float | None:
        """Width between walls across the way into the unknown at a frontier,
        or None when there is no wall close on both sides."""
        row, column = self.cell_of(point)
        reach = int(round(30.0 / self.cell))
        rows = slice(max(row - reach, 0), row + reach + 1)
        columns = slice(max(column - reach, 0), column + reach + 1)
        hidden = np.argwhere(unknown[rows, columns])
        if not len(hidden):
            return None
        offsets = hidden + [rows.start, columns.start] - [row, column]
        into = offsets.mean(axis=0)[::-1]  # (row, column) -> (x, y)
        length = float(np.hypot(*into))
        if length < 1e-6:
            return None
        across = np.array([-into[1], into[0]]) / length
        width = 0.0
        for side in (1.0, -1.0):
            hit = None
            for distance in np.arange(self.cell, 150.0, self.cell):
                cell = self.cell_of((point[0] + side * across[0] * distance, point[1] + side * across[1] * distance))
                if not (0 <= cell[0] < occupied.shape[0] and 0 <= cell[1] < occupied.shape[1]):
                    break
                if occupied[cell]:
                    hit = distance
                    break
            if hit is None:
                return None
            width += hit
        return width

    def corridor(self, pose: Sequence[float]) -> Corridor | None:
        """The corridor the car is in, if mapped walls close by run in two
        roughly parallel lines either side of it."""
        if not self.log_odds.size:
            return None
        here = np.array(pose[:2], dtype=float)
        reach = int(math.ceil(CORRIDOR_LOOK_CM / self.cell))
        row, column = self.cell_of(here)
        rows = slice(max(row - reach, 0), row + reach + 1)
        columns = slice(max(column - reach, 0), column + reach + 1)
        cells = np.argwhere(self.occupied()[rows, columns])
        if len(cells) < 2 * CORRIDOR_MIN_WALL_CELLS:
            return None
        points = np.array([self.centre_of((r + rows.start, c + columns.start)) for r, c in cells])
        points = points[np.hypot(*(points - here).T) <= CORRIDOR_LOOK_CM]
        if len(points) < 2 * CORRIDOR_MIN_WALL_CELLS:
            return None
        offsets = points - here
        # The axis is the direction the walls share. With short stretches of
        # wall, all the points together can spread as much across as along,
        # so try both directions and keep the one giving two straight sides.
        _, axes = np.linalg.eigh(np.cov(offsets.T))
        best = None
        for along in (axes[:, 1], axes[:, 0]):
            across = np.array([-along[1], along[0]])
            side = offsets @ across
            left, right = offsets[side > 0], offsets[side < 0]
            if len(left) < CORRIDOR_MIN_WALL_CELLS or len(right) < CORRIDOR_MIN_WALL_CELLS:
                continue
            # Fit the direction about each side's own centre, so the gap
            # between the walls does not count as spread.
            spread = np.vstack([left - left.mean(axis=0), right - right.mean(axis=0)])
            values, side_axes = np.linalg.eigh(spread.T @ spread)
            straightness = values[1] / max(values[0], 1e-6)
            refined = side_axes[:, 1]
            if straightness >= CORRIDOR_STRAIGHTNESS and abs(float(refined @ along)) > math.cos(math.radians(30.0)):
                if best is None or straightness > best[0]:
                    best = (straightness, refined, left, right)
        if best is None:
            return None
        _, along, left, right = best
        across = np.array([-along[1], along[0]])
        left_offset = float(np.median(left @ across))
        right_offset = float(np.median(right @ across))
        if left_offset * right_offset >= 0:
            return None
        width = abs(left_offset - right_offset)
        if not CORRIDOR_WIDTH_CM[0] <= width <= CORRIDOR_WIDTH_CM[1]:
            return None
        axis = math.degrees(math.atan2(-along[0], along[1]))
        centre = (left_offset + right_offset) / 2
        # Express the centre offset relative to the car facing along the axis.
        if abs((axis - float(pose[2]) + 180.0) % 360.0 - 180.0) > 90.0:
            axis += 180.0
            centre = -centre
        return Corridor(axis, centre, width)

    def clear_run(self, pose: Sequence[float], limit_cm: float, known_only: bool = False) -> float:
        """How far the car can drive straight ahead before coming within its
        radius of a mapped wall. Walls met at a glancing angle never echo, so
        the sonar alone cannot be trusted for this. Starting already inside
        that margin, it may drive only where it gets no nearer to a wall.

        With `known_only`, floor the map has never seen counts as a wall too:
        for moves no sensor checks first (reversing, sliding sideways)."""
        if not self.log_odds.size:
            return 0.0 if known_only else limit_cm
        # Every mapped wall counts here, confirmed or not: safety first.
        blocked = ~self.free() if known_only else self.occupied()
        clearance = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 5) * self.cell
        heading = direction(float(pose[2]))
        start = np.array(pose[:2], dtype=float)
        first = self.cell_of(start)
        floor_cm = min(ROBOT_RADIUS_CM, float(clearance[first])) if self._inside(first) else ROBOT_RADIUS_CM
        travelled = 0.0
        for step in np.arange(self.cell / 2, limit_cm + 1e-9, self.cell / 2):
            cell = self.cell_of(start + heading * step)
            if not self._inside(cell) or clearance[cell] < floor_cm - 1e-6:
                break
            travelled = step
        return travelled

    def match_scan(
        self,
        scan: Sequence[Mapping[str, object]],
        pose: Sequence[float],
        search_cm: float,
        search_deg: float,
    ) -> "ScanMatch | None":
        """Where a sweep fits the map best, near `pose`.

        Tries shifts within search_cm and turns within search_deg. An echo
        landing within MATCH_HIT_CM of something already mapped is a hit, one
        landing on floor the map knows is empty a miss, and one ending in
        unexplored space says nothing. The score is the share of hits among
        hits and misses, each hit weighted by how close it lands (so the exact
        fit beats one merely near). Unlike line registration this needs no
        flat walls: clutter, furniture and corners all count. None when too
        few echoes land on mapped ground to judge."""
        if not self.log_odds.size:
            return None
        readings = [
            (float(item["angle_deg"]), float(value))
            for item in scan
            if isinstance(value := item.get("distance_cm"), (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 3.0 <= value <= SONAR_MAX_RANGE_CM
        ]
        if len(readings) < MATCH_MIN_POINTS:
            return None
        angles = np.radians([angle for angle, _ in readings])
        ranges = np.array([distance for _, distance in readings])
        occupied = self.occupied()
        gap = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * self.cell
        near = gap <= MATCH_HIT_CM
        closeness = np.exp(-0.5 * (gap / MATCH_SPREAD_CM) ** 2) * near
        free = self.free()
        rows, columns = occupied.shape
        offsets = np.arange(-search_cm, search_cm + 1e-9, MATCH_STEP_CM)
        shift_x, shift_y = (grid.ravel() for grid in np.meshgrid(offsets, offsets))
        turns = np.arange(-search_deg, search_deg + 1e-9, MATCH_STEP_DEG)
        beam = np.radians(MATCH_BEAM_RAYS_DEG)
        scores = np.zeros((len(turns), len(shift_x)))
        hit_counts = np.zeros_like(scores, dtype=int)
        judged_counts = np.zeros_like(scores, dtype=int)
        for row, turn in enumerate(turns):
            heading = math.radians(float(pose[2]) + turn)
            forward = np.array([-math.sin(heading), math.cos(heading)])
            origin = np.array(pose[:2], dtype=float) + forward * SONAR_FORWARD_OFFSET_CM
            # The echo came from somewhere across the beam: try rays spanning it.
            rays = heading + angles[None, :] + beam[:, None]  # B x N
            ends_x = origin[0] - np.sin(rays) * ranges[None, :]
            ends_y = origin[1] + np.cos(rays) * ranges[None, :]
            xs = ends_x[None] + shift_x[:, None, None]  # S x B x N
            ys = ends_y[None] + shift_y[:, None, None]
            cols = np.floor((xs - self.origin[0]) / self.cell).astype(int)
            rws = np.floor((ys - self.origin[1]) / self.cell).astype(int)
            inside = (rws >= 0) & (rws < rows) & (cols >= 0) & (cols < columns)
            rws_c, cols_c = np.clip(rws, 0, rows - 1), np.clip(cols, 0, columns - 1)
            hit = (near[rws_c, cols_c] & inside).any(axis=1)  # S x N
            miss = (free[rws_c, cols_c] & inside).all(axis=1) & ~hit
            hits = hit.sum(axis=1)
            judged = hits + miss.sum(axis=1)
            weight = (closeness[rws_c, cols_c] * inside).max(axis=1).sum(axis=1)
            scores[row] = np.where(judged >= MATCH_MIN_POINTS, weight / np.maximum(judged, 1), 0.0)
            hit_counts[row], judged_counts[row] = hits, judged
        # Prefer the smallest correction among equally good fits.
        shift = np.hypot(shift_x, shift_y)[None, :]
        penalty = 0.002 * (shift / MATCH_STEP_CM + np.abs(turns)[:, None] / MATCH_STEP_DEG)
        row, column = np.unravel_index(int(np.argmax(scores - penalty)), scores.shape)
        if judged_counts[row, column] < MATCH_MIN_POINTS:
            return None
        score = float(scores[row, column])
        # Ambiguous when a pose well away from the best fits nearly as well:
        # a sweep of one long wall pins the distance to it, not where along it.
        far = (
            np.hypot(shift_x[None, :] - shift_x[column], shift_y[None, :] - shift_y[column]) > MATCH_UNIQUE_CM
        ) | (np.abs(turns - turns[row])[:, None] > MATCH_UNIQUE_DEG)
        centre = int(np.argmin(shift[0]))
        zero_turn = int(np.argmin(np.abs(turns)))
        return ScanMatch(
            (float(pose[0]) + float(shift_x[column]), float(pose[1]) + float(shift_y[column]), float(pose[2]) + float(turns[row])),
            score,
            int(hit_counts[row, column]),
            int(judged_counts[row, column]),
            float(scores[zero_turn, centre]),
            score - float(scores[far].max(initial=0.0)),
        )

    def wall_between(self, start: Sequence[float], end: Sequence[float]) -> bool:
        """Whether a confirmed wall lies on the straight line between two points."""
        if not self.log_odds.size:
            return False
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        for fraction in np.linspace(0.0, 1.0, max(2, int(length / (self.cell / 2)) + 1)):
            cell = self.cell_of((start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])))
            if self._inside(cell) and self.log_odds[cell] > CONFIRMED_ABOVE:
                return True
        return False

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.log_odds.shape[0] and 0 <= cell[1] < self.log_odds.shape[1]

    def line_of_sight(self, start: Sequence[float], end: Sequence[float], traversable: np.ndarray) -> bool:
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        for fraction in np.linspace(0.0, 1.0, max(2, int(length / (self.cell / 2)) + 1)):
            cell = self.cell_of((start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])))
            if not (0 <= cell[0] < traversable.shape[0] and 0 <= cell[1] < traversable.shape[1]):
                return False
            if not traversable[cell]:
                return False
        return True

    def explored_fraction(self) -> float:
        known = ~self.unknown()
        return float(known.mean()) if known.size else 0.0

    def _distances(self, start: tuple[int, int]) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
        """Dijkstra over traversable cells from the car's cell."""
        traversable = self.traversable()
        rows, columns = traversable.shape
        # The car may have stopped inside the margin kept from walls; let it
        # drive out through free cells close by rather than be boxed in.
        if 0 <= start[0] < rows and 0 <= start[1] < columns:
            escape = int(math.ceil((ROBOT_RADIUS_CM + FRONTIER_REACH_CM) / self.cell))
            nearby = np.zeros(traversable.shape, dtype=np.uint8)
            cv2.circle(nearby, (start[1], start[0]), escape, 1, -1)
            traversable = traversable | (self.free() & (nearby > 0))
            traversable[start] = True
        # Steps near walls cost more, so paths run down the middle of corridors.
        clearance = cv2.distanceTransform((~self.occupied()).astype(np.uint8), cv2.DIST_L2, 5) * self.cell
        crowding = 1.0 + CLEARANCE_PENALTY * np.clip((CLEARANCE_COMFORT_CM - clearance) / CLEARANCE_COMFORT_CM, 0.0, 1.0)
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
                candidate = cost + length * self.cell * crowding[nr, nc]
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
        unknown, occupied, blocked = self.worth_seeing(), self.occupied(), self._blocked()
        found: list[Frontier] = []
        reach = int(math.ceil(FRONTIER_REACH_CM / self.cell))
        near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
        piece = max(1, int(round(FRONTIER_PIECE_CM / self.cell)))
        pieces: list[np.ndarray] = []
        for label in range(1, labels_count):
            component = np.argwhere(labels == label)
            if len(component) < FRONTIER_MIN_CELLS:
                continue
            keys = [tuple(key) for key in component // piece]
            for key in set(keys):
                members = component[[index for index, other in enumerate(keys) if other == key]]
                if len(members) >= max(2, FRONTIER_MIN_CELLS // 2):
                    pieces.append(members)
        for members in pieces:
            mask = np.zeros(cells.shape, dtype=np.uint8)
            mask[members[:, 0], members[:, 1]] = 1
            near = cv2.dilate(mask, near_kernel) > 0
            standpoints = np.argwhere(near & np.isfinite(distance))
            if not len(standpoints):
                continue
            centroid = self.centre_of(tuple(np.round(members.mean(axis=0)).astype(int)))
            middle = self.centre_of(tuple(min(members, key=lambda cell: math.dist(self.centre_of(tuple(cell)), centroid))))
            # Stand as near the middle of the frontier as the car can get, and
            # judge from there looking at it; from the frontier itself rays
            # could start past a wall. Standing on it, look the way the car arrives.
            target = tuple(min(standpoints, key=lambda cell: math.dist(self.centre_of(tuple(cell)), middle)))
            stand = self.centre_of(target)
            if math.dist(stand, middle) > self.cell:
                facing = math.degrees(math.atan2(-(middle[0] - stand[0]), middle[1] - stand[1]))
            else:
                facing = math.degrees(math.atan2(-(stand[0] - pose[0]), stand[1] - pose[1]))
            gain = self.information_gain(stand, facing, unknown, blocked)
            width = self.opening_width(middle, unknown, occupied)
            if width is not None and width <= NARROW_OPENING_CM:
                kind = "doorway"
            elif gain >= OPEN_GAIN_CELLS:
                kind = "open"
            else:
                kind = "edge"
            found.append(
                Frontier(centroid, self.centre_of(target), len(members), float(distance[target]), gain, kind, width, middle)
            )
        found.extend(self._lead_frontiers(pose, distance, unknown, blocked))
        return found, parents, start

    def _lead_frontiers(
        self, pose: Sequence[float], distance: np.ndarray, unknown: np.ndarray, blocked: np.ndarray
    ) -> list[Frontier]:
        """Openings seen in sweeps that are still worth a look."""
        frontiers = []
        remaining = []
        reach = int(math.ceil((FRONTIER_REACH_CM + ROBOT_RADIUS_CM) / self.cell))
        for mouth, into in self.leads:
            behind = self.cell_of(mouth + into * LEAD_BEHIND_CM)
            settled = self._inside(behind) and not unknown[behind]
            visited = any(math.dist(stood, mouth) <= LEAD_VISITED_CM for stood in self.stood)
            if settled or visited:
                continue
            remaining.append((mouth, into))
            row, column = self.cell_of(mouth)
            rows = slice(max(row - reach, 0), row + reach + 1)
            columns = slice(max(column - reach, 0), column + reach + 1)
            window = distance[rows, columns]
            if not np.isfinite(window).any():
                continue
            # Stand at the reachable spot nearest the mouth, looking into it.
            cells = np.argwhere(np.isfinite(window)) + [rows.start, columns.start]
            target = tuple(min(cells, key=lambda cell: math.dist(self.centre_of(tuple(cell)), mouth)))
            stand = self.centre_of(target)
            facing = math.degrees(math.atan2(-into[0], into[1]))
            gain = max(LEAD_MIN_GAIN_CELLS, self.information_gain(stand, facing, unknown, blocked))
            point = (float(mouth[0]), float(mouth[1]))
            frontiers.append(Frontier(point, stand, 1, float(distance[target]), gain, "lead", None, point))
        self.leads = remaining
        return frontiers

    def next_waypoint(
        self,
        pose: Sequence[float],
        avoid_within_deg: float | None = None,
        previous_target: Sequence[float] | None = None,
        learner: ExplorationLearner | None = None,
    ) -> Waypoint | None:
        """The next point to head for on the way to the most promising frontier.

        With `avoid_within_deg`, frontiers lying within that angle of straight
        ahead are skipped (the sonar says the way ahead is blocked). A frontier
        near `previous_target` is kept unless another promises clearly more."""
        learner = learner or ExplorationLearner()
        frontiers, parents, start = self.frontiers(pose)
        here = np.array(pose[:2], dtype=float)

        def turn_to(point: Sequence[float]) -> float:
            offset = np.array(point) - here
            bearing = math.degrees(math.atan2(-offset[0], offset[1]))
            return (bearing - float(pose[2]) + 180.0) % 360.0 - 180.0

        def value(frontier: Frontier) -> float:
            return learner.utility(frontier, turn_to(frontier.target))

        frontiers = [frontier for frontier in frontiers if frontier.gain >= GAIN_MIN_CELLS]
        if avoid_within_deg is not None:
            frontiers = [frontier for frontier in frontiers if abs(turn_to(frontier.target)) > avoid_within_deg]
        if not frontiers:
            return None
        best = max(frontiers, key=value)
        if not learner.worth_it(value(best)):
            return None
        if previous_target is not None:
            kept = [
                frontier
                for frontier in frontiers
                if math.hypot(frontier.target[0] - previous_target[0], frontier.target[1] - previous_target[1])
                <= TARGET_STICK_RADIUS_CM
            ]
            if kept:
                same = max(kept, key=value)
                if value(same) * TARGET_SWITCH_FACTOR >= value(best):
                    best = same
        if best.middle is not None and math.hypot(best.target[0] - here[0], best.target[1] - here[1]) <= FRONTIER_REACH_CM:
            # Already where it wanted to stand: face the frontier and go on into it.
            look = best.middle
            return Waypoint(turn_to(look), math.hypot(look[0] - here[0], look[1] - here[1]), best)
        waypoint = self._waypoint_along(here, self.cell_of(best.target), parents, start)
        return Waypoint(turn_to(waypoint), math.hypot(waypoint[0] - here[0], waypoint[1] - here[1]), best)

    def _waypoint_along(
        self, here: np.ndarray, goal: tuple[int, int], parents: dict, start: tuple[int, int]
    ) -> tuple[float, float]:
        """The furthest point along the path to `goal`, within a leg, that can
        be reached in a straight line."""
        path = [goal]
        while path[-1] != start and path[-1] in parents:
            path.append(parents[path[-1]])
        path.reverse()
        traversable = self.traversable()
        # At least the first step along the path, even if the car's own cell
        # is too near a wall to count as traversable.
        waypoint = self.centre_of(path[min(1, len(path) - 1)])
        for cell in path[1:]:
            point = self.centre_of(cell)
            if math.hypot(point[0] - here[0], point[1] - here[1]) > LEG_MAX_CM:
                break
            if not self.line_of_sight(here, point, traversable):
                break
            waypoint = point
        return waypoint

    def path_lengths(self, pose: Sequence[float]) -> tuple[np.ndarray, dict, tuple[int, int]]:
        """Path cost (cm, inf where unreachable) to every cell from the car, with
        the parents to follow a path back."""
        start = self.cell_of(pose[:2])
        distance, parents = self._distances(start)
        return distance, parents, start

    def route(self, pose: Sequence[float], goal: Sequence[float], routes=None) -> tuple[float, float] | None:
        """Bearing (counterclockwise from the heading) and distance of the next
        straight leg toward `goal`, or None when it cannot be reached.
        `routes` is a path_lengths() result to reuse."""
        if not self.log_odds.size:
            return None
        distance, parents, start = routes or self.path_lengths(pose)
        cell = self.cell_of(goal)
        if not self._inside(cell) or not math.isfinite(distance[cell]):
            return None
        here = np.array(pose[:2], dtype=float)
        point = self._waypoint_along(here, cell, parents, start)
        offset = np.array(point) - here
        bearing = math.degrees(math.atan2(-offset[0], offset[1]))
        return (bearing - float(pose[2]) + 180.0) % 360.0 - 180.0, float(np.hypot(*offset))

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
