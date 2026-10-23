"""A two-room floor plan joined by a corridor, for judging how well the
explorer finds its way round, not just whether single decisions look right.

The car starts in room A facing its far wall; the corridor to room B opens
off the left wall. Motion is exact, so only the exploring is being tested.
Missing echoes read 300 cm as on the real sensor, whether the space is open
or the sound glanced off a wall at a steep angle.
"""

import math

import numpy as np

from robocar.exploration import CoverageGrid
from robocar.mapping import AUTO_WALL_STANDOFF_CM, plan_explore_step
from robocar.surfaces import SONAR_BEAM_WIDTH_DEGREES, direction
from sonar_room import SonarRoom

NO_ECHO_CM = 300.0
ROBOT_RADIUS_CM = 15.0
SWEEP_ANGLES = range(57, -61, -3)

FLOOR_WALLS = (
    # Room A
    ((-150.0, 150.0), (150.0, 150.0)),
    ((150.0, -100.0), (150.0, 150.0)),
    ((-150.0, -100.0), (150.0, -100.0)),
    ((-150.0, -100.0), (-150.0, 40.0)),
    ((-150.0, 120.0), (-150.0, 150.0)),
    # Corridor
    ((-400.0, 40.0), (-150.0, 40.0)),
    ((-400.0, 120.0), (-150.0, 120.0)),
    # Room B
    ((-400.0, -60.0), (-400.0, 40.0)),
    ((-400.0, 120.0), (-400.0, 260.0)),
    ((-650.0, 260.0), (-400.0, 260.0)),
    ((-650.0, -60.0), (-650.0, 260.0)),
    ((-650.0, -60.0), (-400.0, -60.0)),
)
# (x_min, y_min, x_max, y_max)
ROOM_A = (-150.0, -100.0, 150.0, 150.0)
CORRIDOR = (-400.0, 40.0, -150.0, 120.0)
ROOM_B = (-650.0, -60.0, -400.0, 260.0)
START = (0.0, -40.0, 0.0)


def _inside(point, box, margin: float = 0.0) -> bool:
    return box[0] + margin <= point[0] <= box[2] - margin and box[1] + margin <= point[1] <= box[3] - margin


class FloorPlan:
    def __init__(self) -> None:
        self.sonar = SonarRoom(walls=FLOOR_WALLS, posts=())
        self.segments = [tuple(map(np.array, wall)) for wall in FLOOR_WALLS]
        cells = []
        for box in (ROOM_A, CORRIDOR, ROOM_B):
            for x in np.arange(box[0] + 2.5, box[2], 5.0):
                for y in np.arange(box[1] + 2.5, box[3], 5.0):
                    cells.append((x, y))
        self.floor = np.unique(np.array(cells), axis=0)

    def scan(self, pose) -> list[dict[str, object]]:
        readings = self.sonar.scan(pose, SWEEP_ANGLES)
        for reading in readings:
            if reading["distance_cm"] is None or reading["distance_cm"] > NO_ECHO_CM:
                reading["distance_cm"] = NO_ECHO_CM
        return readings

    def clearance(self, point) -> float:
        best = math.inf
        for start, end in self.segments:
            edge = end - start
            t = min(1.0, max(0.0, float((np.asarray(point) - start) @ edge / (edge @ edge))))
            best = min(best, float(np.linalg.norm(np.asarray(point) - (start + t * edge))))
        return best

    def drive(self, pose, distance_cm: float) -> tuple[tuple[float, float, float], bool]:
        """Move straight, stopping short (and reporting a bump) at a wall."""
        heading = direction(pose[2])
        step = math.copysign(2.0, distance_cm)
        travelled = 0.0
        position = np.array(pose[:2], dtype=float)
        while abs(travelled) < abs(distance_cm):
            candidate = position + heading * step
            if self.clearance(candidate) < ROBOT_RADIUS_CM:
                return (float(position[0]), float(position[1]), pose[2]), True
            position = candidate
            travelled += step
        return (float(position[0]), float(position[1]), pose[2]), False

    def coverage(self, grid: CoverageGrid) -> float:
        """Fraction of the floor the grid knows (free or occupied)."""
        known = ~grid.unknown()
        seen = 0
        for x, y in self.floor:
            row, column = grid.cell_of((x, y))
            if 0 <= row < known.shape[0] and 0 <= column < known.shape[1] and known[row, column]:
                seen += 1
        return seen / len(self.floor)

    def room_coverage(self, grid: CoverageGrid, box) -> float:
        known = ~grid.unknown()
        cells = [cell for cell in self.floor if _inside(cell, box)]
        seen = 0
        for x, y in cells:
            row, column = grid.cell_of((x, y))
            if 0 <= row < known.shape[0] and 0 <= column < known.shape[1] and known[row, column]:
                seen += 1
        return seen / len(cells)


def explore(moves: int = 60, plan=plan_explore_step, start=START, on_sweep=None):
    """Run the planner on the floor plan; returns per-move coverage and the run."""
    floor = FloorPlan()
    grid = CoverageGrid()
    pose = start
    turns_without_progress = backup_bursts = 0
    just_backed_up = False
    target = None
    history = []
    bumps = 0
    for _ in range(moves):
        scan = floor.scan(pose)
        grid.integrate(scan, pose, SONAR_BEAM_WIDTH_DEGREES)
        if on_sweep is not None:
            on_sweep(grid, pose, scan)
        step = plan(scan, pose, grid, turns_without_progress, backup_bursts, just_backed_up, target)
        target = step.target
        history.append((pose, step, floor.coverage(grid)))
        if step.action in ("done", "blocked"):
            break
        if step.action == "forward":
            pose, bumped = floor.drive(pose, step.distance_cm)
            bumps += bumped
            turns_without_progress = 0
            just_backed_up = False
        elif step.action == "turn":
            pose = (pose[0], pose[1], pose[2] + step.degrees)
            turns_without_progress += 1
            backup_bursts = 0
            just_backed_up = False
            if step.distance_cm > 0:
                ahead = floor.sonar.distance(pose, 0.0)
                ahead = NO_ECHO_CM if ahead is None else ahead
                if ahead - step.distance_cm >= AUTO_WALL_STANDOFF_CM:
                    pose, bumped = floor.drive(pose, step.distance_cm)
                    bumps += bumped
                    turns_without_progress = 0
        elif step.action == "strafe":
            # Slide sideways: drive along the heading 90 degrees to the right.
            sideways = (pose[0], pose[1], pose[2] - 90.0)
            moved, bumped = floor.drive(sideways, step.distance_cm)
            pose = (moved[0], moved[1], pose[2])
            bumps += bumped
        else:
            pose, bumped = floor.drive(pose, -15.0)
            backup_bursts += 1
            turns_without_progress = 0
            just_backed_up = True
    return floor, grid, history, bumps
