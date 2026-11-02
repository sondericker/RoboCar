"""Deciding the next move while exploring, from the latest sweep and the
coverage map: forward toward the most promising unexplored edge, turn
toward it, slide sideways out of a tight spot, back out, or call it done.

The way ahead must be clear across the car's width. From a sonar sweep the
nearest reading within the beam of a direction says how far that is; from a
lidar sweep (2 degree beam) a car-wide corridor is checked against the
points, with the sonar riding along having a say for glass. People and
animals the camera sees close ahead block the way like walls.
"""

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from robocar.exploration import CELL_CM, CoverageGrid, ExplorationLearner
from robocar.surfaces import direction

FORWARD_CORRIDOR_DEGREES = 12
# Half the width kept clear along a way the car could drive (the car is about
# 20 cm across on its mecanum wheels), judged from lidar points.
CORRIDOR_HALF_WIDTH_CM = 14.0
# A lidar corridor with nothing in it is open at least this far.
LIDAR_OPEN_CM = 300.0
AUTO_MIN_CLEARANCE_CM = 60.0
# Forward legs run as far toward the next waypoint as a straight line allows,
# up to AUTO_LONG_STEP_CM between sweeps, stopping AUTO_WALL_STANDOFF_CM short
# of whatever the sonar sees ahead.
AUTO_LONG_STEP_CM = 60.0
AUTO_MIN_STEP_CM = 10.0
# A person or animal the camera sees within AUTO_BLOCKED_CONE_DEGREES of
# where the car would drive, nearer than this (or at an unknown distance),
# blocks the way.
AUTO_HAZARD_RANGE_CM = 150.0
# Too close to a wall to spin: slide this far sideways toward the roomier
# side (readings beyond AUTO_STRAFE_SIDE_DEGREES), if it has at least
# AUTO_STRAFE_MIN_ROOM_CM.
AUTO_STRAFE_CM = 12.0
AUTO_STRAFE_SIDE_DEGREES = 40.0
AUTO_STRAFE_MIN_ROOM_CM = 30.0
# In a corridor, a target within this angle of its direction is reached by
# following the centre line, aiming this far ahead along it.
AUTO_CORRIDOR_FOLLOW_DEGREES = 70.0
AUTO_CORRIDOR_AIM_CM = 60.0
AUTO_WALL_STANDOFF_CM = 45.0
CELL_AREA_M2 = (CELL_CM / 100.0) ** 2
AUTO_MAX_BACKUP_BURSTS = 1
# Head for the next waypoint once within this angle of it.
AUTO_HEADING_TOLERANCE_DEGREES = 10.0
# When the sonar blocks a waypoint within this angle of ahead, pick another.
AUTO_BLOCKED_CONE_DEGREES = 25.0
# With no reachable frontier, look round this many turns before calling it explored.
AUTO_LOOK_AROUND_TURNS = 3
# Turns made just to look cover about a sweep's width (114 degrees), so three
# looks see all round.
AUTO_LOOK_TURN_DEGREES = 110.0
# A turn toward a target is made in one stop, up to this much, as pieces the
# camera strip can check (at most AUTO_TURN_PIECE_DEGREES each) with no sweep
# in between: a sweep only pays once the car faces somewhere new.
AUTO_MAX_TURN_PER_STOP_DEGREES = 150.0
# A side reading this short at the sweep's edge (57 degrees) puts a wall about
# 17 cm from the centre: just room for the car to spin.
AUTO_TURN_MIN_CLEARANCE_CM = 20.0
# About 400 degrees of turning without finding a way forward, then back up.
AUTO_MAX_TURNS_WITHOUT_PROGRESS = 12


def is_lidar_scan(scan: Sequence[dict[str, object]]) -> bool:
    return any(item.get("sensor") == "lidar" for item in scan)


def _corridor_clearance(points: np.ndarray, heading_deg: float, open_cm: float) -> float:
    """How far a car-wide corridor in direction `heading_deg` runs before
    meeting one of `points` (sweep readings, local x right / y ahead from
    the sensor); `open_cm` when nothing is in it."""
    if not len(points):
        return open_cm
    radians = math.radians(heading_deg)
    ahead = points[:, 0] * -math.sin(radians) + points[:, 1] * math.cos(radians)
    across = points[:, 0] * math.cos(radians) + points[:, 1] * math.sin(radians)
    inside = (ahead > 0) & (np.abs(across) <= CORRIDOR_HALF_WIDTH_CM)
    return float(ahead[inside].min()) if inside.any() else open_cm


def _lidar_points(scan: Sequence[dict[str, object]]) -> tuple[np.ndarray, list[float]]:
    """Points the lidar hit, and the angles where it saw nothing it trusts."""
    points, blind = [], []
    for item in scan:
        value = item.get("distance_cm")
        angle = float(item["angle_deg"])
        if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
            blind.append(angle)
            continue
        radians = math.radians(angle)
        points.append((-math.sin(radians) * float(value), math.cos(radians) * float(value)))
    return np.array(points, dtype=float).reshape(-1, 2), blind


def _sonar_near(scan: Sequence[dict[str, object]], angle: float) -> float | None:
    """The sonar's nearest reading recorded alongside a lidar sweep within
    its beam of `angle`: glass and other things the lidar misses."""
    values = [
        float(item["sonar_cm"])
        for item in scan
        if isinstance(item.get("sonar_cm"), (int, float))
        and abs(float(item["angle_deg"]) - angle) <= FORWARD_CORRIDOR_DEGREES
    ]
    return min(values) if values else None


def forward_clearance(scan: Sequence[dict[str, object]]) -> float | None:
    """Clear distance in the corridor ahead; None without a straight-ahead echo.

    From a sonar sweep, the nearest reading within its beam of ahead. From a
    lidar sweep, how far a car-wide corridor runs before meeting a point
    (the sonar riding along has a say too, for glass)."""
    if is_lidar_scan(scan):
        points, blind = _lidar_points(scan)
        ahead_seen = any(abs(float(item["angle_deg"])) <= 3 for item in scan if item.get("distance_cm") is not None)
        sonar = _sonar_near(scan, 0.0)
        if not ahead_seen and sonar is None:
            return None
        clearance = _corridor_clearance(points, 0.0, LIDAR_OPEN_CM)
        return min(clearance, sonar) if sonar is not None else clearance
    if not any(item["angle_deg"] == 0 and item["distance_cm"] is not None for item in scan):
        return None
    return min(
        float(item["distance_cm"])
        for item in scan
        if item["distance_cm"] is not None
        and abs(float(item["angle_deg"])) <= FORWARD_CORRIDOR_DEGREES
    )


def direction_clearances(scan: Sequence[dict[str, object]]) -> dict[float, float]:
    """Clearance in each swept direction, taken across a car-wide corridor."""
    if is_lidar_scan(scan):
        points, _ = _lidar_points(scan)
        clearances = {}
        for item in scan:
            if item.get("distance_cm") is None:
                continue
            angle = float(item["angle_deg"])
            clearance = _corridor_clearance(points, angle, LIDAR_OPEN_CM)
            sonar = _sonar_near(scan, angle)
            clearances[angle] = min(clearance, sonar) if sonar is not None else clearance
        return clearances
    valid = {
        float(item["angle_deg"]): float(item["distance_cm"])
        for item in scan
        if item["distance_cm"] is not None
    }
    return {
        angle: min(
            distance for other, distance in valid.items() if abs(other - angle) <= FORWARD_CORRIDOR_DEGREES
        )
        for angle in valid
    }


@dataclass(frozen=True)
class ExploreStep:
    action: str  # "forward", "turn", "strafe", "reverse", "blocked" or "done"
    degrees: float = 0.0
    reason: str = ""
    target: tuple[float, float] | None = None
    # For "forward", the leg. For "turn", a leg to drive straight after the
    # turn if a forward ping confirms the way (0: sweep after turning). For
    # "strafe", how far to slide sideways (positive: to the right).
    distance_cm: float = 0.0


def plan_explore_step(
    scan: Sequence[dict[str, object]],
    pose: Sequence[float],
    grid: CoverageGrid,
    turns_without_progress: int,
    backup_bursts: int,
    just_backed_up: bool = False,
    previous_target: Sequence[float] | None = None,
    learner: ExplorationLearner | None = None,
    hazards: Sequence[tuple[float, float | None]] = (),
) -> ExploreStep:
    """Plan the next move; people and animals the camera sees close ahead
    (`hazards`: bearing and range) block the way like a wall would."""
    step = _plan_explore_step(
        scan, pose, grid, turns_without_progress, backup_bursts, just_backed_up, previous_target, learner, hazards
    )
    if hazard_within(hazards, 0.0) and step.action != "forward":
        step = replace(step, reason="a person or animal is close ahead; " + step.reason)
    return step


def hazard_within(hazards: Sequence[tuple[float, float | None]], heading_change: float) -> bool:
    """Whether a person or animal is close ahead once the car has turned by `heading_change`."""
    return any(
        abs((bearing - heading_change + 180.0) % 360.0 - 180.0) <= AUTO_BLOCKED_CONE_DEGREES
        and (distance is None or distance < AUTO_HAZARD_RANGE_CM)
        for bearing, distance in hazards
    )


def _plan_explore_step(
    scan: Sequence[dict[str, object]],
    pose: Sequence[float],
    grid: CoverageGrid,
    turns_without_progress: int,
    backup_bursts: int,
    just_backed_up: bool,
    previous_target: Sequence[float] | None,
    learner: ExplorationLearner | None,
    hazards: Sequence[tuple[float, float | None]],
) -> ExploreStep:
    """Decide the next autonomous move toward the most promising frontier.

    The coverage grid holds everything every sweep has seen. The learner
    weighs how much unknown floor each frontier could reveal (a long reading
    off to one side leaves a lot) against the driving and turning needed to
    get there, using what earlier moves actually revealed. The car turns the
    short way toward the next waypoint and drives as far toward it as a
    straight line allows; the sonar sweep has the final say on driving.
    """
    learner = learner or ExplorationLearner()
    learner.observe(pose, grid.known_count())
    if turns_without_progress == 0:
        learner.last_turn = 0.0
    ahead = forward_clearance(scan)
    clearances = direction_clearances(scan)
    # Right after backing out of a dead end, never drive straight back in.
    ahead_clear = (
        ahead is not None
        and ahead >= AUTO_MIN_CLEARANCE_CM
        and not just_backed_up
        and not hazard_within(hazards, 0.0)
    )
    can_turn = bool(clearances) and min(clearances.values()) >= AUTO_TURN_MIN_CLEARANCE_CM
    turning_allowed = can_turn and turns_without_progress < AUTO_MAX_TURNS_WITHOUT_PROGRESS
    if not can_turn and clearances:
        # Too near a wall to spin: slide sideways (the wheels are mecanum)
        # toward whichever side has more room, rather than back up blind.
        left = [value for angle, value in clearances.items() if angle >= AUTO_STRAFE_SIDE_DEGREES]
        right = [value for angle, value in clearances.items() if angle <= -AUTO_STRAFE_SIDE_DEGREES]
        room_left, room_right = min(left, default=0.0), min(right, default=0.0)
        room = max(room_left, room_right)
        if room >= AUTO_STRAFE_MIN_ROOM_CM:
            slide = AUTO_STRAFE_CM if room_right >= room_left else -AUTO_STRAFE_CM
            side = "right" if slide > 0 else "left"
            return ExploreStep("strafe", reason=f"too close to turn; sliding {AUTO_STRAFE_CM:.0f} cm {side}", distance_cm=slide)

    waypoint = grid.next_waypoint(pose, previous_target=previous_target, learner=learner)
    if waypoint is None:
        learner.finish(grid.known_count())
        if turning_allowed and turns_without_progress < AUTO_LOOK_AROUND_TURNS:
            return ExploreStep("turn", -AUTO_LOOK_TURN_DEGREES, "nothing worth a trip in view yet; turning to look")
        if can_turn or ahead_clear:
            return ExploreStep("done", reason="no unexplored edges left worth a trip")
        if backup_bursts < AUTO_MAX_BACKUP_BURSTS and not just_backed_up:
            return ExploreStep("reverse", reason="too close to turn; backing up once")
        return ExploreStep("blocked", reason="no safe path after looking all around")
    if not ahead_clear and abs(waypoint.bearing_deg) <= AUTO_BLOCKED_CONE_DEGREES:
        waypoint = (
            grid.next_waypoint(pose, avoid_within_deg=AUTO_BLOCKED_CONE_DEGREES, learner=learner) or waypoint
        )
    learner.pursue(waypoint.frontier, grid.known_count())
    bearing = waypoint.bearing_deg
    distance_to_waypoint = waypoint.distance_cm
    frontier = waypoint.frontier
    corridor = grid.corridor(pose)
    in_corridor = False
    if corridor is not None:
        # Heading for something along the corridor: follow its centre line
        # rather than picking a way between mapped cells.
        here = np.array(pose[:2], dtype=float)
        offset = np.array(frontier.target) - here
        toward = math.degrees(math.atan2(-offset[0], offset[1]))
        for axis in (corridor.axis_deg, corridor.axis_deg + 180.0):
            if abs((toward - axis + 180.0) % 360.0 - 180.0) <= AUTO_CORRIDOR_FOLLOW_DEGREES:
                centre = corridor.centre_offset_cm if axis == corridor.axis_deg else -corridor.centre_offset_cm
                aim = here + direction(axis) * AUTO_CORRIDOR_AIM_CM + direction(axis + 90.0) * centre
                aim_offset = aim - here
                bearing = (math.degrees(math.atan2(-aim_offset[0], aim_offset[1])) - float(pose[2]) + 180.0) % 360.0 - 180.0
                distance_to_waypoint = min(float(np.hypot(*offset)), AUTO_LONG_STEP_CM)
                in_corridor = True
                break
    target = frontier.target
    what = {
        "open": "open space",
        "doorway": f"a {frontier.width_cm:.0f} cm opening" if frontier.width_cm else "an opening",
        "edge": "an unexplored edge",
        "lead": "an opening the sonar saw into",
    }[frontier.kind]
    where = f"{what} {frontier.path_cm:.0f} cm away, could show {frontier.gain * CELL_AREA_M2:.1f} m²"
    if in_corridor:
        where = f"the {corridor.width_cm:.0f} cm corridor's centre line, toward {where}"

    if abs(bearing) > AUTO_HEADING_TOLERANCE_DEGREES and turning_allowed:
        limit = AUTO_MAX_TURN_PER_STOP_DEGREES
        degrees = max(-limit, min(limit, bearing))
        # Just lining up with a waypoint the map shows clear: no need for a
        # sweep in between, only a ping ahead before driving on.
        then = 0.0
        if degrees == bearing and distance_to_waypoint >= AUTO_MIN_STEP_CM + 10.0 and not hazard_within(hazards, degrees):
            facing = (pose[0], pose[1], float(pose[2]) + degrees)
            then = min(AUTO_LONG_STEP_CM, distance_to_waypoint, grid.clear_run(facing, AUTO_LONG_STEP_CM))
            if then < AUTO_MIN_STEP_CM:
                then = 0.0
        reason = f"turning toward {where} ({bearing:+.0f}°)" + (f", then {then:.0f} cm" if then else "")
        learner.last_turn = 0.0 if then else degrees
        return ExploreStep("turn", degrees, reason, target, distance_cm=then)
    # The map has a say too: a wall met at a glancing angle never echoes.
    mapped_run = grid.clear_run(pose, AUTO_LONG_STEP_CM)
    if ahead_clear and mapped_run >= AUTO_MIN_STEP_CM:
        leg = min(AUTO_LONG_STEP_CM, distance_to_waypoint, ahead - AUTO_WALL_STANDOFF_CM, mapped_run)
        leg = max(AUTO_MIN_STEP_CM, leg)
        return ExploreStep("forward", reason=f"driving {leg:.0f} cm toward {where}", target=target, distance_cm=leg)
    if ahead_clear and turning_allowed:
        # The sonar says clear but the map shows a wall close ahead: face the
        # waypoint squarely, or if already facing it, head somewhere else.
        if abs(bearing) >= 3.0:
            return ExploreStep("turn", bearing, f"a mapped wall is close ahead; squaring up toward {where}", target)
        other = grid.next_waypoint(pose, avoid_within_deg=AUTO_BLOCKED_CONE_DEGREES, learner=learner)
        if other is not None:
            limit = AUTO_MAX_TURN_PER_STOP_DEGREES
            degrees = max(-limit, min(limit, other.bearing_deg))
            return ExploreStep("turn", degrees, "a mapped wall blocks the way; heading elsewhere", other.frontier.target)
    if turning_allowed:
        return ExploreStep("turn", -AUTO_LOOK_TURN_DEGREES, "way ahead blocked; turning to look", target)
    if backup_bursts < AUTO_MAX_BACKUP_BURSTS and not just_backed_up:
        return ExploreStep("reverse", reason="too close to turn; backing up once", target=target)
    return ExploreStep("blocked", reason="no safe path after looking all around", target=target)
