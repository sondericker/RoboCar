"""Room mapping: choosing what to do next among several ways of mapping.

After each action the car weighs what it could do next:

- "far_spin": drive straight across the room to the farthest open place the
  last full circle showed, and turn a full circle there (the preferred way:
  long hops across open floor, a whole view at each end);
- "spin": drive to the nearby open place a full turn would show the most new
  floor from, and turn a full circle there;
- "frontier": one sweep ahead and one move toward the most promising
  unexplored edge (for corridors, doorways and corners too tight to spin).

Each option is worth the new floor it is expected to show per second it is
expected to take, times a standing preference for the method and times what
the method has really delivered against its promises so far (its track
record, learned run after run). Time is the driving at the speed the car has
really been managing, plus a full circle (or a sweep) at what those have
really taken. The best is done; when nothing promises enough, the room is
mapped.
"""

import math
from dataclasses import dataclass, field

from robocar.exploration import CoverageGrid, ExplorationLearner, Frontier, SpinStation

METHODS = ("far_spin", "spin", "frontier")
# The standing preference: crossing the room to spin is the favourite.
PREFERENCE = {"far_spin": 1.6, "spin": 1.0, "frontier": 0.5}
# Starting guesses for what things take, refined from what they really took.
SPIN_SECONDS = 45.0
SWEEP_SECONDS = 12.0
DRIVE_CM_PER_SECOND = 12.0
# Track records stay within these, and move this far toward each new result.
TRACK_RECORD_LIMITS = (0.1, 2.0)
TIMING_LIMITS = {"spin_seconds": (15.0, 180.0), "sweep_seconds": (4.0, 60.0), "drive_cm_per_second": (3.0, 60.0)}
LEARNING_RATE = 0.25
# The room counts as mapped once even the best option would bring less than
# this share of the best rate of new floor this run has managed (or less
# than the floor below, whatever the run): what is left is scraps.
STOP_SHARE_OF_BEST = 0.05
MIN_CELLS_PER_SECOND = 0.5


@dataclass(frozen=True)
class MappingOption:
    method: str
    gain: float  # cells of unknown floor it should bring into view
    seconds: float  # how long it should take
    score: float  # what it is worth: cells a second, weighted
    station: SpinStation | None = None
    frontier: Frontier | None = None

    def describe(self) -> str:
        if self.method == "frontier":
            return f"a step toward an unexplored edge ({self.gain:.0f} cells, about {self.seconds:.0f} s)"
        where = "across the room" if self.method == "far_spin" else "nearby"
        distance = self.station.path_cm if self.station is not None else 0.0
        return f"a full circle {where}, {distance:.0f} cm away ({self.gain:.0f} cells, about {self.seconds:.0f} s)"


@dataclass
class MethodRecords:
    """What each way of mapping has really delivered, and how long things take."""

    track: dict[str, float] = field(default_factory=lambda: {method: 1.0 for method in METHODS})
    tries: dict[str, int] = field(default_factory=lambda: {method: 0 for method in METHODS})
    spin_seconds: float = SPIN_SECONDS
    sweep_seconds: float = SWEEP_SECONDS
    drive_cm_per_second: float = DRIVE_CM_PER_SECOND

    def note(self, method: str, promised: float, delivered: float) -> None:
        """A method promised `promised` new cells and brought `delivered`."""
        if method not in self.track or promised <= 0:
            return
        low, high = TRACK_RECORD_LIMITS
        ratio = min(high, max(low, delivered / promised))
        self.track[method] += LEARNING_RATE * (ratio - self.track[method])
        self.tries[method] = self.tries.get(method, 0) + 1

    def note_timing(self, name: str, value: float) -> None:
        if name not in TIMING_LIMITS or not math.isfinite(value) or value <= 0:
            return
        low, high = TIMING_LIMITS[name]
        current = getattr(self, name)
        setattr(self, name, min(high, max(low, current + LEARNING_RATE * (value - current))))

    def to_json(self) -> dict[str, object]:
        return {
            "track": {method: round(value, 3) for method, value in self.track.items()},
            "tries": dict(self.tries),
            "spin_seconds": round(self.spin_seconds, 1),
            "sweep_seconds": round(self.sweep_seconds, 1),
            "drive_cm_per_second": round(self.drive_cm_per_second, 1),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "MethodRecords":
        records = cls()
        low, high = TRACK_RECORD_LIMITS
        for method, value in dict(data.get("track") or {}).items():
            if method in records.track and isinstance(value, (int, float)):
                records.track[method] = min(high, max(low, float(value)))
        for method, value in dict(data.get("tries") or {}).items():
            if method in records.tries and isinstance(value, int):
                records.tries[method] = value
        for name, (low, high) in TIMING_LIMITS.items():
            value = data.get(name)
            if isinstance(value, (int, float)):
                setattr(records, name, min(high, max(low, float(value))))
        return records


def options(
    grid: CoverageGrid,
    pose,
    avoid,
    learner: ExplorationLearner,
    records: MethodRecords,
    frontier_spent: bool = False,
) -> list[MappingOption]:
    """Every way of carrying on from here, best first."""
    found: list[MappingOption] = []

    def add(method: str, gain: float, seconds: float, **where) -> None:
        if gain <= 0 or seconds <= 0:
            return
        score = gain / seconds * PREFERENCE[method] * records.track[method]
        found.append(MappingOption(method, gain, seconds, score, **where))

    def spin_seconds(path_cm: float) -> float:
        return path_cm / records.drive_cm_per_second + records.spin_seconds

    far = grid.far_station(pose, avoid=avoid)
    if far is not None:
        add("far_spin", far.gain, spin_seconds(far.path_cm), station=far)
    near = grid.spin_station(pose, avoid=avoid)
    if near is not None and (far is None or near.target != far.target):
        add("spin", near.gain, spin_seconds(near.path_cm), station=near)
    if not frontier_spent:
        frontiers, _, _ = grid.frontiers(pose)
        if frontiers:
            best = max(frontiers, key=lambda frontier: learner.expected(frontier) / (frontier.path_cm + 100.0))
            # A step is one sweep, from where it ends up facing the frontier:
            # it can show no more than that sweep would, however much unknown
            # floor lies beyond.
            offset = (best.target[0] - pose[0], best.target[1] - pose[1])
            facing = math.degrees(math.atan2(-offset[0], offset[1]))
            seen = grid.information_gain(best.target, facing, grid.worth_seeing(), grid._blocked())
            add(
                "frontier",
                min(learner.expected(best), seen),
                records.sweep_seconds + best.path_cm / records.drive_cm_per_second,
                frontier=best,
            )
    found.sort(key=lambda option: option.score, reverse=True)
    return found


def expected_rate(option: MappingOption, records: MethodRecords) -> float:
    """New cells a second the option should really bring, by its method's record."""
    return option.gain * records.track[option.method] / option.seconds


def choose(found: list[MappingOption], records: MethodRecords, best_rate: float = 0.0) -> MappingOption | None:
    """The best option, or None when nothing promises enough new floor: less
    than a small share of the best rate this run has managed."""
    if not found:
        return None
    best = found[0]
    enough = max(MIN_CELLS_PER_SECOND, STOP_SHARE_OF_BEST * best_rate)
    return best if expected_rate(best, records) >= enough else None
