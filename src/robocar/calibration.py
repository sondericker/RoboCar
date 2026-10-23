"""Motion calibration: measure how far the car really turns and moves.

Run attended, with the car facing a flat wall about 1-1.5 m away and room to
turn. Turns are measured with the camera: a dense strip of pictures across
the pan range is taken first, and after the turn the forward view is matched
against it. Each turn size is repeated, adjusting the command by the measured
ratio, until it lands within tolerance. Forward steps are measured with the
sonar against the wall, adjusting the burst duration until each step size is
right, and every forward step is followed by the same burst in reverse so the
car stays in place.

The result maps turn size and direction to "real degrees per odometry
degree" and step size to burst duration; turning and exploring use it.
"""

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

logger = logging.getLogger(__name__)

TURN_SIZES_DEGREES = (15.0, 30.0, 45.0)
TURN_TOLERANCE_DEGREES = 2.0
TURN_ATTEMPTS = 4
STRIP_STEP_DEGREES = 5
STEP_SIZES_CM = (10.0, 20.0, 30.0)
# Steps must land within 7% (at least 1 cm) of the size asked for.
STEP_TOLERANCE_FRACTION = 0.07
STEP_TOLERANCE_MIN_CM = 1.0
STEP_ATTEMPTS = 4
STEP_DURATION_LIMITS_SECONDS = (0.05, 0.8)
MINIMUM_WALL_GAP_CM = 45.0
RANGE_SAMPLES = 7


@dataclass
class Calibration:
    """Real degrees per odometry degree by direction and size, and burst durations by step."""

    turn_ratios: dict[str, dict[float, float]] = field(default_factory=dict)
    step_durations: dict[float, float] = field(default_factory=dict)
    forward_ratio: float | None = None
    reverse_ratio: float | None = None
    created_at: str | None = None
    log: list[str] = field(default_factory=list)

    def turn_ratio(self, degrees: float) -> float | None:
        """Interpolated ratio for a turn of this size and direction."""
        table = self.turn_ratios.get("left" if degrees > 0 else "right")
        if not table:
            return None
        sizes = sorted(table)
        size = abs(degrees)
        if size <= sizes[0]:
            return table[sizes[0]]
        if size >= sizes[-1]:
            return table[sizes[-1]]
        for low, high in zip(sizes, sizes[1:]):
            if low <= size <= high:
                fraction = (size - low) / (high - low)
                return table[low] + fraction * (table[high] - table[low])
        return None

    def mean_turn_ratio(self) -> float | None:
        values = [ratio for table in self.turn_ratios.values() for ratio in table.values()]
        return statistics.median(values) if values else None

    def step_duration(self, distance_cm: float) -> float | None:
        if not self.step_durations:
            return None
        sizes = sorted(self.step_durations)
        if distance_cm <= sizes[0]:
            return self.step_durations[sizes[0]] * distance_cm / sizes[0]
        if distance_cm >= sizes[-1]:
            return self.step_durations[sizes[-1]] * distance_cm / sizes[-1]
        for low, high in zip(sizes, sizes[1:]):
            if low <= distance_cm <= high:
                fraction = (distance_cm - low) / (high - low)
                return self.step_durations[low] + fraction * (self.step_durations[high] - self.step_durations[low])
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "turn_ratios": {
                direction: {str(size): round(ratio, 4) for size, ratio in table.items()}
                for direction, table in self.turn_ratios.items()
            },
            "step_durations": {str(size): round(duration, 4) for size, duration in self.step_durations.items()},
            "forward_ratio": None if self.forward_ratio is None else round(self.forward_ratio, 4),
            "reverse_ratio": None if self.reverse_ratio is None else round(self.reverse_ratio, 4),
            "created_at": self.created_at,
            "log": self.log[-80:],
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "Calibration":
        return cls(
            {
                direction: {float(size): float(ratio) for size, ratio in table.items()}
                for direction, table in dict(data.get("turn_ratios") or {}).items()
            },
            {float(size): float(duration) for size, duration in dict(data.get("step_durations") or {}).items()},
            data.get("forward_ratio"),
            data.get("reverse_ratio"),
            data.get("created_at"),
            list(data.get("log") or []),
        )


class CalibrationStore:
    """Keeps the calibration in a JSON file beside the map database."""

    def __init__(self, path: Path | None) -> None:
        self._path = path

    def load(self) -> Calibration | None:
        if self._path is None or not self._path.exists():
            return None
        try:
            return Calibration.from_json(json.loads(self._path.read_text()))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Ignoring unreadable calibration %s: %s", self._path, exc)
            return None

    def save(self, calibration: Calibration) -> None:
        if self._path is None:
            return
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(calibration.to_json(), indent=1))
        temporary.replace(self._path)


class Rig(Protocol):
    """What calibration needs from the mapping controller."""

    def heading_odometry(self) -> float: ...

    def distance_odometry(self) -> float: ...

    def turn(self, degrees: float, rotation_scale: float) -> bool: ...

    def burst(self, throttle: float, seconds: float) -> bool: ...

    def photograph(self, pan_offsets: Sequence[int]) -> dict[int, tuple[bytes, ...]]: ...

    def measure_turn(self, strip: dict[int, tuple[bytes, ...]], current: tuple[bytes, ...] | None) -> float | None: ...

    def range_ahead(self) -> float | None: ...

    def clearance_all_round(self) -> float | None: ...

    def stopped(self) -> bool: ...

    def report(self, message: str) -> None: ...


def run_calibration(rig: Rig, previous: Calibration | None, step_seconds_guess: Callable[[float], float]) -> Calibration:
    calibration = Calibration(created_at=datetime.now(timezone.utc).isoformat())

    def note(message: str) -> None:
        calibration.log.append(message)
        rig.report(message)
        logger.info("Calibration: %s", message)

    clearance = rig.clearance_all_round()
    if clearance is None or clearance < 25.0:
        raise RuntimeError("Calibration needs at least 25 cm of space all round the car")

    _calibrate_turns(rig, calibration, previous, note)
    if rig.stopped():
        return calibration
    _calibrate_steps(rig, calibration, previous, step_seconds_guess, note)
    return calibration


def _calibrate_turns(rig: Rig, calibration: Calibration, previous: Calibration | None, note: Callable[[str], None]) -> None:
    offsets = list(range(-60, 61, STRIP_STEP_DEGREES))
    # The starting view, to come back to afterwards: the step tests need the
    # car square to the wall again.
    home = rig.photograph(offsets)
    _turn_tests(rig, calibration, previous, note, offsets)
    for _ in range(3):
        if rig.stopped():
            return
        away = rig.measure_turn(home, rig.photograph([0]).get(0))
        if away is None:
            note("could not see the starting view again; step tests may be less accurate")
            return
        if abs(away) <= TURN_TOLERANCE_DEGREES:
            note(f"back at the starting heading ({away:+.1f}°)")
            return
        ratio = calibration.turn_ratio(-away) or (previous.turn_ratio(-away) if previous else None) or 1.0
        if not rig.turn(-away, ratio):
            return


def _turn_tests(
    rig: Rig,
    calibration: Calibration,
    previous: Calibration | None,
    note: Callable[[str], None],
    offsets: list[int],
) -> None:
    for size in TURN_SIZES_DEGREES:
        # Alternate directions so the car ends up facing about where it started.
        for direction, sign in (("right", -1.0), ("left", 1.0)):
            if rig.stopped():
                return
            target = sign * size
            ratio = (previous.turn_ratio(target) if previous else None) or 1.0
            for attempt in range(1, TURN_ATTEMPTS + 1):
                strip = rig.photograph(offsets)
                before = rig.heading_odometry()
                if not rig.turn(target, ratio):
                    return
                turned = rig.heading_odometry() - before
                actual = rig.measure_turn(strip, rig.photograph([0]).get(0))
                if actual is None or abs(turned) < 1.0:
                    note(f"{direction} {size:.0f}°: camera could not measure the turn (try more light or texture)")
                    break
                ratio = actual / turned
                error = actual - target
                note(
                    f"{direction} {size:.0f}° try {attempt}: wheels {abs(turned):.1f}°, "
                    f"camera {abs(actual):.1f}° (error {error:+.1f}°)"
                )
                calibration.turn_ratios.setdefault(direction, {})[size] = ratio
                if abs(error) <= TURN_TOLERANCE_DEGREES:
                    break


def _range(rig: Rig) -> float | None:
    samples = [value for value in (rig.range_ahead() for _ in range(RANGE_SAMPLES)) if value is not None]
    return statistics.median(samples) if len(samples) >= RANGE_SAMPLES // 2 else None


def _calibrate_steps(
    rig: Rig,
    calibration: Calibration,
    previous: Calibration | None,
    step_seconds_guess: Callable[[float], float],
    note: Callable[[str], None],
) -> None:
    forward_ratios: list[float] = []
    reverse_ratios: list[float] = []
    for size in STEP_SIZES_CM:
        seconds = (previous.step_duration(size) if previous else None) or step_seconds_guess(size)
        tried: list[tuple[float, float]] = []  # (seconds, centimetres moved)
        for attempt in range(1, STEP_ATTEMPTS + 1):
            if rig.stopped():
                return
            start = _range(rig)
            if start is None:
                note("no steady sonar reading straight ahead; face a flat wall about 1 m away")
                return
            if start - size < MINIMUM_WALL_GAP_CM:
                note(f"wall only {start:.0f} cm ahead; stopping forward tests to keep {MINIMUM_WALL_GAP_CM:.0f} cm clear")
                return
            odometry_before = rig.distance_odometry()
            if not rig.burst(1.0, seconds):
                return
            odometry_forward = rig.distance_odometry() - odometry_before
            after = _range(rig)
            if after is None:
                note("lost the wall reading after moving; stopping forward tests")
                return
            moved = start - after
            # Go back the same way; this also measures reverse travel.
            odometry_before = rig.distance_odometry()
            if not rig.burst(-1.0, seconds):
                return
            odometry_back = odometry_before - rig.distance_odometry()
            back = _range(rig)
            if moved > 0.5 and odometry_forward > 0.5:
                forward_ratios.append(moved / odometry_forward)
            if back is not None and odometry_back > 0.5 and back - after > 0.5:
                reverse_ratios.append((back - after) / odometry_back)
            note(f"{size:.0f} cm step try {attempt}: {seconds:.3f} s moved {moved:.1f} cm")
            if moved <= 0.5:
                seconds = min(STEP_DURATION_LIMITS_SECONDS[1], seconds * 2)
                continue
            calibration.step_durations[size] = seconds
            if abs(moved - size) <= max(STEP_TOLERANCE_MIN_CM, STEP_TOLERANCE_FRACTION * size):
                break
            tried.append((seconds, moved))
            low, high = STEP_DURATION_LIMITS_SECONDS
            last_seconds, last_moved = tried[-1]
            proposal = last_seconds * size / last_moved
            if len(tried) > 1:
                earlier_seconds, earlier_moved = tried[-2]
                if abs(last_moved - earlier_moved) > 0.5 and abs(last_seconds - earlier_seconds) > 1e-3:
                    # Travel is roughly linear in burst length but with an offset
                    # (dead zone, ramps), so fit a line through the last two tries.
                    slope = (last_moved - earlier_moved) / (last_seconds - earlier_seconds)
                    if slope > 0:
                        proposal = last_seconds + (size - last_moved) / slope
            seconds = min(high, max(low, proposal))
            calibration.step_durations[size] = seconds
    if forward_ratios:
        calibration.forward_ratio = statistics.median(forward_ratios)
    if reverse_ratios:
        calibration.reverse_ratio = statistics.median(reverse_ratios)
    if calibration.forward_ratio is not None:
        note(f"forward travel is {calibration.forward_ratio:.2f}x what the wheels report")
