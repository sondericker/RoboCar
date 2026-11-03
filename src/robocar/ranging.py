"""The range sensors on the pan/tilt head, together: the ultrasonic, and the
TF-Luna lidar when fitted.

For safety the nearer of the two counts: the sonar's wide beam catches a
chair leg beside the line and glass the light passes through, the lidar
dark fabric that soaks up sound. For measuring (the cat, a calibration step)
the lidar's narrow beam is preferred, with the sonar as a fallback.
"""

import logging
import statistics

from robocar.drive import pan_bearing, pan_scale
from robocar.range_sensor import RangeSensorController

logger = logging.getLogger(__name__)

# Each sweep: the head first parks at the lead-in position and then only
# ever steps upward, so every reading is reached from the same side and servo
# backlash cancels out.
SCAN_LEAD_IN_PAN = 30
SCAN_FIRST_SETTLE_SECONDS = 0.3
# Single sonar readings (the sonar is not swept; it may watch for glass).
SCAN_SAMPLE_INTERVAL_SECONDS = 0.05
# A reading every degree across the head's whole range
# (121 of them, about 7 s a sweep), each the median of three frames.
LIDAR_SCAN_ANGLES = tuple(range(30, 151, 1))
LIDAR_STEP_SETTLE_SECONDS = 0.025
LIDAR_SAMPLES_PER_ANGLE = 3
LIDAR_SAMPLE_INTERVAL_SECONDS = 0.011
LIDAR_SAMPLE_SPREAD_CM = 8.0


class HeadRanging:
    def __init__(self, sonar: RangeSensorController, lidar: RangeSensorController | None = None) -> None:
        self.sonar = sonar
        self.lidar = lidar

    def lidar_ready(self) -> bool:
        return self.lidar is not None and bool(self.lidar.status()["enabled"])

    def sonar_ready(self) -> bool:
        return bool(self.sonar.status()["enabled"])

    def enabled(self) -> bool:
        """Whether there is anything to range with."""
        return self.lidar_ready() or self.sonar_ready()

    def settle_seconds(self, precise: bool = False) -> float:
        """How long after pointing the head a reading shows only the new
        direction (the sonar reports a running median; the lidar does not)."""
        lidar = self.lidar if self.lidar_ready() else None
        if lidar is not None and (precise or not self.sonar_ready()):
            return lidar.settle_seconds
        return max(self.sonar.settle_seconds, lidar.settle_seconds if lidar is not None else 0.0)

    def now(self, precise: bool = False) -> float | None:
        """A range where the head points, read now: the nearer of the two,
        or with `precise` the lidar's when it has one."""
        lidar = self.read(self.lidar) if self.lidar is not None and self.lidar_ready() else None
        if precise and lidar is not None:
            return lidar
        sonar = self.read(self.sonar) if self.sonar_ready() else None
        readings = [value for value in (lidar, sonar) if value is not None]
        return min(readings) if readings else None

    @staticmethod
    def read(sensor: RangeSensorController) -> float | None:
        """A reading, or None if the sensor failed this time (the lidar shares
        the motor board's I2C bus, where a transfer occasionally fails)."""
        try:
            return sensor.sample_now()
        except Exception as exc:
            logger.debug("Range reading failed: %s", exc)
            return None


# Learning the pan servo's scale from a turn on the spot: the lidar's view
# should move across the pan by exactly the gyro's turn, in real degrees.
PAN_SCALE_SEARCH = (0.7, 1.3)
PAN_SCALE_SEARCH_STEP = 0.005
PAN_SCALE_MIN_READINGS = 8
PAN_SCALE_MAX_ERROR_CM = 3.0
# Neighbouring readings further apart than this straddle an edge, where a
# small angle error makes a large range error: not used for matching.
PAN_SCALE_EDGE_CM = 10.0


def with_current_angles(scan: list[dict[str, object]]) -> list[dict[str, object]]:
    """A stored sweep with each reading's direction worked out from its pan
    by the current servo scale (older sweeps stored only the direction, as
    90 - pan)."""
    out = []
    for item in scan:
        if "angle_deg" not in item or ("pan" not in item and pan_scale() == 1.0):
            # Nothing to change: an old sweep read at the scale it was taken at.
            out.append(item)
            continue
        pan = item.get("pan", 90.0 - float(item["angle_deg"]))
        out.append({**item, "pan": pan, "angle_deg": round(pan_bearing(float(pan)), 2)})
    return out


def estimate_pan_scale(
    before: list[dict[str, object]], after: list[dict[str, object]], turn_deg: float
) -> tuple[float, float, int] | None:
    """(scale, median error cm, readings) that best explains how the lidar's
    view moved across the pan between two sweeps from the same spot with the
    car turned `turn_deg` (counterclockwise) between them; None if they do
    not overlap enough or nothing fits well."""

    def profile(scan: list[dict[str, object]]) -> dict[int, float]:
        return {
            int(item["pan"]): float(item["distance_cm"])
            for item in scan
            if item.get("sensor") == "lidar" and item.get("distance_cm") is not None and "pan" in item
        }

    first, second = profile(before), profile(after)
    if len(first) < PAN_SCALE_MIN_READINGS or len(second) < PAN_SCALE_MIN_READINGS:
        return None
    best: tuple[float, float, int] | None = None
    steps = int(round((PAN_SCALE_SEARCH[1] - PAN_SCALE_SEARCH[0]) / PAN_SCALE_SEARCH_STEP))
    for index in range(steps + 1):
        scale = PAN_SCALE_SEARCH[0] + index * PAN_SCALE_SEARCH_STEP
        # A counterclockwise turn moves what was seen toward the right: to
        # higher pans, by the turn in commanded degrees.
        shift = turn_deg / scale
        errors = []
        for pan, distance in second.items():
            source = pan - shift
            low = int(source // 1)
            if low not in first or low + 1 not in first or abs(first[low] - first[low + 1]) > PAN_SCALE_EDGE_CM:
                continue
            expected = first[low] + (source - low) * (first[low + 1] - first[low])
            errors.append(abs(distance - expected))
        if len(errors) >= PAN_SCALE_MIN_READINGS:
            error = statistics.median(errors)
            if best is None or error < best[1]:
                best = (round(scale, 3), error, len(errors))
    if best is None or best[1] > PAN_SCALE_MAX_ERROR_CM:
        return None
    return best
