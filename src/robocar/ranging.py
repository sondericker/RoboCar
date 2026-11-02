"""The range sensors on the pan/tilt head, together: the ultrasonic, and the
TF-Luna lidar when fitted.

For safety the nearer of the two counts: the sonar's wide beam catches a
chair leg beside the line and glass the light passes through, the lidar
dark fabric that soaks up sound. For measuring (the cat, a calibration step)
the lidar's narrow beam is preferred, with the sonar as a fallback.
"""

import logging

from robocar.range_sensor import RangeSensorController

logger = logging.getLogger(__name__)

# Pan positions for each sweep: 40 directions 3 degrees apart, overlapping the
# ultrasonic beam so surfaces show up as continuous range profiles. The head
# first parks at the lead-in position and then only ever steps upward, so
# every reading is reached from the same side and servo backlash cancels out.
SCAN_LEAD_IN_PAN = 30
SCAN_ANGLES = tuple(range(33, 151, 3))
SCAN_FIRST_SETTLE_SECONDS = 0.3
SCAN_STEP_SETTLE_SECONDS = 0.07
SCAN_SAMPLES_PER_ANGLE = 3
SCAN_SAMPLE_INTERVAL_SECONDS = 0.05
# With the lidar: a reading every degree across the head's whole range
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
