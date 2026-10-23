import time
import unittest

from robocar.range_sensor import RangeSensorController


class FakeRangeSensor:
    def __init__(self, distance: float | None = 42.5) -> None:
        self.distance = distance
        self.error: Exception | None = None
        self.closed = False

    def get_distance(self) -> float | None:
        if self.error is not None:
            raise self.error
        return self.distance

    def close(self) -> None:
        self.closed = True


class RangeSensorControllerTests(unittest.TestCase):
    def wait_for_status(self, controller: RangeSensorController, key: str, value: object) -> dict[str, object]:
        deadline = time.monotonic() + 1
        status = controller.status()
        while status[key] != value and time.monotonic() < deadline:
            time.sleep(0.01)
            status = controller.status()
        self.assertEqual(status[key], value)
        return status

    def test_reads_and_reports_distance_in_centimeters(self):
        driver = FakeRangeSensor()
        controller = RangeSensorController(driver, physical=True, sample_interval=0.01)
        try:
            status = self.wait_for_status(controller, "distance_cm", 42.5)
            self.assertEqual(status["mode"], "hardware")
            self.assertIsNone(status["error"])
            self.assertIsNotNone(status["updated_at"])
            driver.distance = 18.2
            self.wait_for_status(controller, "distance_cm", 18.2)
        finally:
            controller.close()
        self.assertTrue(driver.closed)

    def test_reports_missing_echo_and_sensor_errors(self):
        driver = FakeRangeSensor(distance=None)
        controller = RangeSensorController(driver, physical=True, sample_interval=0.01)
        try:
            self.wait_for_status(controller, "error", "No echo received")
            driver.error = OSError("GPIO unavailable")
            status = self.wait_for_status(controller, "error", "OSError: GPIO unavailable")
            self.assertIsNone(status["distance_cm"])
        finally:
            controller.close()

    def test_disabled_sensor_has_no_fabricated_reading(self):
        controller = RangeSensorController(None, physical=False)
        try:
            self.assertEqual(
                controller.status(),
                {
                    "enabled": False,
                    "mode": "disabled",
                    "distance_cm": None,
                    "error": None,
                    "updated_at": None,
                },
            )
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
