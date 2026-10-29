import time
import unittest
import unittest.mock

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
                    "fault": None,
                    "distance_cm": None,
                    "error": None,
                    "updated_at": None,
                },
            )
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()


class SmoothingTests(unittest.TestCase):
    def test_kit_sensor_is_rebuilt_with_a_short_median(self):
        import sys
        import tempfile
        import types
        from pathlib import Path

        from robocar.range_sensor import RANGE_QUEUE_LENGTH, FreenoveRangeSensor

        created = []

        class DistanceSensor:
            def __init__(self, echo, trigger, max_distance=1, queue_len=9, partial=False):
                self.arguments = (echo, trigger, max_distance, queue_len, partial)
                self.closed = False
                created.append(self)
                self.distance = 0.5

            def close(self):
                self.closed = True

        class Ultrasonic:
            def __init__(self):
                self.echo_pin, self.trigger_pin, self.max_distance = 22, 27, 3.0
                self.sensor = DistanceSensor(22, 27, 3.0)

            def get_distance(self):
                return round(self.sensor.distance * 100, 1)

            def close(self):
                self.sensor.close()

        with tempfile.TemporaryDirectory() as directory, unittest.mock.patch.dict(
            sys.modules,
            {
                "ultrasonic": types.SimpleNamespace(Ultrasonic=Ultrasonic),
                "gpiozero": types.SimpleNamespace(DistanceSensor=DistanceSensor),
            },
        ):
            sensor = FreenoveRangeSensor(Path(directory))
            try:
                self.assertTrue(created[0].closed)
                self.assertEqual(created[1].arguments, (22, 27, 3.0, RANGE_QUEUE_LENGTH, True))
                self.assertAlmostEqual(sensor.settle_seconds, RANGE_QUEUE_LENGTH * 0.06 + 0.02)
                self.assertEqual(sensor.get_distance(), 50.0)
            finally:
                sensor.close()
