import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robocar.calibration import MINIMUM_WALL_GAP_CM, Calibration, CalibrationStore, _calibrate_steps
from robocar.localization import compose
from test_places import SimulatedHouse


def true_turn(odometry_degrees: float) -> float:
    """A car whose small turns overshoot more, and whose left turns run hotter."""
    size = abs(odometry_degrees)
    factor = 1.30 - 0.15 * min(1.0, max(0.0, (size - 15) / 30))
    factor *= 1.05 if odometry_degrees > 0 else 0.97
    return odometry_degrees * factor


def true_step(seconds: float) -> float:
    """Forward travel with a dead zone, unlike the odometry's model."""
    return max(0.0, 105.0 * (seconds + 0.25) / 2 - 8.0)


def real_step(seconds: float) -> float:
    """Roughly what the real car's bursts covered: nothing below 0.08 s."""
    return max(0.0, 90.0 * (seconds - 0.08))


class StepRig:
    """Just enough car for the step tests: a wall ahead, a dead zone, reverse
    that falls short, and one sonar glitch."""

    def __init__(self, wall_cm: float = 110.0) -> None:
        self.position = 0.0
        self.wall_cm = wall_cm
        # The car slewing during a burst leaves the sonar on something else
        # for a whole reading, not just one ping.
        self.glitches = [165.0] * 7
        self.closest = wall_cm

    def distance_odometry(self) -> float:
        return self.position

    def burst(self, throttle: float, seconds: float) -> bool:
        travel = real_step(seconds) * (1.0 if throttle > 0 else 0.85)
        self.position += travel if throttle > 0 else -travel
        self.closest = min(self.closest, self.wall_cm - self.position)
        return True

    def range_ahead(self) -> float | None:
        if self.position > 0 and self.glitches:
            return self.glitches.pop()
        return self.wall_cm - self.position

    def stopped(self) -> bool:
        return False


class CalibrationTableTests(unittest.TestCase):
    def test_turn_ratios_are_interpolated_by_size_and_direction(self):
        calibration = Calibration({"left": {15.0: 1.4, 45.0: 1.2}, "right": {15.0: 1.3, 45.0: 1.1}})
        self.assertAlmostEqual(calibration.turn_ratio(30.0), 1.3)
        self.assertAlmostEqual(calibration.turn_ratio(-30.0), 1.2)
        self.assertAlmostEqual(calibration.turn_ratio(5.0), 1.4)
        self.assertAlmostEqual(calibration.turn_ratio(-90.0), 1.1)

    def test_step_durations_are_interpolated_and_extrapolated(self):
        calibration = Calibration(step_durations={10.0: 0.1, 30.0: 0.4})
        self.assertAlmostEqual(calibration.step_duration(20.0), 0.25)
        self.assertAlmostEqual(calibration.step_duration(5.0), 0.05)
        # Past the largest step it follows the last two, keeping the dead zone.
        self.assertAlmostEqual(calibration.step_duration(60.0), 0.85)

    def test_calibration_survives_a_round_trip_through_its_file(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CalibrationStore(Path(directory) / "calibration.json")
            self.assertIsNone(store.load())
            store.save(Calibration({"left": {15.0: 1.4}}, {20.0: 0.2}, 0.9, 0.95, "2026-10-24T00:00:00+00:00"))
            loaded = store.load()
            self.assertEqual(loaded.turn_ratios, {"left": {15.0: 1.4}})
            self.assertEqual(loaded.step_durations, {20.0: 0.2})
            self.assertEqual(loaded.forward_ratio, 0.9)


class CalibrationRunTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("SCAN_FIRST_SETTLE_SECONDS", 0.0),
            ("SCAN_STEP_SETTLE_SECONDS", 0.0),
            ("SCAN_SAMPLE_INTERVAL_SECONDS", 0.0),
            ("SCAN_SAMPLES_PER_ANGLE", 1),
            ("AUTO_ALIGN_SETTLE_SECONDS", 0.0),
        ):
            patcher = patch(f"robocar.mapping.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Two sizes of each keep the test quick.
        for name, value in (("TURN_SIZES_DEGREES", (15.0, 45.0)), ("STEP_SIZES_CM", (20.0, 30.0))):
            patcher = patch(f"robocar.calibration.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_calibration(self, house: SimulatedHouse) -> None:
        mapping = house.mapping

        def turn_by(degrees, rotation_scale=None):
            odometry_degrees = degrees / (rotation_scale or 1.0)
            house.odometry = compose(house.odometry, (0.0, 0.0, odometry_degrees))
            house.truth = compose(house.truth, (0.0, 0.0, true_turn(odometry_degrees)))
            return True

        def move_burst(throttle, seconds, *, steering=0):
            commanded = math.copysign(121.9 * 0.9 * (seconds + 0.25) / 2, throttle)
            house.odometry = compose(house.odometry, (0.0, commanded, 0.0))
            house.truth = compose(house.truth, (0.0, math.copysign(true_step(seconds), throttle), 0.0))
            return True

        with patch.object(mapping, "_turn_by", side_effect=turn_by), patch.object(mapping, "_move_burst", side_effect=move_burst):
            mapping._enabled = True
            mapping._run_calibration()

    def test_calibration_learns_how_the_car_really_turns_and_moves(self):
        house = SimulatedHouse(self.directory / "mapping.sqlite3")
        try:
            # Facing the front wall about 110 cm away.
            house.begin_session("calibration", (0.0, 0.0, 0.0))
            self.run_calibration(house)
            self.assertEqual(house.mapping.status()["state"], "calibrated")
            calibration = house.mapping._calibration
            for size in (15.0, 45.0):
                for degrees in (size, -size):
                    truth_ratio = true_turn(degrees / 1.25) / (degrees / 1.25)
                    self.assertAlmostEqual(calibration.turn_ratio(degrees), truth_ratio, delta=0.08, msg=degrees)
            for size, seconds in calibration.step_durations.items():
                self.assertAlmostEqual(true_step(seconds), size, delta=max(1.0, 0.07 * size), msg=size)
            self.assertTrue((self.directory / "calibration.json").exists())
        finally:
            house.close()

        # A restarted controller starts from the calibration.
        reopened = SimulatedHouse(self.directory / "mapping.sqlite3")
        try:
            self.assertIsNotNone(reopened.mapping.calibration())
            self.assertGreater(reopened.mapping._odometry_scale[1], 1.15)
            self.assertAlmostEqual(reopened.mapping._step_seconds(), calibration.step_duration(25.0), delta=1e-3)
        finally:
            reopened.close()

    def test_calibration_refuses_when_the_car_is_hemmed_in(self):
        house = SimulatedHouse(self.directory / "mapping.sqlite3")
        try:
            # Right against the front wall.
            house.begin_session("cramped", (0.0, 95.0, 0.0))
            self.run_calibration(house)
            self.assertEqual(house.mapping.status()["state"], "error")
            self.assertIn("25 cm", house.mapping.status()["error"])
            self.assertIsNone(house.mapping.calibration())
        finally:
            house.close()


if __name__ == "__main__":
    unittest.main()


class StepSearchTests(unittest.TestCase):
    def test_steps_converge_without_creeping_into_the_wall(self):
        rig = StepRig()
        calibration = Calibration()
        notes: list[str] = []
        # A poor first guess, like the real car's dead zone gave.
        _calibrate_steps(rig, calibration, None, lambda size: 0.06, notes.append)
        self.assertTrue(any("sonar jumped" in line for line in notes), notes)
        for size, seconds in calibration.step_durations.items():
            self.assertAlmostEqual(real_step(seconds), size, delta=max(1.0, 0.07 * size) + 0.5, msg=notes)
        # Never further than the largest step from where it started.
        self.assertGreater(rig.closest, rig.wall_cm - 20.0 - 3.0)
        self.assertGreater(rig.closest, MINIMUM_WALL_GAP_CM)
        self.assertLess(abs(rig.position), 3.0)
