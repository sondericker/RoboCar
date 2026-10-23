import time
import unittest

from robocar.drive import DriveController, DriveInhibitedError, PanTiltController, SimulatedDriver, SimulatedPanTilt


class FlakyDriver(SimulatedDriver):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 0

    def set_motor_model(self, left_front: int, left_rear: int, right_front: int, right_rear: int) -> None:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise OSError("I2C write failed")
        super().set_motor_model(left_front, left_rear, right_front, right_rear)


class DriveControllerTests(unittest.TestCase):
    def wait_for_command(self, driver: SimulatedDriver, expected: tuple[int, int, int, int]) -> None:
        deadline = time.monotonic() + 1
        while driver.command != expected and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(driver.command, expected)

    def test_mecanum_forward_strafe_and_rotation_mix(self):
        driver = SimulatedDriver()
        drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=False)
        try:
            self.assertEqual(drive.set_command(1, 0, 0), (1, 0, 0))
            self.wait_for_command(driver, (1000, 1000, 1000, 1000))
            drive.set_command(0, 1, 0)
            self.wait_for_command(driver, (1000, -1000, -1000, 1000))
            drive.set_command(0, 0, 1)
            self.wait_for_command(driver, (-1000, -1000, 1000, 1000))
            drive.set_command(1, 1, 1)
            self.wait_for_command(driver, (333, -333, 333, 1000))
            drive.set_command(2, 0, -2)
            self.wait_for_command(driver, (1000, 1000, 0, 0))
        finally:
            drive.close()

    def test_motor_output_ramps_up_and_down_over_quarter_second(self):
        driver = SimulatedDriver()
        drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=False)
        try:
            drive.set_command(1, 0, 0)
            time.sleep(0.12)
            self.assertTrue(all(0 < duty < 1000 for duty in driver.command))
            self.wait_for_command(driver, (1000, 1000, 1000, 1000))

            drive.stop()
            time.sleep(0.12)
            self.assertTrue(all(0 < duty < 1000 for duty in driver.command))
            self.wait_for_command(driver, (0, 0, 0, 0))
        finally:
            drive.close()

    def test_watchdog_stops_stale_nonzero_command(self):
        driver = SimulatedDriver()
        drive = DriveController(driver, max_duty=1000, command_timeout=0.05, physical=False)
        try:
            drive.set_command(1, 0, 0)
            time.sleep(0.02)
            self.assertTrue(any(duty > 0 for duty in driver.command))
            self.wait_for_command(driver, (0, 0, 0, 0))
        finally:
            drive.close()

    def test_output_failure_stops_motors(self):
        driver = FlakyDriver()
        drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=False)
        try:
            drive.set_command(1, 0, 0)
            self.wait_for_command(driver, (1000, 1000, 1000, 1000))
            driver.failures_remaining = 1
            with self.assertLogs("robocar.drive", level="ERROR"):
                drive.set_command(-1, 0, 0)
                self.wait_for_command(driver, (0, 0, 0, 0))
            with self.assertRaises(RuntimeError):
                drive.set_command(1, 0, 0)
        finally:
            drive.close()

    def test_stop_writes_zero_after_output_worker_fails(self):
        driver = FlakyDriver()
        drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=False)
        try:
            drive.set_command(1, 0, 0)
            self.wait_for_command(driver, (1000, 1000, 1000, 1000))
            driver.failures_remaining = 2
            with self.assertLogs("robocar.drive", level="ERROR"):
                drive.set_command(-1, 0, 0)
                deadline = time.monotonic() + 1
                while drive.status()["output_error"] is None and time.monotonic() < deadline:
                    time.sleep(0.01)
            self.assertEqual(driver.command, (1000, 1000, 1000, 1000))
            drive.stop()
            self.assertEqual(driver.command, (0, 0, 0, 0))
        finally:
            drive.close()

    def test_odometry_integrates_ramped_output_in_each_direction(self):
        driver = SimulatedDriver()
        drive = DriveController(driver, max_duty=4095, command_timeout=1, physical=False)
        try:
            for command in ((1, 0, 0), (0, 0, 1), (0, 1, 0)):
                before = drive.odometry()
                drive.set_command(*command)
                time.sleep(0.25)
                drive.stop()
                self.assertTrue(drive.wait_until_stopped(timeout=1))
                after = drive.odometry()
                if command == (1, 0, 0):
                    # Forward along +y; about 27 cm for a ramped 0.25 s burst.
                    self.assertGreater(after[1] - before[1], 15)
                    self.assertLess(after[1] - before[1], 40)
                    self.assertAlmostEqual(after[0], before[0], delta=0.01)
                elif command == (0, 0, 1):
                    # Positive steering turns left (counterclockwise) in place.
                    self.assertGreater(after[2] - before[2], 5)
                    self.assertAlmostEqual(after[1], before[1], delta=0.01)
                else:
                    # Positive strafe slides right; after the small left turn
                    # that is still mostly +x.
                    self.assertGreater(after[0] - before[0], 10)
        finally:
            drive.close()

    def test_mapping_inhibition_blocks_motion_until_released(self):
        driver = SimulatedDriver()
        drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=False)
        try:
            self.assertTrue(drive.set_inhibited(True))
            with self.assertRaises(DriveInhibitedError):
                drive.set_command(1, 0, 0)
            self.assertEqual(driver.command, (0, 0, 0, 0))
            drive.stop()
            self.assertTrue(drive.set_inhibited(False))
            drive.set_command(1, 0, 0)
            self.wait_for_command(driver, (1000, 1000, 1000, 1000))
        finally:
            drive.close()

    def test_mapping_inhibition_blocks_manual_pan_tilt(self):
        driver = SimulatedPanTilt()
        controller = PanTiltController(driver, physical=False)
        try:
            controller.set_inhibited(True)
            with self.assertRaises(RuntimeError):
                controller.set_angles(45, 90)
            self.assertEqual(driver.pan, 90)
            controller.set_angles(45, 90, override=True)
            self.assertEqual(driver.pan, 45)
        finally:
            controller.set_inhibited(False)
            controller.close()


if __name__ == "__main__":
    unittest.main()
