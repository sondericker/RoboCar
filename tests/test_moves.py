"""Moves measured by the accelerometer, step calibration by it, the sonar
left off when the lidar does its job, and the lidar's fan ahead."""

import math
import random
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from robocar.calibration import run_calibration
from robocar.config import Config
from robocar.drive import DriveController, PanTiltController, SimulatedDriver, SimulatedPanTilt, pan_bearing
from robocar.imu import GRAVITY_CM_S2, ImuController, MoveMeasure
from robocar.mapping import AUTO_FORWARD_COMMAND, BURSTS_NEEDED, MappingController
from robocar.range_sensor import RangeSensorController
from robocar.store import MappingStore
from robocar.web import create_app


class Driver:
    model = "MPU-6500"
    sample_hz = 100.0

    def details(self):
        return {}

    def close(self):
        pass


def rotate(axis, angle, vector):
    cos, sin = math.cos(angle), math.sin(angle)
    along = sum(a * v for a, v in zip(axis, vector))
    cross = (axis[1] * vector[2] - axis[2] * vector[1], axis[2] * vector[0] - axis[0] * vector[2], axis[0] * vector[1] - axis[1] * vector[0])
    return tuple(vector[i] * cos + cross[i] * sin + axis[i] * along * (1 - cos) for i in range(3))


class BoardOnACar:
    """An IMU board on a car, tilted 6 degrees, its forward along -y: with a
    constant accelerometer offset, a 4% scale error, noise, roller vibration
    while moving, and the car pitching as it speeds up and slows down."""

    def __init__(self, seed=1, forward=(0.0, -1.0, 0.0)):
        self.rng = random.Random(seed)
        self.now = 1000.0
        self.imu = ImuController(Driver(), start=False, clock=lambda: self.now)
        self.up = rotate((1.0, 0.0, 0.0), math.radians(6), (0.0, 0.0, 1.0))
        self.forward = forward
        u = self.up
        self.right = (forward[1] * u[2] - forward[2] * u[1], forward[2] * u[0] - forward[0] * u[2], forward[0] * u[1] - forward[1] * u[0])

    def sample(self, acceleration, pitch, pitch_rate, t, moving):
        gravity = rotate(self.right, pitch, self.up)
        accel = tuple(gravity[i] + acceleration[i] / GRAVITY_CM_S2 for i in range(3))
        shake = 0.3 * math.sin(2 * math.pi * 23 * t) if moving else 0.0
        noise = 0.05 if moving else 0.004
        offset = (0.02, -0.015, 0.01)
        accel = tuple(1.04 * accel[i] + offset[i] + noise * self.rng.gauss(0, 1) + (shake if i == 2 else 0.3 * shake) for i in range(3))
        # The board pitching one way makes gravity seem to turn the other.
        gyro = tuple(-self.right[i] * math.degrees(pitch_rate) + (3.2, -2.1, -1.1)[i] + 0.05 * self.rng.gauss(0, 1) for i in range(3))
        return gyro, accel

    def feed(self, count, sample):
        for index in range(count):
            gyro, accel = sample(index * 0.01)
            self.now += 0.01
            self.imu.ingest([gyro], None, self.now, accels=[accel])

    def rest(self, seconds=1.5):
        self.feed(int(seconds * 100), lambda t: self.sample((0.0, 0.0, 0.0), 0.0, 0.0, t, False))

    def move(self, speed=50.0, cruise=0.3, ahead=1.0, right=0.0, ramp=0.25, pitch_degrees=1.5):
        """A move at rest to rest; returns the measure and the true distance."""
        way = tuple(ahead * f + right * r for f, r in zip(self.forward, self.right))
        size = math.sqrt(sum(value * value for value in way))
        way = tuple(value / size for value in way)
        total = 2 * ramp + cruise
        rate = speed / ramp
        move = self.imu.begin_move()
        previous = [0.0]

        def sample(t):
            push = rate if t < ramp else 0.0 if t < ramp + cruise else -rate if t < total else 0.0
            pitch = -math.radians(pitch_degrees) * push / rate * ahead / math.hypot(ahead, right)
            pitch_rate = (pitch - previous[0]) / 0.01
            previous[0] = pitch
            return self.sample(tuple(push * w for w in way), pitch, pitch_rate, t, t < total)

        self.feed(int((total + 0.15) * 100), sample)
        return self.imu.end_move(move), speed * (ramp + cruise)


class MoveMeasureTests(unittest.TestCase):
    def test_learns_which_way_is_forward_then_measures_moves_any_way(self):
        board = BoardOnACar()
        board.rest()
        self.assertTrue(board.imu.ready())
        first, _ = board.move()
        self.assertIsNone(first.ahead_cm)
        self.assertTrue(board.imu.learn_forward(first, 1.0, 0.0))
        board.rest()
        for speed, cruise, ahead, right in ((50, 0.3, 1, 0), (30, 0.0, 1, 0), (60, 0.6, 1, 0), (40, 0.1, -1, 0), (40, 0.2, 0, 1), (40, 0.2, 0.7, 0.7)):
            measure, true = board.move(speed, cruise, ahead, right)
            share = math.hypot(ahead, right)
            self.assertAlmostEqual(measure.ahead_cm, true * ahead / share, delta=3.0)
            self.assertAlmostEqual(measure.right_cm, true * right / share, delta=3.0)
            self.assertLess(measure.end_speed_cm_s, 15.0)
            board.rest()

    def test_without_following_gravity_the_pitching_would_spoil_it(self):
        # The same moves with the gyro's pitching hidden from the tracker read
        # far longer: the tracker's work is what keeps them right.
        board = BoardOnACar()
        board.rest()
        first, _ = board.move()
        board.imu.learn_forward(first, 1.0, 0.0)
        board.rest()
        right_measure, true = board.move(pitch_degrees=3.0)
        board.rest()
        original = board.sample
        board.sample = lambda acc, pitch, rate, t, moving: (original(acc, pitch, 0.0, t, moving)[0], original(acc, pitch, rate, t, moving)[1])
        wrong_measure, _ = board.move(pitch_degrees=3.0)
        self.assertLess(abs(right_measure.ahead_cm - true), 3.0)
        self.assertGreater(abs(wrong_measure.ahead_cm - true), 2 * abs(right_measure.ahead_cm - true))

    def test_lost_samples_spoil_a_move(self):
        board = BoardOnACar()
        board.rest()
        move = board.imu.begin_move()
        board.imu.ingest([], None, board.now, overflowed=True)
        board.rest(0.2)
        self.assertTrue(board.imu.end_move(move).lost_samples)
        self.assertIsNone(board.imu.end_move(move))


class FakeImu:
    """An IMU that hands the drive whatever move a test says happened."""

    def __init__(self, ahead=None, right=0.0, end_speed=2.0):
        self.next = (ahead, right, end_speed)
        self.forward = (0.0, -1.0, 0.0) if ahead is not None else None
        self.learned = []

    def fitted(self):
        return True

    def ready(self):
        return True

    def heading_now(self):
        return 0.0

    def upset(self):
        return False

    def knock_since(self, since):
        return 0.0

    def rate_dps(self):
        return 0.0

    def speed_cm_s(self):
        return 30.0  # it moves when told

    def forward_axis(self):
        return self.forward

    def set_forward_axis(self, axis):
        self.forward = tuple(axis)

    def begin_move(self):
        return 1

    def end_move(self, move_id):
        ahead, right, end_speed = self.next
        known = self.forward is not None and ahead is not None
        plane = (ahead or 0.0, right or 0.0)
        return MoveMeasure(ahead if known else None, right if known else None, plane, 0.6, end_speed, 40.0, False)

    def learn_forward(self, measure, ahead_share, right_share):
        self.learned.append((ahead_share, right_share))
        self.forward = (0.0, -1.0, 0.0)
        return True


class Camera:
    def capture_snapshot(self):
        return b"jpeg"


def build(imu, folder=":memory:"):
    drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True, imu=imu)
    pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
    sonar = RangeSensorController(None, physical=False)
    mapping = MappingController(Camera(), drive, pan_tilt, sonar, MappingStore(Path(folder)))
    return mapping, drive, (mapping, drive, pan_tilt, sonar)


def close(parts):
    for part in parts:
        part.close()


class DriveMoveTests(unittest.TestCase):
    def test_a_measured_move_puts_the_odometry_where_the_car_went(self):
        mapping, drive, parts = build(FakeImu(ahead=12.0, right=-1.0))
        try:
            time.sleep(0.05)
            self.assertTrue(drive.measures_distance())
            self.assertTrue(mapping._move_burst(AUTO_FORWARD_COMMAND, 0.3))
            x, y, _ = drive.odometry()
            # Heading 0 faces +y; 1 cm to the left is -x.
            self.assertAlmostEqual(y, 12.0, places=3)
            self.assertAlmostEqual(x, -1.0, places=3)
            move = mapping.status()["last_move"]
            self.assertTrue(move["used"])
            self.assertGreater(move["wheels_cm"], 12.0)
            self.assertEqual(mapping.status()["distances_measured_by"], "accelerometer")
        finally:
            close(parts)

    def test_a_drifting_sum_is_not_trusted(self):
        mapping, drive, parts = build(FakeImu(ahead=12.0, end_speed=60.0))
        try:
            time.sleep(0.05)
            mapping._move_burst(AUTO_FORWARD_COMMAND, 0.3)
            self.assertFalse(mapping.status()["last_move"]["used"])
            self.assertGreater(drive.odometry()[1], 20.0)  # the wheels' guess stands
        finally:
            close(parts)

    def test_the_first_forward_move_teaches_which_way_is_forward(self):
        imu = FakeImu(ahead=10.0)
        imu.forward = None
        mapping, drive, parts = build(imu)
        try:
            time.sleep(0.05)
            self.assertFalse(drive.measures_distance())
            mapping._move_burst(AUTO_FORWARD_COMMAND, 0.3)
            self.assertEqual(imu.learned, [(AUTO_FORWARD_COMMAND, 0.0)])
            self.assertTrue(drive.measures_distance())
            # Not from an arc: which way it went is not known.
            mapping._move_burst(0.0, 0.3, steering=0.5)
            self.assertEqual(len(imu.learned), 1)
        finally:
            close(parts)

    def test_burst_lengths_are_learned_from_measured_moves(self):
        imu = FakeImu(ahead=10.0)
        mapping, drive, parts = build(imu)
        try:
            time.sleep(0.05)
            # This car goes 60 cm a second of burst, less 4 cm of dead zone.
            for seconds in (0.15, 0.3, 0.45, 0.25):
                imu.next = (60.0 * seconds - 4.0, 0.0, 2.0)
                mapping._move_burst(AUTO_FORWARD_COMMAND, seconds)
            self.assertGreaterEqual(len(mapping._bursts), BURSTS_NEEDED)
            self.assertAlmostEqual(mapping._step_seconds(20.0), 24.0 / 60.0, delta=0.01)
        finally:
            close(parts)


class FeelCalibrationTests(unittest.TestCase):
    def test_steps_are_calibrated_by_the_accelerometer_without_a_wall(self):
        class Rig:
            """A car that goes 70 cm per second of burst past a 0.05 s dead zone,
            with nothing ahead for the lidar to see."""

            def __init__(self):
                self.position = 0.0
                self.moves = []
                self.last = None

            def heading_odometry(self):
                return 0.0

            def distance_odometry(self):
                return self.position

            def turn(self, degrees, rotation_scale):
                raise AssertionError("no turns with a gyro")

            def burst(self, throttle, seconds):
                moved = math.copysign(70.0 * max(0.0, seconds - 0.05), throttle)
                self.position += moved
                self.moves.append(moved)
                # The first move only teaches the accelerometer which way it faces.
                self.last = None if len(self.moves) == 1 else moved
                return True

            def last_move_cm(self):
                return self.last

            def photograph(self, offsets):
                return {}

            def measure_turn(self, strip, current):
                return None

            def range_ahead(self):
                return None

            def clearance_all_round(self):
                return 100.0

            def stopped(self):
                return False

            def report(self, message):
                pass

        rig = Rig()
        result = run_calibration(rig, None, lambda cm: 0.2, turn_source="gyro", feel_steps=True)
        for size, seconds in result.step_durations.items():
            self.assertAlmostEqual(70.0 * (seconds - 0.05), size, delta=max(1.0, 0.07 * size), msg=size)
        self.assertEqual(set(result.step_durations), {5.0, 10.0, 20.0})
        self.assertLess(abs(rig.position), 6.0)
        self.assertTrue(any("accelerometer" in line for line in result.log))


class SonarTests(unittest.TestCase):
    def test_the_sonar_is_left_off_when_the_lidar_works(self):
        lidar = RangeSensorController(None, True, name="Lidar")
        lidar.status = lambda: {"enabled": True, "mode": "hardware"}
        opened = []
        with patch("robocar.web._open_lidar", return_value=lidar), patch(
            "robocar.web.FreenoveRangeSensor", side_effect=lambda path: opened.append(path)
        ), patch("robocar.web.FreenoveDriver", side_effect=OSError(121, "off")), patch(
            "robocar.web.FreenovePanTilt", side_effect=OSError(121, "off")
        ), patch("robocar.web.BoardADC", side_effect=OSError(121, "off")), patch(
            "robocar.web._open_imu", return_value=None
        ), patch("robocar.web.connected_models", return_value=[]):
            app = create_app(Config(mapping_database=Path(":memory:"), hardware_enabled=True, freenove_server_path=Path("/tmp")))
            try:
                self.assertEqual(opened, [])
                self.assertEqual(app.test_client().get("/api/status").get_json()["range"]["mode"], "disabled")
            finally:
                for name in ("robocar_mapping", "robocar_drive", "robocar_pan_tilt", "robocar_range_sensor", "robocar_camera", "robocar_power"):
                    app.extensions[name].close()
            # Asked for, it is used even with the lidar.
            app = create_app(
                Config(mapping_database=Path(":memory:"), hardware_enabled=True, freenove_server_path=Path("/tmp"), sonar="on")
            )
            try:
                self.assertEqual(len(opened), 1)
            finally:
                for name in ("robocar_mapping", "robocar_drive", "robocar_pan_tilt", "robocar_range_sensor", "robocar_camera", "robocar_power"):
                    app.extensions[name].close()

    def test_the_sonar_setting_is_read(self):
        with patch.dict("os.environ", {"ROBOCAR_SONAR": "on"}):
            self.assertEqual(Config.from_env().sonar, "on")
        with patch.dict("os.environ", {"ROBOCAR_SONAR": "loud"}), self.assertRaises(ValueError):
            Config.from_env()


class LidarFanTests(unittest.TestCase):
    def test_a_leg_off_to_one_side_still_blocks_the_way(self):
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)

        class Lidar:
            """A wall 150 cm ahead, and a chair leg 50 cm out, 10 cm left of the
            line the car would drive (13 degrees off straight ahead)."""

            def get_distance(self):
                bearing = pan_bearing(pan_tilt.status()["pan"])
                leg = math.degrees(math.atan2(10.0, 50.0 + 7.0))
                return 52.0 if abs(bearing - leg) <= 1.5 else 150.0

            def close(self):
                pass

        lidar = RangeSensorController(Lidar(), True, sample_interval=0.05, name="Lidar")
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        sonar = RangeSensorController(None, physical=False)
        mapping = MappingController(Camera(), drive, pan_tilt, sonar, MappingStore(Path(":memory:")), lidar=lidar)
        try:
            ahead = mapping._ping_ahead()
            # A single reading straight ahead would have said 150 cm.
            self.assertIsNotNone(ahead)
            self.assertLess(ahead, 60.0)
            self.assertEqual(pan_tilt.status()["pan"], 90)
        finally:
            for part in (mapping, drive, pan_tilt, sonar, lidar):
                part.close()


class EffortTests(unittest.TestCase):
    """Keeping the motors' torque up: for the battery, and whenever the IMU
    sees a command not moving the car."""

    class Imu:
        def __init__(self):
            self.moving = False

        def fitted(self):
            return True

        def ready(self):
            return True

        def heading_now(self):
            return 0.0

        def upset(self):
            return False

        def knock_since(self, since):
            return 0.0

        def rate_dps(self):
            return 0.0

        def speed_cm_s(self):
            return 30.0 if self.moving else 0.0

        def forward_axis(self):
            return None

    def test_a_low_battery_gets_more_duty_and_the_odometry_does_not_notice(self):
        tired = DriveController(SimulatedDriver(), max_duty=1800, command_timeout=1, physical=True, supply=lambda: 6.6, cells=2)
        fresh = DriveController(SimulatedDriver(), max_duty=1800, command_timeout=1, physical=True)
        try:
            for drive in (tired, fresh):
                drive.set_command(0.5, 0.0, 0.0)
            time.sleep(0.5)
            gain = 7.8 / 6.6
            self.assertAlmostEqual(tired._driver.command[0], round(0.5 * 1800 * gain), delta=2)
            self.assertAlmostEqual(fresh._driver.command[0], 900, delta=2)
            self.assertAlmostEqual(tired.status()["supply_gain"], round(gain, 3), places=3)
            # Both think they have gone about as far.
            self.assertAlmostEqual(tired.odometry()[1], fresh.odometry()[1], delta=0.15 * fresh.odometry()[1])
        finally:
            tired.close()
            fresh.close()

    def test_an_unknown_or_impossible_battery_reading_changes_nothing(self):
        for reading in (None, 0.4, 30.0):
            drive = DriveController(SimulatedDriver(), max_duty=1800, command_timeout=1, physical=True, supply=lambda: reading)
            try:
                drive.set_command(0.5, 0.0, 0.0)
                time.sleep(0.4)
                self.assertEqual(drive.status()["supply_gain"], 1.0)
            finally:
                drive.close()

    def test_more_effort_until_it_moves_and_the_next_start_begins_there(self):
        imu = self.Imu()
        drive = DriveController(SimulatedDriver(), max_duty=1800, command_timeout=5, physical=True, imu=imu)
        try:
            drive.set_command(0.3, 0.0, 0.0)
            time.sleep(1.15)
            self.assertAlmostEqual(drive.status()["effort_boost"], 1.25**2, places=3)
            self.assertAlmostEqual(drive._driver.command[0], round(0.3 * 1800 * 1.25**2), delta=3)
            imu.moving = True
            time.sleep(0.6)
            self.assertAlmostEqual(drive.status()["effort_boost"], 1.25**2, places=3)
            learned = drive.learned_boost()
            self.assertGreater(learned, 1.0)
            drive.set_command(0.0, 0.0, 0.0)
            drive.set_command(0.3, 0.0, 0.0)
            self.assertAlmostEqual(drive.status()["effort_boost"], learned, places=3)
        finally:
            drive.close()

    def test_it_stops_raising_the_effort_at_the_most(self):
        imu = self.Imu()
        drive = DriveController(SimulatedDriver(), max_duty=3000, command_timeout=10, physical=True, imu=imu)
        try:
            drive.set_command(0.9, 0.0, 0.0)
            time.sleep(3.0)
            self.assertEqual(drive.status()["effort_boost"], 2.0)
            # Never past the board's full duty.
            self.assertLessEqual(max(abs(value) for value in drive._driver.command), 4095)
        finally:
            drive.close()

    def test_the_imu_tells_moving_from_still(self):
        board = BoardOnACar()
        board.rest()
        self.assertEqual(board.imu.speed_cm_s(), 0.0)
        move = board.imu.begin_move()
        peak = 0.0
        for index in range(40):
            gyro, accel = board.sample(tuple(200.0 * f for f in board.forward), 0.0, 0.0, index * 0.01, True)
            board.now += 0.01
            board.imu.ingest([gyro], None, board.now, accels=[accel])
            peak = max(peak, board.imu.speed_cm_s())
        board.imu.end_move(move)
        self.assertGreater(peak, 20.0)
        board.rest()
        self.assertLess(board.imu.speed_cm_s(), 6.0)


if __name__ == "__main__":
    unittest.main()
