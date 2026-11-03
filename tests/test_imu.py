"""The gyro: the chip driver, the heading tracker, and how the drive, turns,
moves, localization and calibration use it."""

import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from robocar.calibration import Calibration, run_calibration
from robocar.config import Config
from robocar.drive import (
    TURN_DEGREES_PER_SECOND_AT_FULL_DUTY,
    DriveController,
    DriveInhibitedError,
    PanTiltController,
    SimulatedDriver,
    SimulatedPanTilt,
    TurnPower,
    TurnSteerer,
    bearing_pan,
    pan_bearing,
    pan_scale,
    set_pan_scale,
)
from robocar.imu import (
    ACCEL_LSB_PER_G,
    FIFO_EN_ACCEL_GYRO,
    GYRO_LSB_PER_DPS,
    INT_FIFO_OVERFLOW,
    MAX_READ_ERRORS,
    RECOVER_SECONDS,
    REG_FIFO_COUNT_H,
    REG_FIFO_EN,
    REG_FIFO_R_W,
    REG_INT_STATUS,
    REG_WHO_AM_I,
    ImuController,
    MPU6500,
)
from robocar.localization import predict_motion
from robocar.ranging import estimate_pan_scale, with_current_angles
from robocar.mapping import BUMP_STOP_G, GYRO_TURN_TOLERANCE_DEGREES, MappingController
from robocar.range_sensor import RangeSensorController
from robocar.store import MappingStore
from robocar.web import create_app


class FakeBus:
    """An MPU-6500 on an SMBus: identity, a FIFO of gyro samples, the accelerometer."""

    def __init__(self, identity: int = 0x70, address: int = 0x68) -> None:
        self.identity, self.address = identity, address
        self.registers: dict[int, int] = {}
        self.fifo = bytearray()
        self.overflow = False
        self.accel = (0, 0, 8192)
        self.closed = False

    def push(self, *samples: tuple[float, float, float], accel=(0.0, 0.0, 1.0)) -> None:
        """Gyro samples (degrees a second), each with the accelerometer's
        reading (g) ahead of it, as the FIFO holds them."""
        for sample in samples:
            for value, scale in [(a, ACCEL_LSB_PER_G) for a in accel] + [(g, GYRO_LSB_PER_DPS) for g in sample]:
                raw = int(round(value * scale)) & 0xFFFF
                self.fifo += bytes((raw >> 8, raw & 0xFF))

    def _check(self, address: int) -> None:
        if address != self.address:
            raise OSError(121, "Remote I/O error")

    def read_byte_data(self, address: int, register: int) -> int:
        self._check(address)
        if register == REG_WHO_AM_I:
            return self.identity
        if register == REG_INT_STATUS:
            status = INT_FIFO_OVERFLOW if self.overflow else 0
            self.overflow = False
            return status
        return self.registers.get(register, 0)

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self._check(address)
        self.registers[register] = value
        if register == 0x6A and value & 0x04:
            self.fifo.clear()

    def read_i2c_block_data(self, address: int, register: int, length: int) -> list[int]:
        self._check(address)
        if register == REG_FIFO_COUNT_H:
            return [len(self.fifo) >> 8, len(self.fifo) & 0xFF]
        if register == REG_FIFO_R_W:
            data, self.fifo = list(self.fifo[:length]), self.fifo[length:]
            return data
        if register == 0x3B:
            out = []
            for value in self.accel:
                raw = value & 0xFFFF
                out += [raw >> 8, raw & 0xFF]
            return out[:length]
        return [0] * length

    def close(self) -> None:
        self.closed = True


class FakeDriver:
    """Gyro samples and accelerometer readings handed to the tracker by a test."""

    model = "MPU-6500"
    sample_hz = 100.0

    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float]] = []
        self.accel = (0.0, 0.0, 1.0)
        self.fail = False
        self.configured = 0

    def reset_fifo(self) -> None:
        pass

    def configure(self) -> None:
        self.configured += 1
        if self.fail:
            raise OSError(121, "Remote I/O error")

    def read_samples(self):
        if self.fail:
            raise OSError(121, "Remote I/O error")
        samples, self.samples = self.samples, []
        return [(sample, self.accel) for sample in samples], False

    def details(self):
        return {"model": self.model, "bus": 2, "address": "0x68"}

    def close(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def feed(imu: ImuController, driver: FakeDriver, clock: Clock, seconds: float, rate=(0.0, 0.0, 0.0), bias=(0.0, 0.0, 0.0), noise: float = 0.0, step: int = 2) -> None:
    """Run the tracker for `seconds` of 100 Hz samples, two at a time."""
    count = int(round(seconds * 100))
    for index in range(0, count, step):
        batch = min(step, count - index)
        for offset in range(batch):
            wobble = noise * math.sin(1.7 * (index + offset))
            driver.samples.append(tuple(r + b + wobble for r, b in zip(rate, bias)))
        clock.now += batch / 100
        imu.step()


class MPU6500Tests(unittest.TestCase):
    def test_finds_the_chip_and_reads_its_fifo_in_degrees_a_second(self):
        bus = FakeBus()
        with patch("robocar.imu.time.sleep"):
            mpu = MPU6500(2, smbus=bus)
        self.assertEqual(mpu.model, "MPU-6500")
        self.assertEqual(mpu.address, 0x68)
        self.assertEqual(bus.registers[REG_FIFO_EN], FIFO_EN_ACCEL_GYRO)
        bus.push((10.0, -250.0, 1990.0), (0.0, 0.0, -0.5), accel=(-0.5, 0.0, 1.0))
        samples, overflowed = mpu.read_samples()
        self.assertFalse(overflowed)
        self.assertEqual(len(samples), 2)
        for got, want in zip(samples[0][0] + samples[1][0], (10.0, -250.0, 1990.0, 0.0, 0.0, -0.5)):
            self.assertAlmostEqual(got, want, delta=0.07)
        self.assertEqual(samples[0][1], (-0.5, 0.0, 1.0))

    def test_reads_more_samples_than_one_block_holds(self):
        bus = FakeBus()
        with patch("robocar.imu.time.sleep"):
            mpu = MPU6500(2, smbus=bus)
        bus.push(*[(float(i), 0.0, 0.0) for i in range(12)])
        samples, _ = mpu.read_samples()
        self.assertEqual([round(gyro[0]) for gyro, _ in samples], list(range(12)))

    def test_an_overflowed_fifo_is_emptied_and_reported(self):
        bus = FakeBus()
        with patch("robocar.imu.time.sleep"):
            mpu = MPU6500(2, smbus=bus)
        bus.push((1.0, 2.0, 3.0))
        bus.overflow = True
        samples, overflowed = mpu.read_samples()
        self.assertTrue(overflowed)
        self.assertEqual(samples, [])
        self.assertEqual(len(bus.fifo), 0)

    def test_the_second_address_and_a_missing_chip(self):
        with patch("robocar.imu.time.sleep"):
            self.assertEqual(MPU6500(2, smbus=FakeBus(0x71, 0x69)).model, "MPU-9250")
            missing = FakeBus(address=0x50)
            with self.assertRaises(RuntimeError):
                MPU6500(2, smbus=missing)
        self.assertTrue(missing.closed)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.driver = FakeDriver()
        self.clock = Clock()
        self.imu = ImuController(self.driver, start=False, clock=self.clock)

    def test_measures_its_zero_while_still_then_counts_turns(self):
        bias = (3.2, -2.1, -1.05)
        self.assertFalse(self.imu.ready())
        feed(self.imu, self.driver, self.clock, 1.0, bias=bias, noise=0.05)
        self.assertTrue(self.imu.ready())
        self.assertTrue(self.imu.still())
        self.assertAlmostEqual(self.imu.heading_now(), 0.0, delta=0.05)
        # A minute parked: the heading is held whatever the zero does.
        feed(self.imu, self.driver, self.clock, 60.0, bias=bias, noise=0.05)
        self.assertAlmostEqual(self.imu.heading_now(), 0.0, delta=0.05)
        # A quarter turn left at 90 degrees a second.
        feed(self.imu, self.driver, self.clock, 1.0, rate=(0.0, 0.0, 90.0), bias=bias)
        feed(self.imu, self.driver, self.clock, 1.0, bias=bias)
        self.assertAlmostEqual(self.imu.heading_now(), 90.0, delta=0.5)
        # And back to the right.
        feed(self.imu, self.driver, self.clock, 0.5, rate=(0.0, 0.0, -180.0), bias=bias)
        feed(self.imu, self.driver, self.clock, 1.0, bias=bias)
        self.assertAlmostEqual(self.imu.heading_now(), 0.0, delta=0.5)

    def test_the_start_of_a_turn_is_not_lost_to_stillness(self):
        feed(self.imu, self.driver, self.clock, 1.0)
        self.assertTrue(self.imu.still())
        # A turn that speeds up from nothing, as the motors ramp.
        for index in range(30):
            self.driver.samples.append((0.0, 0.0, 10.0 * index))
            self.clock.now += 0.01
            self.imu.step()
        feed(self.imu, self.driver, self.clock, 1.0)
        self.assertAlmostEqual(self.imu.heading_now(), sum(10.0 * i for i in range(30)) / 100, delta=0.2)

    def test_turns_about_whichever_way_is_up(self):
        # Board mounted on its side: gravity along -x, so -x is up.
        self.driver.accel = (-1.0, 0.0, 0.0)
        feed(self.imu, self.driver, self.clock, 1.0)
        feed(self.imu, self.driver, self.clock, 1.0, rate=(-45.0, 0.0, 0.0))
        feed(self.imu, self.driver, self.clock, 1.0)
        self.assertAlmostEqual(self.imu.heading_now(), 45.0, delta=0.5)
        self.assertLess(self.imu.tilt_degrees(), 1.0)

    def test_the_heading_is_carried_on_to_now_between_samples(self):
        feed(self.imu, self.driver, self.clock, 1.0)
        feed(self.imu, self.driver, self.clock, 0.5, rate=(0.0, 0.0, 100.0))
        before = self.imu.heading_now()
        self.clock.now += 0.02
        self.assertAlmostEqual(self.imu.heading_now() - before, 2.0, delta=0.1)

    def test_lost_samples_are_bridged_at_the_last_rate(self):
        feed(self.imu, self.driver, self.clock, 1.0)
        feed(self.imu, self.driver, self.clock, 0.5, rate=(0.0, 0.0, 60.0))
        self.clock.now += 0.3
        self.imu.ingest([], (0.0, 0.0, 1.0), self.clock.now, overflowed=True)
        feed(self.imu, self.driver, self.clock, 0.2, rate=(0.0, 0.0, 60.0))
        feed(self.imu, self.driver, self.clock, 1.0)
        self.assertAlmostEqual(self.imu.heading_now(), 60.0, delta=1.0)
        self.assertEqual(self.imu.status()["overflows"], 1)

    def test_tipped_over_until_set_down_again(self):
        feed(self.imu, self.driver, self.clock, 1.0)
        self.driver.accel = (0.0, math.sin(math.radians(60)), math.cos(math.radians(60)))
        feed(self.imu, self.driver, self.clock, 0.1)
        self.assertFalse(self.imu.upset())
        feed(self.imu, self.driver, self.clock, 0.5)
        self.assertTrue(self.imu.upset())
        # Lying on its side does not become the new level.
        feed(self.imu, self.driver, self.clock, 10.0)
        self.assertTrue(self.imu.upset())
        self.driver.accel = (0.0, 0.0, 1.0)
        feed(self.imu, self.driver, self.clock, 0.5)
        self.assertTrue(self.imu.upset())
        feed(self.imu, self.driver, self.clock, 1.0)
        self.assertFalse(self.imu.upset())

    def test_knocks(self):
        feed(self.imu, self.driver, self.clock, 1.0)
        start = self.clock.now
        self.assertEqual(self.imu.knock_since(start), 0.0)
        self.driver.accel = (-1.5, 0.0, 1.0)
        feed(self.imu, self.driver, self.clock, 0.02)
        self.driver.accel = (0.0, 0.0, 1.0)
        feed(self.imu, self.driver, self.clock, 0.5)
        self.assertGreaterEqual(self.imu.knock_since(start), 1.2)
        self.assertEqual(self.imu.knock_since(self.clock.now), 0.0)

    def test_stops_trusting_a_silent_gyro_and_picks_up_again(self):
        feed(self.imu, self.driver, self.clock, 1.0)
        self.assertTrue(self.imu.ready())
        self.driver.fail = True
        for _ in range(MAX_READ_ERRORS):
            self.clock.now += 0.01
            self.imu.step()
        self.assertFalse(self.imu.ready())
        self.assertIsNone(self.imu.heading_now())
        self.assertEqual(self.imu.status()["mode"], "unavailable")
        self.driver.fail = False
        self.clock.now += RECOVER_SECONDS
        feed(self.imu, self.driver, self.clock, 1.0)
        self.assertTrue(self.imu.ready())
        self.assertGreaterEqual(self.driver.configured, 1)

    def test_measures_its_sample_period_against_the_clock(self):
        # The chip runs 2% fast: 102 samples each second.
        feed(self.imu, self.driver, self.clock, 1.0)
        for _ in range(700):
            self.driver.samples += [(0.0, 0.0, 0.0)] * 2
            self.clock.now += 2 / 102
            self.imu.step()
        self.assertAlmostEqual(self.imu.status()["sample_hz"], 102.0, delta=0.3)

    def test_status(self):
        status = ImuController(None).status()
        self.assertEqual(status["mode"], "absent")
        self.assertFalse(status["enabled"])
        status = ImuController(None, fault="bus 2: no IMU").status()
        self.assertEqual((status["mode"], status["fault"]), ("unavailable", "bus 2: no IMU"))
        self.assertEqual(self.imu.status()["mode"], "settling")
        feed(self.imu, self.driver, self.clock, 1.0)
        status = self.imu.status()
        self.assertEqual(status["mode"], "hardware")
        self.assertEqual(status["model"], "MPU-6500")


class GyroRig:
    """A gyro on a simulated car: the real turn rate is `scale` times what the
    wheel model assumes, and it coasts on for `coast_seconds` after the
    motors are cut (they ramp down anyway). The heading follows the drive's
    motor outputs whenever the drive asks for it."""

    def __init__(self, scale: float, coast_seconds: float = 0.05) -> None:
        self.scale, self.coast_seconds = scale, coast_seconds
        self.drive: DriveController | None = None
        self.heading = 0.0
        self.true_rate = 0.0
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.upset_now = False
        self.knock = 0.0

    def fitted(self) -> bool:
        return True

    def ready(self) -> bool:
        return True

    def upset(self) -> bool:
        return self.upset_now

    def knock_since(self, since: float) -> float:
        return self.knock

    def rate_dps(self) -> float:
        with self.lock:
            return self.true_rate

    def speed_cm_s(self) -> float:
        return 0.0

    # This rig measures turns only, not moves.
    def forward_axis(self):
        return None

    def set_forward_axis(self, axis) -> None:
        pass

    def begin_move(self):
        return None

    def end_move(self, move_id):
        return None

    def heading_now(self) -> float:
        with self.lock:
            now = time.monotonic()
            elapsed, self.last = now - self.last, now
            output = self.drive._output if self.drive is not None else (0, 0, 0, 0)
            fl, rl, fr, rr = (DriveController._wheel_speed_fraction(value) for value in output)
            commanded = (-fl - rl + fr + rr) / 4 * TURN_DEGREES_PER_SECOND_AT_FULL_DUTY * self.scale
            # Inertia: the body follows the wheels with a short lag.
            blend = min(1.0, elapsed / self.coast_seconds) if self.coast_seconds else 1.0
            self.true_rate += (commanded - self.true_rate) * blend
            self.heading += self.true_rate * elapsed
            return self.heading


def build(gyro=None, max_duty: int = 4095):
    drive = DriveController(SimulatedDriver(), max_duty=max_duty, command_timeout=1, physical=True, imu=gyro)
    if gyro is not None:
        gyro.drive = drive
    pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
    range_sensor = RangeSensorController(None, physical=False)
    mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
    return mapping, drive, (mapping, drive, pan_tilt, range_sensor)


def close(parts):
    for part in parts:
        part.close()


class FakeCamera:
    def capture_snapshot(self) -> bytes:
        return b"jpeg"


class DriveGyroTests(unittest.TestCase):
    def test_the_odometry_heading_follows_the_gyro(self):
        gyro = GyroRig(scale=0.5)
        mapping, drive, parts = build(gyro)
        try:
            time.sleep(0.05)
            self.assertEqual(drive.heading_source(), "gyro")
            drive.set_command(0, 0, 1.0)
            time.sleep(0.5)
            drive.stop()
            drive.wait_until_stopped()
            time.sleep(0.1)
            # The odometry turned as far as the gyro did, not the wheel model.
            self.assertAlmostEqual(drive.odometry()[2], gyro.heading, delta=2.0)
            # A push while parked still counts.
            with gyro.lock:
                gyro.heading += 30.0
            time.sleep(0.05)
            self.assertAlmostEqual(drive.odometry()[2], gyro.heading, delta=0.5)
        finally:
            close(parts)

    def test_wheels_take_over_while_the_gyro_cannot_be_trusted(self):
        gyro = GyroRig(scale=0.5)
        mapping, drive, parts = build(gyro)
        try:
            time.sleep(0.05)
            with patch.object(gyro, "heading_now", return_value=None), patch.object(gyro, "ready", return_value=False):
                time.sleep(0.05)
                self.assertEqual(drive.heading_source(), "wheels")
                drive.set_command(0, 0, 1.0)
                time.sleep(0.3)
                drive.stop()
                drive.wait_until_stopped()
                wheels = drive.odometry()[2]
                self.assertGreater(wheels, 10.0)
            time.sleep(0.05)
            self.assertEqual(drive.heading_source(), "gyro")
            # Picked up again where it left off: no jump to the gyro's own count.
            self.assertAlmostEqual(drive.odometry()[2], wheels, delta=1.0)
        finally:
            close(parts)

    def test_no_driving_while_tipped_over(self):
        gyro = GyroRig(scale=1.0)
        mapping, drive, parts = build(gyro)
        try:
            drive.set_command(0.5, 0, 0)
            gyro.upset_now = True
            time.sleep(0.05)
            self.assertEqual(drive.status()["throttle"], 0)
            self.assertEqual(drive._output, (0.0, 0.0, 0.0, 0.0))
            with self.assertRaises(DriveInhibitedError):
                drive.set_command(0.5, 0, 0)
            drive.set_command(0, 0, 0)  # stopping is always allowed
            gyro.upset_now = False
            drive.set_command(0.5, 0, 0)
        finally:
            close(parts)


class GyroTurnTests(unittest.TestCase):
    def test_turns_land_on_target_although_the_wheel_model_is_wrong(self):
        # The car really turns at 60% of what the wheel model assumes.
        gyro = GyroRig(scale=0.6, coast_seconds=0.08)
        mapping, drive, parts = build(gyro)
        try:
            time.sleep(0.05)
            for index, target in enumerate((90.0, -45.0, 20.0, -120.0, 10.0)):
                before = gyro.heading
                self.assertTrue(mapping._turn_by(target))
                time.sleep(0.2)
                # The first turn teaches it how this car coasts.
                slack = 5.0 if index == 0 else GYRO_TURN_TOLERANCE_DEGREES + 0.5
                self.assertAlmostEqual(gyro.heading - before, target, delta=slack)
            # It learned how far the turns coast.
            self.assertGreater(mapping._coast.observations, 0)
        finally:
            close(parts)

    def test_camera_alignment_is_skipped_with_a_gyro(self):
        gyro = GyroRig(scale=1.0)
        mapping, drive, parts = build(gyro)
        try:
            time.sleep(0.05)
            with patch.object(mapping, "_capture_views", side_effect=AssertionError("camera used")):
                self.assertTrue(mapping._turn_and_align(30.0))
        finally:
            close(parts)

    def test_a_stuck_turn_tries_more_power_then_gives_up(self):
        gyro = GyroRig(scale=0.0)
        mapping, drive, parts = build(gyro)
        try:
            time.sleep(0.05)
            started = time.monotonic()
            self.assertTrue(mapping._turn_by(90.0))
            self.assertLess(time.monotonic() - started, 4.0)
            # It learned that this much power does not turn this car.
            self.assertGreaterEqual(mapping._turn_power.breakaway, 0.9)
        finally:
            close(parts)

    def test_turn_ratios_follow_what_measures_the_turns(self):
        gyro = GyroRig(scale=1.0)
        mapping, drive, parts = build(gyro)
        try:
            time.sleep(0.05)
            # A calibration made with the wheels says nothing about the gyro.
            mapping._calibration = Calibration({"left": {10.0: 0.4}, "right": {10.0: 0.4}}, turn_source="wheels")
            self.assertEqual(mapping._rotation_ratio(30.0), 1.0)
            self.assertEqual(mapping.status()["turns_measured_by"], "gyro")
            # Nor does one made against the camera with a gyro: the gyro is the
            # reference, and lidar fits refine its scale while mapping.
            mapping._calibration = Calibration({"left": {10.0: 1.2}, "right": {10.0: 1.2}}, turn_source="gyro")
            mapping._reset_localization()
            self.assertEqual(mapping._rotation_ratio(30.0), 1.0)
            # Without a gyro the wheels' calibration is used as before.
            mapping._calibration = Calibration({"left": {10.0: 0.4}, "right": {10.0: 0.5}}, turn_source="wheels")
            with patch.object(drive, "heading_source", return_value="wheels"):
                mapping._reset_localization()
                self.assertAlmostEqual(mapping._rotation_ratio(30.0), 0.4 * 0.45 / 0.45 * (0.45 / 0.45), delta=0.06)
        finally:
            close(parts)


class CoastTests(unittest.TestCase):
    def test_learns_how_far_turns_coast_and_keeps_it(self):
        from robocar.drive import CoastModel

        model = CoastModel()
        # This car coasts 0.0004 deg per (deg/s)^2 plus 0.05 s worth of turning.
        for rate in (300, 120, 250, 60, 200, 90, 280, 150) * 3:
            model.learn(rate, 0.0004 * rate * rate + 0.05 * rate)
        self.assertAlmostEqual(model.predict(220), 0.0004 * 220 * 220 + 0.05 * 220, delta=0.5)
        again = CoastModel.from_json(json.loads(json.dumps(model.to_json())))
        self.assertAlmostEqual(again.predict(220), model.predict(220), places=6)
        with self.assertRaises(ValueError):
            CoastModel.from_json({"a": -1, "b": 0, "p": [[1, 0], [0, 1]]})

    def test_the_mapping_controller_saves_and_reloads_it(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "mapping.sqlite3"
            gyro = GyroRig(scale=0.8)
            drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True, imu=gyro)
            gyro.drive = drive
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
            range_sensor = RangeSensorController(None, physical=False)
            mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(database))
            try:
                time.sleep(0.05)
                while mapping._coast.observations < 5:
                    mapping._turn_by(40.0)
                learned = mapping._coast.predict(200)
                self.assertTrue((Path(folder) / "turning.json").exists())
                mapping.close()
                again = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(database))
                self.assertAlmostEqual(again._coast.predict(200), learned, delta=0.01)
                again.close()
            finally:
                drive.close()
                pan_tilt.close()
                range_sensor.close()


class BumpTests(unittest.TestCase):
    def test_a_move_ends_early_when_the_car_bumps_into_something(self):
        gyro = GyroRig(scale=1.0)
        mapping, drive, parts = build(gyro)
        try:
            time.sleep(0.05)
            gyro.knock = BUMP_STOP_G + 0.3
            started = time.monotonic()
            self.assertTrue(mapping._move_burst(0.5, 2.0))
            self.assertLess(time.monotonic() - started, 0.6)
            self.assertEqual(mapping.status()["last_bump"]["g"], round(BUMP_STOP_G + 0.3, 2))
        finally:
            close(parts)


class FusionTests(unittest.TestCase):
    def test_a_gyro_turn_is_trusted_over_a_disagreeing_camera(self):
        from robocar.localization import VisualMotion

        camera = VisualMotion(30.0, None, 3, 1.0)
        wheels = predict_motion((0.0, 0.0, 12.0), camera)
        gyro = predict_motion((0.0, 0.0, 12.0), camera, gyro=True)
        # The wheels are overruled (they slipped); the gyro is not.
        self.assertAlmostEqual(wheels.delta[2], 30.0, delta=0.5)
        self.assertAlmostEqual(gyro.delta[2], 12.0, delta=0.5)
        self.assertTrue(gyro.visual_rejected)
        self.assertLess(gyro.covariance[2, 2], wheels.covariance[2, 2])
        # A camera that agrees still refines it a little.
        agreeing = predict_motion((0.0, 0.0, 90.0), VisualMotion(91.0, None, 3, 0.5), gyro=True)
        self.assertTrue(agreeing.visual_used)
        self.assertAlmostEqual(agreeing.delta[2], 90.3, delta=0.5)


class CalibrationSourceTests(unittest.TestCase):
    def test_the_turn_source_is_kept_and_read_back(self):
        calibration = Calibration({"left": {10.0: 1.01}}, turn_source="gyro")
        self.assertEqual(Calibration.from_json(calibration.to_json()).turn_source, "gyro")
        self.assertEqual(Calibration.from_json({"turn_ratios": {}}).turn_source, "wheels")

    def test_no_turn_tests_with_a_gyro(self):
        ratios_used = []

        class Rig:
            heading = 0.0

            def heading_odometry(self):
                return self.heading

            def distance_odometry(self):
                return 0.0

            def turn(self, degrees, rotation_scale):
                ratios_used.append(rotation_scale)
                self.heading += degrees / rotation_scale
                return True

            def burst(self, throttle, seconds):
                return True

            def photograph(self, pan_offsets):
                return {offset: (b"jpeg",) for offset in pan_offsets}

            def measure_turn(self, strip, current):
                return None

            def range_ahead(self):
                return None

            def clearance_all_round(self):
                return 100.0

            def stopped(self):
                return len(ratios_used) >= 1

            def report(self, message):
                pass

        previous = Calibration({"left": {10.0: 0.3}, "right": {10.0: 0.3}}, turn_source="wheels")
        result = run_calibration(Rig(), previous, lambda cm: 0.1, turn_source="gyro")
        self.assertEqual(result.turn_source, "gyro")
        self.assertEqual(ratios_used, [])
        self.assertEqual(result.turn_ratios, {})
        self.assertTrue(any("gyro" in line for line in result.log))
        # With the wheels, earlier wheel ratios are the starting guess.
        run_calibration(Rig(), previous, lambda cm: 0.1, turn_source="wheels")
        self.assertEqual(ratios_used, [0.3])


class WebTests(unittest.TestCase):
    def test_status_reports_no_gyro_outside_hardware_mode(self):
        app = create_app(Config(mapping_database=Path(":memory:")))
        try:
            status = app.test_client().get("/api/status").get_json()
            self.assertIsNone(status["imu"])
            self.assertEqual(status["drive"]["heading_source"], "wheels")
            self.assertFalse(status["drive"]["upset"])
            self.assertIn("turns_measured_by", status["mapping"])
            self.assertIn('id="imu-status"', app.test_client().get("/").get_data(as_text=True))
        finally:
            for name in ("robocar_mapping", "robocar_drive", "robocar_pan_tilt", "robocar_range_sensor", "robocar_camera", "robocar_power"):
                app.extensions[name].close()

    def test_config_reads_the_imu_settings(self):
        with patch.dict("os.environ", {"ROBOCAR_IMU": "on", "ROBOCAR_IMU_BUS": "2"}):
            config = Config.from_env()
        self.assertEqual((config.imu, config.imu_bus), ("on", 2))
        with patch.dict("os.environ", {"ROBOCAR_IMU": "sometimes"}), self.assertRaises(ValueError):
            Config.from_env()


class PanScaleTests(unittest.TestCase):
    def tearDown(self):
        set_pan_scale(1.0)

    def test_pan_and_bearing_follow_the_scale(self):
        set_pan_scale(0.85)
        self.assertAlmostEqual(pan_bearing(30), 51.0)
        self.assertAlmostEqual(pan_bearing(150), -51.0)
        for bearing in (-40.0, 0.0, 25.5):
            self.assertAlmostEqual(pan_bearing(bearing_pan(bearing)), bearing)
        self.assertEqual(set_pan_scale(5.0), 1.4)

    @staticmethod
    def sweep(room, turn_deg, scale):
        """A lidar sweep of a room of ranges (by true bearing) by a head whose
        servo turns `scale` of what it is told, the car turned `turn_deg`."""
        return [
            {"pan": pan, "angle_deg": 90 - pan, "sensor": "lidar", "distance_cm": room((90 - pan) * scale + turn_deg)}
            for pan in range(30, 151)
        ]

    def test_learns_the_servo_scale_from_a_turn_on_the_spot(self):
        room = lambda bearing: 150 + 60 * math.sin(math.radians(3 * bearing)) + 25 * math.cos(math.radians(7 * bearing))
        for scale, turn in ((0.85, 40.0), (0.85, -60.0), (1.0, 30.0), (0.9, 85.0)):
            estimate = estimate_pan_scale(self.sweep(room, 0.0, scale), self.sweep(room, turn, scale), turn)
            self.assertIsNotNone(estimate)
            self.assertAlmostEqual(estimate[0], scale, delta=0.01)
        # Sweeps that do not overlap teach nothing.
        self.assertIsNone(estimate_pan_scale(self.sweep(room, 0.0, 0.85), self.sweep(room, 150.0, 0.85), 150.0))

    def test_stored_sweeps_take_the_current_scale(self):
        old = [{"angle_deg": 60, "distance_cm": 100.0}, {"angle_deg": -3, "distance_cm": 90.0}]
        new = [{"angle_deg": 51.0, "pan": 30, "distance_cm": 100.0}]
        set_pan_scale(0.8)
        self.assertEqual([item["angle_deg"] for item in with_current_angles(old)], [48.0, -2.4])
        self.assertEqual(with_current_angles(new)[0]["angle_deg"], 48.0)
        self.assertEqual(with_current_angles(old)[0]["pan"], 30)

    def test_mapping_refines_the_scale_after_a_gyro_turn_on_the_spot(self):
        room = lambda bearing: 150 + 60 * math.sin(math.radians(3 * bearing)) + 25 * math.cos(math.radians(7 * bearing))
        with tempfile.TemporaryDirectory() as folder:
            gyro = GyroRig(scale=1.0)
            drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True, imu=gyro)
            gyro.drive = drive
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
            pan_tilt._physical = True
            range_sensor = RangeSensorController(None, physical=False)
            store = MappingStore(Path(folder) / "mapping.sqlite3")
            mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, store)
            try:
                time.sleep(0.05)
                with patch.object(pan_tilt, "status", return_value={"mode": "hardware", "pan": 90, "tilt": 90}):
                    mapping._last_scan = self.sweep(room, 0.0, 0.85)
                    mapping._learn_pan_scale(self.sweep(room, 45.0, 0.85), (0.0, 0.0, 45.0), "gyro")
                    self.assertAlmostEqual(pan_scale(), 1.0 + 0.3 * (0.85 - 1.0), delta=0.01)
                    # Not from the wheels' idea of a turn, nor after driving.
                    before = pan_scale()
                    mapping._learn_pan_scale(self.sweep(room, 45.0, 0.85), (0.0, 0.0, 45.0), "wheels")
                    mapping._learn_pan_scale(self.sweep(room, 45.0, 0.85), (0.0, 20.0, 45.0), "gyro")
                    self.assertEqual(pan_scale(), before)
                    self.assertEqual(json.loads((Path(folder) / "turning.json").read_text())["pan_scale"], before)
            finally:
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()


class StartUpTests(unittest.TestCase):
    def test_a_motor_board_that_stops_answering_at_start_is_a_fault_not_a_crash(self):
        class FlakyDriver(SimulatedDriver):
            def set_motor_model(self, *outputs):
                raise OSError(121, "Remote I/O error")

        with patch("robocar.web.FreenoveDriver", return_value=FlakyDriver()), patch(
            "robocar.web.FreenovePanTilt", side_effect=OSError(121, "Remote I/O error")
        ), patch("robocar.web.FreenoveRangeSensor", side_effect=OSError(121, "Remote I/O error")), patch(
            "robocar.web.BoardADC", side_effect=OSError(121, "Remote I/O error")
        ), patch("robocar.web._open_imu", return_value=None), patch("robocar.web._open_lidar", return_value=None), patch(
            "robocar.web.connected_models", return_value=[]
        ):
            app = create_app(Config(mapping_database=Path(":memory:"), hardware_enabled=True, freenove_server_path=Path("/tmp")))
        try:
            status = app.test_client().get("/api/status").get_json()
            self.assertEqual(status["drive"]["mode"], "unavailable")
            self.assertIn("Remote I/O", status["drive"]["fault"])
        finally:
            for name in ("robocar_mapping", "robocar_drive", "robocar_pan_tilt", "robocar_range_sensor", "robocar_camera", "robocar_power"):
                app.extensions[name].close()

    def test_the_battery_monitor_is_looked_for_again(self):
        from robocar.power import PowerMonitor

        class ADC:
            def battery_volts(self):
                return 7.9

            def close(self):
                pass

        attempts = []

        def opener():
            attempts.append(1)
            if len(attempts) < 2:
                raise OSError(121, "Remote I/O error")
            return ADC()

        monitor = PowerMonitor(None, None, lambda: False, adc_fault="OSError", interval=3600, adc_opener=opener)
        try:
            with patch("robocar.power.ADC_RETRY_SECONDS", 0.0):
                monitor._adc_retry_at = 0.0
                monitor.sample()
                self.assertFalse(monitor.status()["battery"]["available"])
                monitor._adc_retry_at = 0.0
                monitor.sample()
            self.assertTrue(monitor.status()["battery"]["available"])
            self.assertAlmostEqual(monitor.status()["battery"]["volts"], 7.9)
        finally:
            monitor.close()


class TurnPowerTests(unittest.TestCase):
    """Turning at a wanted rate: the power found from what the gyro sees."""

    @staticmethod
    def run_turn(power, wanted, breakaway, rate_per_command, seconds=3.0):
        """A car that turns at rate_per_command x (power - breakaway), nothing
        below it; returns the rate it settled at."""
        steerer = TurnSteerer(power)
        heading, now, rates = 0.0, 0.0, []
        for _ in range(int(seconds / 0.1)):
            command = abs(steerer.step(wanted, heading, now))
            rate = max(0.0, command - breakaway) * rate_per_command
            heading += rate * 0.1
            now += 0.1
            rates.append(rate)
        return sum(rates[-5:]) / 5, steerer

    def test_finds_the_power_for_the_rate_wanted(self):
        power = TurnPower()
        # The car needs 45% to start turning, then 600 degrees a second per unit.
        rate, _ = self.run_turn(power, 60.0, 0.45, 600.0)
        self.assertAlmostEqual(rate, 60.0, delta=12.0)
        self.assertGreater(power.breakaway, 0.4)
        # The next turn starts close to right at once.
        self.assertAlmostEqual(power.command_for(60.0), 0.45 + 60.0 / 600.0, delta=0.08)

    def test_follows_a_change_of_floor(self):
        power = TurnPower()
        self.run_turn(power, 90.0, 0.55, 500.0)  # carpet
        rate, _ = self.run_turn(power, 90.0, 0.3, 500.0)  # then a smooth floor
        self.assertAlmostEqual(rate, 90.0, delta=18.0)

    def test_owns_up_when_it_cannot_turn(self):
        power = TurnPower()
        _, steerer = self.run_turn(power, 60.0, 5.0, 500.0, seconds=6.0)
        self.assertTrue(steerer.stalled_at_max)

    def test_kept_and_read_back(self):
        power = TurnPower(0.5, 650.0)
        again = TurnPower.from_json(json.loads(json.dumps(power.to_json())))
        self.assertEqual((again.breakaway, again.rate_per_command), (0.5, 650.0))
        with self.assertRaises(ValueError):
            TurnPower.from_json({"breakaway": 3.0, "rate_per_command": 500.0})


if __name__ == "__main__":
    unittest.main()
