"""The IMU: a gyro and accelerometer (MPU-6500 family) on its own I2C bus.

The gyro measures how far the car really turns. Without it turns are guessed
from the motor commands, which wheel slip, the battery and the floor all
upset. The chip samples the gyro at a steady rate into its own FIFO; a
thread drains it, so no sample is lost however busy the Pi is, and adds up
the turn rate about the up axis. Up is found from gravity, so the board may
be mounted any way round.

A gyro reads slightly off zero, and the offset (its bias) wanders with
temperature. Whenever the car sits still the bias is measured again and the
heading is held, so it does not creep while parked.

The accelerometer tells when the car is tipped over or picked up (the drive
then refuses to run the motors) and when it is knocked (a bump ends a move).
It also measures how far each move really went: its samples come through
the FIFO with the gyro's, evenly spaced; gravity is followed with the gyro
(so the car pitching as it speeds up is not taken for speeding up), and what
is left is added up twice. Every move starts and ends at rest, so whatever
speed the sum still shows at the end is drift, and is taken out. The board's
forward direction is learned from the first moves forward.

Boards sold as MPU-9250 often carry an MPU-6500, which has no compass; either
works, as the compass is not used (the motors would upset it anyway).
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)

MPU_ADDRESSES = (0x68, 0x69)
# Where to look when no bus is configured: the Pi 5's extra buses first (2 is
# GPIO12/13 with `dtoverlay=i2c2-pi5,pins_12_13`), the motor board's last.
IMU_SEARCH_BUSES = (2, 3, 0, 1)
MPU_MODELS = {0x70: "MPU-6500", 0x71: "MPU-9250", 0x73: "MPU-9255", 0x74: "MPU-6515", 0x68: "MPU-6050"}
REG_SMPLRT_DIV = 0x19
REG_CONFIG = 0x1A
REG_GYRO_CONFIG = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_ACCEL_CONFIG2 = 0x1D
REG_FIFO_EN = 0x23
REG_INT_STATUS = 0x3A
REG_ACCEL_XOUT_H = 0x3B
REG_USER_CTRL = 0x6A
REG_PWR_MGMT_1 = 0x6B
REG_PWR_MGMT_2 = 0x6C
REG_FIFO_COUNT_H = 0x72
REG_FIFO_R_W = 0x74
REG_WHO_AM_I = 0x75
PWR_RESET = 0x80
PWR_CLOCK_PLL = 0x01
FIFO_EN_ACCEL_GYRO = 0x78
USER_CTRL_FIFO_ENABLE = 0x40
USER_CTRL_FIFO_RESET = 0x04
INT_FIFO_OVERFLOW = 0x10
# 1 kHz internally behind a 41 Hz low-pass, divided down to 100 samples a
# second: accelerometer and gyro, 12 bytes each, so the 512-byte FIFO holds
# 0.42 s (it is read every 10 ms).
IMU_SAMPLE_HZ = 100
CONFIG_DLPF_41HZ = 0x03
SAMPLE_DIVIDER = 1000 // IMU_SAMPLE_HZ - 1
# The low-pass delays what the gyro reports by about this much.
FILTER_DELAY_SECONDS = 0.006
# +-2000 degrees a second (the car spins at several hundred), +-4 g.
GYRO_CONFIG_2000DPS = 0x18
GYRO_LSB_PER_DPS = 16.4
ACCEL_CONFIG_4G = 0x08
ACCEL_LSB_PER_G = 8192.0
GYRO_SATURATED = 32760
FIFO_SAMPLE_BYTES = 12
FIFO_SIZE_BYTES = {"MPU-6050": 1024}
FIFO_DEFAULT_SIZE_BYTES = 512
# SMBus block reads are limited to 32 bytes: two samples at a time.
FIFO_READ_SAMPLES = 2
POLL_SECONDS = 0.01
# Extrapolate the heading this far at most past the newest sample.
MAX_EXTRAPOLATE_SECONDS = 0.05
# The sample period is measured against the Pi's clock once this much data
# has arrived, and kept within this fraction of nominal.
PERIOD_MEASURE_AFTER_SECONDS = 5.0
PERIOD_MAX_ERROR = 0.1
# Still: over half a second every gyro axis varies by less than this, and
# so does the accelerometer.
STILL_WINDOW_SAMPLES = 50
STILL_CHECK_EVERY = 10
STILL_GYRO_STD_DPS = 0.3
STILL_GYRO_SPAN_DPS = 1.5
# While still, one sample this far from the bias on any axis ends it at
# once, so the start of a turn is never held.
STILL_BREAK_DPS = 1.0
# A smooth steady turn varies no more than a parked car, so still also means
# the average sits this close to the zero measured before. Should the zero
# itself have moved further (the chip warming), quiet this long settles it.
STILL_BIAS_SHIFT_DPS = 0.5
STILL_REACQUIRE_CHECKS = 30
STILL_ACCEL_STD_G = 0.02
# Likewise one accelerometer reading this far from its recent average (a
# smooth slide sideways turns nothing for the gyro to see).
STILL_BREAK_G = 0.04
BIAS_BLEND = 0.2
UP_BLEND = 0.1
# The level reference only follows gravity while it stays this close, so a
# car left lying on its side never comes to count as upright.
LEVEL_FOLLOW_DEGREES = 10.0
# Tipped or lifted: tilted past this for a moment; upright again once back
# under the lower angle for a second.
UPSET_TILT_DEGREES = 35.0
UPSET_SECONDS = 0.3
UPRIGHT_TILT_DEGREES = 20.0
UPRIGHT_SECONDS = 1.0
# A knock: the acceleration jumps this far from its recent average. Starting
# and stopping at full power is about 0.5 g.
KNOCK_G = 1.2
KNOCK_AVERAGE_BLEND = 0.05
ACCEL_FILTER_BLEND = 0.2
# Consecutive failed reads before the IMU counts as lost (and the wheels
# take over the heading); while lost it is set up again this often.
MAX_READ_ERRORS = 20
RECOVER_SECONDS = 2.0
# Moves. Gravity is followed by the gyro, and drawn this much a sample toward
# what the accelerometer says while that is within the window of 1 g (far
# slower than a move: a move's own acceleration barely moves it).
GRAVITY_CM_S2 = 980.665
ATTITUDE_BLEND = 0.002
ATTITUDE_WINDOW_G = 0.05
# The forward direction is learned from moves at least this long, blended in
# this much at a time.
FORWARD_LEARN_MIN_CM = 5.0
FORWARD_BLEND = 0.3
# Whether the car is moving at all, all the time: the level acceleration
# added up into a speed that leaks away over about this long (so drift and
# shaking fade), and is zero whenever the car is still.
SPEED_LEAK_SECONDS = 0.7

Vector = tuple[float, float, float]


@dataclass(frozen=True)
class MoveMeasure:
    """How far a move went by the accelerometer. `ahead_cm` and `right_cm`
    are in the car's frame at the start of the move (None until the board's
    forward direction is known); `plane` is the same in the board's own
    level axes. `end_speed_cm_s` is the drift the at-rest end took out: a
    large one means the sum is not to be trusted."""

    ahead_cm: float | None
    right_cm: float | None
    plane: tuple[float, float]
    seconds: float
    end_speed_cm_s: float
    peak_speed_cm_s: float
    lost_samples: bool
    axes: tuple[Vector, Vector] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    def summary(self) -> dict[str, object]:
        return {
            "ahead_cm": None if self.ahead_cm is None else round(self.ahead_cm, 1),
            "right_cm": None if self.right_cm is None else round(self.right_cm, 1),
            "seconds": round(self.seconds, 2),
            "end_speed_cm_s": round(self.end_speed_cm_s, 1),
            "peak_speed_cm_s": round(self.peak_speed_cm_s, 1),
            "lost_samples": self.lost_samples,
        }


class ImuDriver(Protocol):
    model: str
    sample_hz: float

    def reset_fifo(self) -> None: ...

    def read_samples(self) -> tuple[list[tuple[Vector, Vector]], bool]: ...

    def configure(self) -> None: ...

    def details(self) -> dict[str, object]: ...

    def close(self) -> None: ...


class MPU6500:
    """The chip itself: set up, FIFO of gyro samples, accelerometer reads."""

    def __init__(self, bus: int, address: int | None = None, smbus=None) -> None:
        if smbus is None:
            try:
                from smbus2 import SMBus
            except ImportError:
                from smbus import SMBus  # Raspberry Pi OS's python3-smbus
            smbus = SMBus(bus)
        self._bus = smbus
        self.bus_number = bus
        self.sample_hz = float(IMU_SAMPLE_HZ)
        self._lock = threading.Lock()
        found = None
        for candidate in (address,) if address is not None else MPU_ADDRESSES:
            try:
                identity = self._bus.read_byte_data(candidate, REG_WHO_AM_I)
            except OSError:
                continue
            if identity in MPU_MODELS:
                found = (candidate, identity)
                break
        if found is None:
            self._bus.close()
            where = f"0x{address:02x}" if address is not None else "0x68 or 0x69"
            raise RuntimeError(f"No MPU-6500 family IMU answered at {where} on I2C bus {bus}")
        self.address, self.identity = found
        self.model = MPU_MODELS[self.identity]
        self._fifo_limit = FIFO_SIZE_BYTES.get(self.model, FIFO_DEFAULT_SIZE_BYTES) - FIFO_SAMPLE_BYTES
        self.configure()

    def _write(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self.address, register, value)

    def configure(self) -> None:
        with self._lock:
            self._write(REG_PWR_MGMT_1, PWR_RESET)
            time.sleep(0.1)
            self._write(REG_PWR_MGMT_1, PWR_CLOCK_PLL)
            time.sleep(0.05)
            self._write(REG_PWR_MGMT_2, 0x00)
            self._write(REG_CONFIG, CONFIG_DLPF_41HZ)
            self._write(REG_SMPLRT_DIV, SAMPLE_DIVIDER)
            self._write(REG_GYRO_CONFIG, GYRO_CONFIG_2000DPS)
            self._write(REG_ACCEL_CONFIG, ACCEL_CONFIG_4G)
            if self.model != "MPU-6050":
                self._write(REG_ACCEL_CONFIG2, CONFIG_DLPF_41HZ)
            # The gyro needs a moment to start before its samples are good.
            time.sleep(0.05)
        self.reset_fifo()

    def reset_fifo(self) -> None:
        with self._lock:
            self._write(REG_USER_CTRL, 0x00)
            self._write(REG_FIFO_EN, 0x00)
            self._write(REG_USER_CTRL, USER_CTRL_FIFO_RESET)
            self._write(REG_FIFO_EN, FIFO_EN_ACCEL_GYRO)
            self._write(REG_USER_CTRL, USER_CTRL_FIFO_ENABLE)
            self._bus.read_byte_data(self.address, REG_INT_STATUS)  # clears the overflow flag

    def read_samples(self) -> tuple[list[tuple[Vector, Vector]], bool]:
        """(gyro in degrees a second, accelerometer in g) samples waiting in
        the FIFO, oldest first, and whether it overflowed (samples lost; it
        is then emptied)."""
        with self._lock:
            status = self._bus.read_byte_data(self.address, REG_INT_STATUS)
            high, low = self._bus.read_i2c_block_data(self.address, REG_FIFO_COUNT_H, 2)
            count = ((high & 0x1F) << 8) | low
            if status & INT_FIFO_OVERFLOW or count > self._fifo_limit:
                overflowed = True
            else:
                overflowed = False
                samples: list[tuple[Vector, Vector]] = []
                waiting = count // FIFO_SAMPLE_BYTES
                while waiting:
                    batch = min(waiting, FIFO_READ_SAMPLES)
                    data = self._bus.read_i2c_block_data(self.address, REG_FIFO_R_W, batch * FIFO_SAMPLE_BYTES)
                    for start in range(0, batch * FIFO_SAMPLE_BYTES, FIFO_SAMPLE_BYTES):
                        # The FIFO holds the accelerometer's axes, then the gyro's.
                        values = [_signed(data[start + 2 * index], data[start + 2 * index + 1]) for index in range(6)]
                        accel = tuple(value / ACCEL_LSB_PER_G for value in values[:3])
                        gyro = tuple(value / GYRO_LSB_PER_DPS for value in values[3:])
                        samples.append((gyro, accel))
                    waiting -= batch
                return samples, False
        self.reset_fifo()
        return [], overflowed

    def details(self) -> dict[str, object]:
        return {"model": self.model, "bus": self.bus_number, "address": f"0x{self.address:02x}"}

    def close(self) -> None:
        try:
            with self._lock:
                self._write(REG_USER_CTRL, 0x00)
                self._write(REG_PWR_MGMT_1, 0x40)  # sleep
        except OSError:
            pass
        finally:
            self._bus.close()


def _signed(high: int, low: int) -> int:
    value = (high << 8) | low
    return value - 0x10000 if value & 0x8000 else value


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float]) -> Vector | None:
    length = _norm(a)
    return (a[0] / length, a[1] / length, a[2] / length) if length > 1e-6 else None


def _blend(old: Sequence[float], new: Sequence[float], weight: float) -> Vector:
    return tuple(o + weight * (n - o) for o, n in zip(old, new))


def _cross(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _level_axes(up: Vector) -> tuple[Vector, Vector]:
    """Two level axes, at right angles, turning counterclockwise about up."""
    helper = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)
    along = _dot(helper, up)
    first = _unit(tuple(h - along * u for h, u in zip(helper, up))) or (1.0, 0.0, 0.0)
    return first, _cross(up, first)


def _angle_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(ua, ub)))))


class ImuController:
    """Keeps the heading from the gyro, and the tilt and knocks from the
    accelerometer, on a thread of its own (or step by step in tests)."""

    def __init__(self, driver: ImuDriver | None, *, fault: str | None = None, start: bool = True, clock=time.monotonic) -> None:
        self._driver = driver
        self._fault = fault if driver is not None or fault else "not fitted"
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = threading.Event()
        nominal = 1.0 / driver.sample_hz if driver is not None else 0.01
        self._nominal_period = nominal
        self._period = nominal
        self._heading = 0.0
        self._sample_time: float | None = None
        self._rate = 0.0
        self._rates: deque[float] = deque(maxlen=3)
        self._bias: Vector = (0.0, 0.0, 0.0)
        self._bias_known = False
        self._up: Vector | None = None
        self._level: Vector | None = None
        self._still = False
        self._window: deque[Vector] = deque(maxlen=STILL_WINDOW_SAMPLES)
        self._accel_window: deque[Vector] = deque(maxlen=STILL_WINDOW_SAMPLES)
        self._since_check = 0
        self._quiet_checks = 0
        self._accel: Vector | None = None
        self._accel_average: Vector | None = None
        self._tilt = 0.0
        self._upset = False
        self._tilted_since: float | None = None
        self._upright_since: float | None = None
        self._knocks: deque[tuple[float, float]] = deque(maxlen=64)
        self._counted = 0
        self._counted_since: float | None = None
        self._overflows = 0
        self._saturated = 0
        self._errors = 0
        self._read_errors = 0
        self._lost_since: float | None = None
        self._last_error: str | None = None
        # Gravity as the board sees it (followed by the gyro), its size when
        # still, the move being measured, and the board's forward direction.
        self._gravity: Vector | None = None
        self._gravity_size = 1.0
        self._move: dict[str, object] | None = None
        self._move_id = 0
        self._speed: Vector = (0.0, 0.0, 0.0)
        self._forward: Vector | None = None
        self._thread: threading.Thread | None = None
        if driver is not None and start:
            self._thread = threading.Thread(target=self._loop, name="imu", daemon=True)
            self._thread.start()

    # -- reading ---------------------------------------------------------

    def _loop(self) -> None:
        while not self._closed.wait(POLL_SECONDS):
            self.step()

    def step(self) -> None:
        """Drain the gyro samples waiting and read the accelerometer once."""
        driver = self._driver
        if driver is None:
            return
        now = self._clock()
        with self._lock:
            lost_since = self._lost_since
        if lost_since is not None:
            if now - lost_since < RECOVER_SECONDS:
                return
            try:
                driver.configure()
            except Exception as exc:
                self._note_error(exc, now)
                with self._lock:
                    self._lost_since = now
                return
            logger.info("IMU answering again")
            with self._lock:
                self._lost_since = None
                self._read_errors = 0
                self._restart_timing()
        try:
            pairs, overflowed = driver.read_samples()
        except Exception as exc:
            self._note_error(exc, now)
            return
        self.ingest([gyro for gyro, _ in pairs], None, now, overflowed, accels=[accel for _, accel in pairs])

    def _note_error(self, exc: Exception, now: float) -> None:
        with self._lock:
            self._errors += 1
            self._read_errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            if self._read_errors >= MAX_READ_ERRORS and self._lost_since is None:
                logger.warning("IMU stopped answering (%s); turns go by the wheels until it is back", self._last_error)
                self._lost_since = now
                self._restart_timing()

    def _restart_timing(self) -> None:
        self._sample_time = None
        self._counted = 0
        self._counted_since = None

    def ingest(
        self,
        samples: Sequence[Vector],
        accel: Vector | None,
        now: float,
        overflowed: bool = False,
        accels: Sequence[Vector] | None = None,
    ) -> None:
        """Add gyro samples (oldest first, degrees a second, the newest taken
        about now) with the accelerometer's (in g): one for each sample in
        `accels`, or a single reading `accel` for them all."""
        with self._lock:
            self._read_errors = 0
            if overflowed:
                self._overflows += 1
                if self._move is not None:
                    self._move["lost"] = True
                # Samples were lost: bridge the gap at the last known rate.
                if self._sample_time is not None and not self._still:
                    self._heading += self._rate * min(now - self._sample_time, 1.0)
                self._restart_timing()
                self._sample_time = now
            per_sample = accels is not None and len(accels) == len(samples)
            if accel is not None and not per_sample:
                self._add_accel(accel, now)
            if not samples:
                return
            if self._counted_since is None:
                # The first batch only starts the clock for measuring the period.
                self._counted_since = now
            else:
                self._counted += len(samples)
                elapsed = now - self._counted_since
                if elapsed >= PERIOD_MEASURE_AFTER_SECONDS and self._counted:
                    low, high = (1 - PERIOD_MAX_ERROR) * self._nominal_period, (1 + PERIOD_MAX_ERROR) * self._nominal_period
                    self._period = min(high, max(low, elapsed / self._counted))
            count = len(samples)
            for index, sample in enumerate(samples):
                sample_accel = accels[index] if per_sample else accel
                if per_sample:
                    self._add_accel(sample_accel, now - (count - 1 - index) * self._period)
                self._add_gyro(sample)
                if sample_accel is not None:
                    self._track_motion(sample, sample_accel)
            self._sample_time = now

    def _add_gyro(self, sample: Vector) -> None:
        if any(abs(value) * GYRO_LSB_PER_DPS >= GYRO_SATURATED for value in sample):
            self._saturated += 1
        corrected = tuple(value - bias for value, bias in zip(sample, self._bias))
        if self._still and any(abs(value) > STILL_BREAK_DPS for value in corrected):
            self._still = False
            self._since_check = 0
        self._window.append(sample)
        self._since_check += 1
        if self._since_check >= STILL_CHECK_EVERY and len(self._window) == STILL_WINDOW_SAMPLES:
            self._since_check = 0
            self._check_still()
        rate = _dot(corrected, self._up or (0.0, 0.0, 1.0))
        self._rates.append(rate)
        self._rate = sum(self._rates) / len(self._rates)
        if self._still or not self._bias_known:
            # Parked: hold the heading (whatever the gyro's zero does). And
            # nothing counts until the zero has first been measured.
            self._rate = 0.0
            return
        self._heading += rate * self._period

    def _track_motion(self, gyro: Vector, accel: Vector) -> None:
        """Follow gravity with the gyro, and add up the move being measured."""
        if self._gravity is None:
            self._gravity = self._up or _unit(accel)
            if self._gravity is None:
                return
        dt = self._period
        if self._bias_known:
            rate = tuple(math.radians(value - bias) for value, bias in zip(gyro, self._bias))
            # A direction fixed in the room turns the other way in the board's frame.
            turned = _cross(rate, self._gravity)
            self._gravity = _unit(tuple(g - t * dt for g, t in zip(self._gravity, turned))) or self._gravity
        size = _norm(accel)
        if self._still:
            self._gravity = _unit(_blend(self._gravity, accel, 0.2)) or self._gravity
            self._gravity_size += 0.05 * (size - self._gravity_size)
        elif abs(size - self._gravity_size) <= ATTITUDE_WINDOW_G:
            self._gravity = _unit(_blend(self._gravity, _unit(accel) or self._gravity, ATTITUDE_BLEND)) or self._gravity
        linear = tuple((a - self._gravity_size * g) * GRAVITY_CM_S2 for a, g in zip(accel, self._gravity))
        if self._still:
            self._speed = (0.0, 0.0, 0.0)
        else:
            up = _dot(linear, self._gravity)
            level = tuple(value - up * g for value, g in zip(linear, self._gravity))
            keep = math.exp(-dt / SPEED_LEAK_SECONDS)
            self._speed = tuple(v * keep + a * dt for v, a in zip(self._speed, level))
        move = self._move
        if move is None:
            return
        first, second = move["axes"]
        along, across = _dot(linear, first), _dot(linear, second)
        # Into the level frame the move started in (the car may have turned).
        turned = math.radians(self._heading - move["heading"])
        cos_t, sin_t = math.cos(turned), math.sin(turned)
        ax, ay = along * cos_t - across * sin_t, along * sin_t + across * cos_t
        vx, vy = move["velocity"]
        vx, vy = vx + ax * dt, vy + ay * dt
        px, py = move["position"]
        move["velocity"] = (vx, vy)
        move["position"] = (px + vx * dt, py + vy * dt)
        move["seconds"] += dt
        move["peak"] = max(move["peak"], math.hypot(vx, vy))

    def begin_move(self) -> int | None:
        """Start measuring a move (the car at rest now); None if the IMU is not ready."""
        with self._lock:
            if not self._ready_locked() or self._gravity is None:
                return None
            self._move_id += 1
            self._move = {
                "id": self._move_id,
                "axes": _level_axes(self._gravity),
                "heading": self._heading,
                "velocity": (0.0, 0.0),
                "position": (0.0, 0.0),
                "seconds": 0.0,
                "peak": 0.0,
                "lost": False,
            }
            return self._move_id

    def end_move(self, move_id: int | None) -> MoveMeasure | None:
        """Finish measuring (the car at rest again): how far it went."""
        with self._lock:
            move = self._move
            if move_id is None or move is None or move["id"] != move_id:
                return None
            self._move = None
            seconds = float(move["seconds"])
            vx, vy = move["velocity"]
            px, py = move["position"]
            # It ended at rest: speed still showing grew steadily from nothing
            # (a constant error in the acceleration), so take out its distance.
            px, py = px - vx * seconds / 2, py - vy * seconds / 2
            ahead = right = None
            if self._forward is not None:
                first, second = move["axes"]
                forward = _unit((_dot(self._forward, first), _dot(self._forward, second), 0.0))
                if forward is not None:
                    ahead = px * forward[0] + py * forward[1]
                    right = px * forward[1] - py * forward[0]
            return MoveMeasure(
                ahead, right, (px, py), seconds, math.hypot(vx, vy), float(move["peak"]), bool(move["lost"]), move["axes"]
            )

    def learn_forward(self, measure: MoveMeasure, ahead_share: float, right_share: float) -> bool:
        """Learn the board's forward direction from a move whose direction in
        the car's frame is known (forward, or sideways, or between)."""
        distance = math.hypot(*measure.plane)
        if distance < FORWARD_LEARN_MIN_CM or measure.lost_samples or math.hypot(ahead_share, right_share) < 1e-6:
            return False
        axes = measure.axes
        with self._lock:
            # The move went `commanded` clockwise of forward, so forward is
            # that much counterclockwise of the way it went.
            commanded = math.atan2(right_share, ahead_share)
            dx, dy = measure.plane[0] / distance, measure.plane[1] / distance
            fx = dx * math.cos(commanded) - dy * math.sin(commanded)
            fy = dx * math.sin(commanded) + dy * math.cos(commanded)
            first, second = axes
            seen = tuple(fx * a + fy * b for a, b in zip(first, second))
            self._forward = _unit(_blend(self._forward, seen, FORWARD_BLEND)) if self._forward else _unit(seen)
            return True

    def forward_axis(self) -> Vector | None:
        with self._lock:
            return self._forward

    def set_forward_axis(self, axis: Sequence[float] | None) -> None:
        with self._lock:
            self._forward = _unit(tuple(float(value) for value in axis)) if axis is not None else None

    def _check_still(self) -> None:
        gyro = list(self._window)
        still = True
        for axis in range(3):
            values = [sample[axis] for sample in gyro]
            mean = sum(values) / len(values)
            spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            if spread > STILL_GYRO_STD_DPS or max(values) - min(values) > STILL_GYRO_SPAN_DPS:
                still = False
                break
        accel = list(self._accel_window)
        if still and len(accel) >= STILL_WINDOW_SAMPLES // 5:
            for axis in range(3):
                values = [sample[axis] for sample in accel]
                mean = sum(values) / len(values)
                if math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) > STILL_ACCEL_STD_G:
                    still = False
                    break
        mean_gyro = tuple(sum(sample[axis] for sample in gyro) / len(gyro) for axis in range(3))
        self._quiet_checks = self._quiet_checks + 1 if still else 0
        if still and self._bias_known and self._quiet_checks < STILL_REACQUIRE_CHECKS:
            if any(abs(mean - bias) > STILL_BIAS_SHIFT_DPS for mean, bias in zip(mean_gyro, self._bias)):
                still = False
        self._still = still
        if not still:
            return
        self._bias = mean_gyro if not self._bias_known else _blend(self._bias, mean_gyro, BIAS_BLEND)
        self._bias_known = True
        if accel:
            gravity = _unit(tuple(sum(sample[axis] for sample in accel) / len(accel) for axis in range(3)))
            if gravity is not None:
                self._up = _unit(_blend(self._up, gravity, UP_BLEND)) if self._up else gravity
                if self._level is None:
                    self._level = gravity
                elif _angle_degrees(self._level, gravity) <= LEVEL_FOLLOW_DEGREES:
                    self._level = _unit(_blend(self._level, gravity, UP_BLEND)) or self._level

    def _add_accel(self, accel: Vector, now: float) -> None:
        if self._still and self._accel_average is not None:
            if _norm(tuple(a - m for a, m in zip(accel, self._accel_average))) > STILL_BREAK_G:
                self._still = False
                self._since_check = 0
        self._accel_window.append(accel)
        if self._up is None:
            # Until the car has sat still, up is wherever gravity points now.
            self._up = _unit(accel)
        if self._level is None:
            self._level = _unit(accel)
        if self._accel_average is not None:
            jump = _norm(tuple(a - m for a, m in zip(accel, self._accel_average)))
            if jump >= KNOCK_G:
                self._knocks.append((now, jump))
        self._accel_average = accel if self._accel_average is None else _blend(self._accel_average, accel, KNOCK_AVERAGE_BLEND)
        self._accel = accel if self._accel is None else _blend(self._accel, accel, ACCEL_FILTER_BLEND)
        self._tilt = _angle_degrees(self._accel, self._level) if self._level else 0.0
        if not self._upset:
            if self._tilt >= UPSET_TILT_DEGREES:
                self._tilted_since = self._tilted_since if self._tilted_since is not None else now
                if now - self._tilted_since >= UPSET_SECONDS:
                    self._upset = True
                    self._upright_since = None
                    logger.warning("The car is tipped over or lifted (tilted %.0f degrees)", self._tilt)
            else:
                self._tilted_since = None
        else:
            if self._tilt <= UPRIGHT_TILT_DEGREES:
                self._upright_since = self._upright_since if self._upright_since is not None else now
                if now - self._upright_since >= UPRIGHT_SECONDS:
                    self._upset = False
                    self._tilted_since = None
                    logger.info("The car is upright again")
            else:
                self._upright_since = None

    # -- what it knows ---------------------------------------------------

    def ready(self) -> bool:
        """Whether the heading can be trusted: reading, and the bias measured."""
        with self._lock:
            return self._ready_locked()

    def _ready_locked(self) -> bool:
        return self._driver is not None and self._lost_since is None and self._bias_known and self._sample_time is not None

    def fitted(self) -> bool:
        """Whether there is an IMU at all (it may still be settling)."""
        return self._driver is not None

    def heading_now(self) -> float | None:
        """Degrees turned counterclockwise since start, carried on to this
        moment at the latest rate; None when the gyro cannot be trusted."""
        now = self._clock()
        with self._lock:
            if not self._ready_locked():
                return None
            ahead = min(MAX_EXTRAPOLATE_SECONDS, max(0.0, now - self._sample_time + FILTER_DELAY_SECONDS))
            return self._heading + self._rate * ahead

    def rate_dps(self) -> float:
        with self._lock:
            return self._rate

    def still(self) -> bool:
        with self._lock:
            return self._still

    def speed_cm_s(self) -> float:
        """Roughly how fast the car is going now (leaky: for telling moving
        from not moving, not for distances)."""
        with self._lock:
            return 0.0 if self._still else _norm(self._speed)

    def upset(self) -> bool:
        """Tipped over or picked up (and not yet set down for a second)."""
        with self._lock:
            return self._upset and self._lost_since is None

    def tilt_degrees(self) -> float:
        with self._lock:
            return self._tilt

    def knock_since(self, since: float) -> float:
        """The hardest knock (in g) since that moment, 0 if none."""
        with self._lock:
            return max((size for at, size in self._knocks if at >= since), default=0.0)

    def status(self) -> dict[str, object]:
        details = self._driver.details() if self._driver is not None else {}
        with self._lock:
            ready = self._ready_locked()
            if self._driver is None:
                mode, fault = ("absent", None) if self._fault == "not fitted" else ("unavailable", self._fault)
            elif self._lost_since is not None:
                mode, fault = "unavailable", self._last_error
            else:
                mode, fault = ("hardware" if ready else "settling"), None
            return {
                "enabled": ready,
                "mode": mode,
                "fault": fault,
                **details,
                "heading_deg": round(self._heading, 2),
                "rate_dps": round(self._rate, 2),
                "bias_dps": [round(value, 3) for value in self._bias] if self._bias_known else None,
                "up": [round(value, 3) for value in self._up] if self._up else None,
                "still": self._still,
                "tilt_deg": round(self._tilt, 1),
                "upset": self._upset,
                "knocks": len(self._knocks),
                "sample_hz": round(1.0 / self._period, 2),
                "overflows": self._overflows,
                "saturated": self._saturated,
                "errors": self._errors,
                "forward_axis": [round(value, 3) for value in self._forward] if self._forward else None,
            }

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._driver is not None:
            self._driver.close()
