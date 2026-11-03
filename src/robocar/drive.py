import importlib
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from robocar.imu import ImuController, MoveMeasure

logger = logging.getLogger(__name__)
MOTOR_RAMP_SECONDS = 0.25
MOTOR_UPDATE_INTERVAL_SECONDS = 0.01
# Head position at start-up and for "Center head": pan straight ahead, tilt
# level, so the lidar on the head shoots flat across the room.
DEFAULT_PAN = 90
DEFAULT_TILT = 90
# Software travel, kept clear of the servos' end stops.
PAN_LIMITS = (30, 150)
TILT_LIMITS = (30, 150)


# The pan servo turns the head less than it is told: real degrees per
# commanded degree. Measured on the car (11-02) at about 0.85 by turning it on
# the spot by gyro-measured amounts and seeing how far the lidar's view moved
# across the pan; its "60 degrees either side" is really about 51. 1.0 until
# set: the web app sets the car's value, and mapping refines it from turns.
HARDWARE_PAN_SCALE = 0.85
PAN_SCALE_LIMITS = (0.6, 1.4)
_pan_scale = 1.0


def pan_scale() -> float:
    return _pan_scale


def set_pan_scale(scale: float) -> float:
    global _pan_scale
    _pan_scale = min(PAN_SCALE_LIMITS[1], max(PAN_SCALE_LIMITS[0], float(scale)))
    return _pan_scale


def pan_bearing(pan: float) -> float:
    """Direction a pan angle looks, counterclockwise from straight ahead.

    The Freenove pan servo turns the head clockwise as its angle rises: pan 30
    looks left and 150 looks right (checked on the car, and how Freenove's own
    avoidance code reads it)."""
    return (90.0 - pan) * _pan_scale


def bearing_pan(bearing: float) -> float:
    """The pan angle that looks in a direction (the inverse of pan_bearing)."""
    return 90.0 - bearing / _pan_scale
FULL_MOTOR_DUTY = 4095
# Wheel odometry calibration. The forward speed is the earlier estimate of
# 30.48 cm per 0.25 s at full duty; the deadband, strafe and turn rates are
# first guesses. Mapping refines its own scale from sonar registration and camera yaw,
# but calibrate these on the floor for a better starting estimate.
MOTOR_DEADBAND_FRACTION = 0.1
FORWARD_CM_PER_SECOND_AT_FULL_DUTY = 121.9
STRAFE_CM_PER_SECOND_AT_FULL_DUTY = 97.5
# Raised 8x from the first guess of 48 after closed-loop turns on the car
# overshot by roughly that factor (a planned turn came out near 270 degrees).
TURN_DEGREES_PER_SECOND_AT_FULL_DUTY = 384.0


class TurnPower:
    """What power turns the car on the spot how fast, learned from the gyro.

    Below some power (the breakaway) the car does not turn at all: the
    wheels just strain, or one spins. Above it the turn rate grows roughly in
    proportion: rate = gain x (power - breakaway). Turns ask for a rate; this
    says what power to start from, and learns from what the gyro then sees:
    every steady turn refines the line (least squares, slowly forgetting, so
    a change of floor or battery is followed), and a power that did not turn
    the car at all pushes the breakaway above it."""

    MIN_COMMAND = 0.2
    MAX_COMMAND = 1.0
    BREAKAWAY_LIMITS = (0.15, 0.9)
    RATE_LIMITS = (100.0, 2000.0)
    FORGET = 0.9

    def __init__(self, breakaway: float = 0.4, rate_per_command: float = 500.0) -> None:
        self.breakaway = breakaway
        self.rate_per_command = rate_per_command
        # rate = theta[0] * power + theta[1]
        self._p = [[400.0**2, 0.0], [0.0, 200.0**2]]
        self.observations = 0

    def command_for(self, rate_dps: float) -> float:
        """The power to turn at `rate_dps` (either way)."""
        command = self.breakaway + abs(rate_dps) / self.rate_per_command
        return min(self.MAX_COMMAND, max(self.MIN_COMMAND, command))

    def note_stall(self, command: float) -> None:
        """This power did not turn the car."""
        command = abs(command)
        if command + 0.05 > self.breakaway:
            self.breakaway = min(self.BREAKAWAY_LIMITS[1], command + 0.05)
        self.observations += 1

    def note_turning(self, command: float, rate_dps: float) -> None:
        """This power turned the car steadily at this rate."""
        command, rate = abs(command), abs(rate_dps)
        theta = (self.rate_per_command, -self.rate_per_command * self.breakaway)
        phi = (command, 1.0)
        p = self._p
        p_phi = (p[0][0] * phi[0] + p[0][1] * phi[1], p[1][0] * phi[0] + p[1][1] * phi[1])
        denominator = self.FORGET + phi[0] * p_phi[0] + phi[1] * p_phi[1]
        gain = (p_phi[0] / denominator, p_phi[1] / denominator)
        error = rate - (theta[0] * phi[0] + theta[1] * phi[1])
        slope = theta[0] + gain[0] * error
        offset = theta[1] + gain[1] * error
        self._p = [[(p[i][j] - gain[i] * p_phi[j]) / self.FORGET for j in range(2)] for i in range(2)]
        low, high = self.RATE_LIMITS
        slope = min(high, max(low, slope))
        breakaway = -offset / slope
        # It turned at this power, so it starts below it.
        breakaway = min(breakaway, command - 0.02)
        self.rate_per_command = slope
        self.breakaway = min(self.BREAKAWAY_LIMITS[1], max(self.BREAKAWAY_LIMITS[0], breakaway))
        self.observations += 1

    def summary(self) -> dict[str, object]:
        return {
            "breakaway": round(self.breakaway, 2),
            "for_60_dps": round(self.command_for(60.0), 2),
            "for_180_dps": round(self.command_for(180.0), 2),
            "turns": self.observations,
        }

    def to_json(self) -> dict[str, object]:
        return {"breakaway": self.breakaway, "rate_per_command": self.rate_per_command, "observations": self.observations}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "TurnPower":
        model = cls(float(data["breakaway"]), float(data["rate_per_command"]))
        if not (cls.BREAKAWAY_LIMITS[0] <= model.breakaway <= cls.BREAKAWAY_LIMITS[1]):
            raise ValueError("breakaway out of range")
        if not (cls.RATE_LIMITS[0] <= model.rate_per_command <= cls.RATE_LIMITS[1]):
            raise ValueError("turn rate out of range")
        model.observations = int(data.get("observations", 0))
        # Learned before: trust it more than a first guess.
        model._p = [[150.0**2, 0.0], [0.0, 80.0**2]]
        return model


class TurnSteerer:
    """Turns on the spot at a wanted rate, watching the heading (the gyro's)
    and adjusting the power as it goes; tells the TurnPower what it saw.
    Call `step` every control tick with the wanted rate and the heading now;
    it returns the steering command to send."""

    WINDOW_SECONDS = 0.4
    STALL_DEGREES = 2.0
    POWER_STEP = 0.1

    def __init__(self, power: TurnPower) -> None:
        self.power = power
        self.command: float | None = None
        self._window_start: tuple[float, float] | None = None
        self.stalled_at_max = False

    def step(self, rate_dps: float, heading: float, now: float) -> float:
        """Steering to turn at `rate_dps` (counterclockwise positive)."""
        wanted = abs(rate_dps)
        if self.command is None:
            self.command = self.power.command_for(wanted)
            self._window_start = (heading, now)
        start_heading, start_time = self._window_start
        elapsed = now - start_time
        if elapsed >= self.WINDOW_SECONDS:
            turned = abs(heading - start_heading)
            rate = turned / elapsed
            if turned < self.STALL_DEGREES:
                self.power.note_stall(self.command)
                self.stalled_at_max = self.command >= TurnPower.MAX_COMMAND - 1e-9
                self.command = min(TurnPower.MAX_COMMAND, max(self.command + self.POWER_STEP, self.power.command_for(wanted)))
            else:
                self.stalled_at_max = False
                self.power.note_turning(self.command, rate)
                # Toward the wanted rate, a step at a time (the floor, the
                # battery and the load all change what a power does).
                ratio = min(1.5, max(0.67, wanted / max(rate, 1.0)))
                above = max(self.command - self.power.breakaway, 0.02) * ratio
                self.command = min(TurnPower.MAX_COMMAND, max(TurnPower.MIN_COMMAND, self.power.breakaway + above))
            self._window_start = (heading, now)
        return math.copysign(self.command, rate_dps)

    def reset(self) -> None:
        self.command = None
        self._window_start = None
        self.stalled_at_max = False


class CoastModel:
    """How far the car keeps turning after a gyro turn cuts its motors, from
    the turn rate at that moment: a * rate**2 + b * rate.

    The motors are cut at once (not ramped), so the car freewheels to a stop:
    mostly a share in proportion to the rate (its momentum, and the gyro's
    reading running a little behind), with a little that grows faster. Both
    are fitted from what the gyro measures after each turn (recursive least
    squares, slowly forgetting) and kept within physical bounds, so a few
    odd turns cannot teach it nonsense."""

    FORGET = 0.95
    VERSION = 2
    A_LIMIT = 5e-4
    B_LIMIT = 0.25

    def __init__(self) -> None:
        self.a = 0.0
        self.b = 0.05
        self._p = [[(2e-4) ** 2, 0.0], [0.0, 0.05**2]]
        self.observations = 0

    def predict(self, rate_dps: float) -> float:
        rate = abs(rate_dps)
        return max(0.0, self.a * rate * rate + self.b * rate)

    def learn(self, rate_dps: float, coasted_degrees: float) -> None:
        rate = abs(rate_dps)
        if rate < 5.0:
            return
        phi = (rate * rate, rate)
        p = self._p
        p_phi = (p[0][0] * phi[0] + p[0][1] * phi[1], p[1][0] * phi[0] + p[1][1] * phi[1])
        gain_denominator = self.FORGET + phi[0] * p_phi[0] + phi[1] * p_phi[1]
        gain = (p_phi[0] / gain_denominator, p_phi[1] / gain_denominator)
        error = coasted_degrees - (self.a * phi[0] + self.b * phi[1])
        self.a = min(self.A_LIMIT, max(0.0, self.a + gain[0] * error))
        self.b = min(self.B_LIMIT, max(0.0, self.b + gain[1] * error))
        self._p = [
            [(p[i][j] - gain[i] * p_phi[j]) / self.FORGET for j in range(2)]
            for i in range(2)
        ]
        self.observations += 1

    def summary(self) -> dict[str, object]:
        return {"at_100_dps_deg": round(self.predict(100.0), 1), "at_300_dps_deg": round(self.predict(300.0), 1), "turns": self.observations}

    def to_json(self) -> dict[str, object]:
        return {"version": self.VERSION, "a": self.a, "b": self.b, "p": self._p, "observations": self.observations}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "CoastModel":
        model = cls()
        if data.get("version") != cls.VERSION:
            raise ValueError("coast learned with ramped stops; starting again")
        a, b, p = float(data["a"]), float(data["b"]), data["p"]
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= cls.A_LIMIT and 0 <= b <= cls.B_LIMIT):
            raise ValueError("coast model out of range")
        model.a, model.b = a, b
        model._p = [[float(p[0][0]), float(p[0][1])], [float(p[1][0]), float(p[1][1])]]
        model.observations = int(data.get("observations", 0))
        return model


# Effort: the duty sent is the command's times a gain, so the motors get the
# same voltage whatever the battery's state. The battery's reading (under load
# too: it sags as the motors draw) is compared with this voltage per cell; the
# gain may take the duty past the configured maximum (that sets the speed at
# this voltage), up to the board's full duty.
SUPPLY_REFERENCE_VOLTS_PER_CELL = 3.9
SUPPLY_GAIN_LIMITS = (0.85, 1.4)
SUPPLY_PLAUSIBLE_VOLTS_PER_CELL = (2.8, 4.4)
SUPPLY_READ_SECONDS = 0.5
# And from the IMU: half a second after the car is told to move, the gyro or
# accelerometer must see it moving; if not, the effort is raised a step and
# checked again, up to the most. Once it moves, what it took is remembered
# (blended) as where the next start begins, so a tired battery or a thick rug
# is met at once next time.
START_CHECK_SECONDS = 0.5
BOOST_STEP = 1.25
BOOST_LIMITS = (1.0, 2.0)
BOOST_LEARN = 0.2
MOVING_TURN_DPS = 4.0
MOVING_SPEED_CM_S = 6.0
# A command this different from the last counts as a new start.
COMMAND_CHANGE = 0.3

# A move measured by the accelerometer replaces the wheels' guess when the
# speed its sum still showed at the end (drift) is under this, and it is not
# implausibly far past what the wheels could have done.
MOVE_MAX_END_SPEED_CM_S = 25.0
MOVE_MAX_OVER_WHEELS = 3.0
MOVE_OVER_WHEELS_SLACK_CM = 10.0
# The board's forward direction is refined only from moves that went within
# this much of the way they were meant to.
FORWARD_REFINE_DEGREES = 30.0


class MotorDriver(Protocol):
    def set_motor_model(self, left_front: int, left_rear: int, right_front: int, right_rear: int) -> None: ...

    def close(self) -> None: ...


class DriveInhibitedError(RuntimeError):
    pass


class PanTiltInhibitedError(RuntimeError):
    pass


class SimulatedDriver:
    def __init__(self) -> None:
        self.command = (0, 0, 0, 0)

    def set_motor_model(self, left_front: int, left_rear: int, right_front: int, right_rear: int) -> None:
        self.command = (left_front, left_rear, right_front, right_rear)

    def close(self) -> None:
        self.command = (0, 0, 0, 0)


class FreenoveDriver:
    def __init__(self, server_path: Path | None) -> None:
        if server_path is None:
            raise ValueError("FREENOVE_SERVER_PATH is required when hardware is enabled")
        if not server_path.is_dir():
            raise FileNotFoundError(f"Freenove server directory not found: {server_path}")
        sys.path.insert(0, str(server_path))
        try:
            module = importlib.import_module("motor")
            self._car = module.Ordinary_Car()
        except OSError as exc:
            sys.path.remove(str(server_path))
            if exc.errno == 121:
                raise RuntimeError(
                    "Freenove PCA9685 did not acknowledge I2C address 0x40 on bus 1. "
                    "Check board power and wiring; `i2cdetect -y 1` should show `40`."
                ) from exc
            raise
        except Exception:
            sys.path.remove(str(server_path))
            raise
        self._server_path = str(server_path)

    def set_motor_model(self, left_front: int, left_rear: int, right_front: int, right_rear: int) -> None:
        self._car.set_motor_model(left_front, left_rear, right_front, right_rear)

    def close(self) -> None:
        try:
            self._car.close()
        finally:
            if self._server_path in sys.path:
                sys.path.remove(self._server_path)


class SimulatedPanTilt:
    def __init__(self) -> None:
        self.pan = DEFAULT_PAN
        self.tilt = DEFAULT_TILT

    def set_angles(self, pan: int, tilt: int) -> None:
        self.pan, self.tilt = pan, tilt

    def close(self) -> None:
        pass


class FreenovePanTilt:
    def __init__(self, server_path: Path | None) -> None:
        if server_path is None:
            raise ValueError("FREENOVE_SERVER_PATH is required when hardware is enabled")
        if not server_path.is_dir():
            raise FileNotFoundError(f"Freenove server directory not found: {server_path}")
        sys.path.insert(0, str(server_path))
        try:
            module = importlib.import_module("pca9685")
            self._pwm = module.PCA9685(0x40, debug=True)
            self._pwm.set_pwm_freq(50)
        except OSError as exc:
            sys.path.remove(str(server_path))
            if exc.errno == 121:
                raise RuntimeError(
                    "Freenove PCA9685 did not acknowledge I2C address 0x40 on bus 1. "
                    "Check board power and wiring; `i2cdetect -y 1` should show `40`."
                ) from exc
            raise
        except (ImportError, OSError):
            sys.path.remove(str(server_path))
            raise
        self._server_path = str(server_path)

    def set_angles(self, pan: int, tilt: int) -> None:
        # Freenove maps servo 0/1 to PCA9685 channels 8/9 and uses this pulse
        # conversion; configure only those two channels on the shared board.
        pan_pulse = 2500 - int((pan + 10) / 0.09)
        tilt_pulse = 500 + int((tilt + 10) / 0.09)
        self._pwm.set_servo_pulse(8, pan_pulse)
        self._pwm.set_servo_pulse(9, tilt_pulse)

    def close(self) -> None:
        try:
            self._pwm.close()
        finally:
            if self._server_path in sys.path:
                sys.path.remove(self._server_path)


class PanTiltController:
    def __init__(
        self, driver: SimulatedPanTilt | FreenovePanTilt, physical: bool, *, fault: str | None = None
    ) -> None:
        """With `fault` set (the servo board could not be reached) every move is
        refused with that reason, so the rest of the service still runs."""
        self._driver = driver
        self._physical = physical
        self._fault = fault
        self._lock = threading.Lock()
        self._inhibited = False
        self._pan, self._tilt = DEFAULT_PAN, DEFAULT_TILT
        self._driver.set_angles(self._pan, self._tilt)

    def set_angles(
        self,
        pan: int,
        tilt: int,
        *,
        override: bool = False,
    ) -> tuple[int, int]:
        if isinstance(pan, bool) or not isinstance(pan, int) or not PAN_LIMITS[0] <= pan <= PAN_LIMITS[1]:
            raise ValueError(f"pan must be an integer between {PAN_LIMITS[0]} and {PAN_LIMITS[1]}")
        if isinstance(tilt, bool) or not isinstance(tilt, int) or not TILT_LIMITS[0] <= tilt <= TILT_LIMITS[1]:
            raise ValueError(f"tilt must be an integer between {TILT_LIMITS[0]} and {TILT_LIMITS[1]}")
        with self._lock:
            if self._fault:
                raise PanTiltInhibitedError(f"Pan/tilt unavailable: {self._fault}")
            if self._inhibited and not override:
                raise PanTiltInhibitedError("Pan/tilt is controlled by the active mapping scan")
            self._driver.set_angles(pan, tilt)
            self._pan, self._tilt = pan, tilt
        return self._pan, self._tilt

    def set_inhibited(self, inhibited: bool) -> None:
        with self._lock:
            self._inhibited = inhibited

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self._physical and not self._fault,
                "mode": "unavailable" if self._fault else "hardware" if self._physical else "simulation",
                "fault": self._fault,
                "pan": self._pan,
                "tilt": self._tilt,
            }

    def close(self) -> None:
        self._driver.close()


class DriveController:
    def __init__(
        self,
        driver: MotorDriver,
        max_duty: int,
        command_timeout: float,
        physical: bool,
        *,
        fault: str | None = None,
        imu: "ImuController | None" = None,
        supply: Callable[[], float | None] | None = None,
        cells: int = 2,
    ) -> None:
        """With `fault` set (the motor board could not be reached) every command
        but stop is refused with that reason, so the rest of the service runs.

        With an IMU the odometry's heading is the gyro's (the wheels' estimate
        only while the gyro cannot be trusted), and the motors stay off while
        the car is tipped over or picked up.

        `supply` reads the battery's voltage (None while unknown): the duty is
        raised as it falls, and raised further whenever the IMU sees a
        command not moving the car (see START_CHECK_SECONDS)."""
        self._driver = driver
        self._imu = imu
        self._supply = supply
        self._reference_volts = SUPPLY_REFERENCE_VOLTS_PER_CELL * cells
        self._cells = cells
        self._supply_gain = 1.0
        self._supply_read_at = -math.inf
        self._boost = 1.0
        self._learned_boost = 1.0
        self._start_check: float | None = None
        self._start_confirmed = False
        self._boosts = 0
        self._gain = 1.0  # the whole gain on the duty, last tick
        # The gyro heading at the last odometry update (None: not in use).
        self._gyro_heading: float | None = None
        self._heading_source = "wheels"
        self._fault = fault
        self._max_duty = max_duty
        self._timeout = command_timeout
        self._physical = physical
        self._lock = threading.Lock()
        self._last_command = time.monotonic()
        self._command = (0, 0, 0)
        self._target_output = (0, 0, 0, 0)
        self._output = (0.0, 0.0, 0.0, 0.0)
        self._written_output = (0, 0, 0, 0)
        self._last_output_update = time.monotonic()
        self._output_error: Exception | None = None
        self._odometry = (0.0, 0.0, 0.0)
        self._inhibited = False
        self._closed = threading.Event()
        self._driver.set_motor_model(0, 0, 0, 0)
        self._output_worker = threading.Thread(
            target=self._output_loop,
            name="drive-output",
            daemon=True,
        )
        self._output_worker.start()

    def set_command(self, throttle: float, strafe: float, steering: float) -> tuple[float, float, float]:
        throttle = max(-1.0, min(1.0, float(throttle)))
        strafe = max(-1.0, min(1.0, float(strafe)))
        steering = max(-1.0, min(1.0, float(steering)))
        front_left = throttle + strafe - steering
        rear_left = throttle - strafe - steering
        front_right = throttle - strafe + steering
        rear_right = throttle + strafe + steering
        peak = max(1.0, abs(front_left), abs(rear_left), abs(front_right), abs(rear_right))
        with self._lock:
            if self._fault and (throttle, strafe, steering) != (0, 0, 0):
                raise DriveInhibitedError(f"Motor board unavailable: {self._fault}")
            if self._output_error is not None:
                raise RuntimeError("Drive output worker failed") from self._output_error
            if self._inhibited and (throttle, strafe, steering) != (0, 0, 0):
                raise DriveInhibitedError("Drive controls are locked during a mapping scan")
            if (throttle, strafe, steering) != (0, 0, 0) and self._upset():
                raise DriveInhibitedError("The car is tipped over or lifted; set it down on its wheels to drive")
            self._target_output = (
                round(front_left / peak * self._max_duty),
                round(rear_left / peak * self._max_duty),
                round(front_right / peak * self._max_duty),
                round(rear_right / peak * self._max_duty),
            )
            self._last_command = time.monotonic()
            command = (throttle, strafe, steering)
            if command == (0, 0, 0):
                self._start_check = None
                self._start_confirmed = False
            elif self._start_check is None or max(abs(a - b) for a, b in zip(command, self._command)) >= COMMAND_CHANGE:
                # A new start: begin where starts have needed to begin.
                self._start_check = self._last_command
                self._start_confirmed = False
                self._boost = self._learned_boost
            self._command = command
        return self._command

    def set_inhibited(self, inhibited: bool) -> bool:
        with self._lock:
            if inhibited and (
                self._command != (0, 0, 0)
                or any(output != 0 for output in self._output)
            ):
                return False
            self._inhibited = inhibited
            return True

    def inhibit_and_stop(self) -> None:
        with self._lock:
            self._inhibited = True
            self._stop_output_locked()
            self._command = (0, 0, 0)
            self._target_output = (0, 0, 0, 0)
            self._last_output_update = time.monotonic()
            self._last_command = time.monotonic()

    def stop(self, ramp: bool = True) -> None:
        """Stop the motors: ramped down, or with `ramp` off cut at once (they
        then freewheel to a stop, which is gentle; a gyro turn does this so
        that it ends where it is measured to, not a ramp's worth later)."""
        with self._lock:
            self._command = (0, 0, 0)
            self._target_output = (0, 0, 0, 0)
            self._last_command = time.monotonic()
            if self._output_error is not None or not ramp:
                # The output worker is gone, so nothing will ramp the motors down.
                self._stop_output_locked()

    def odometry(self) -> tuple[float, float, float]:
        """Dead-reckoned (x_cm, y_cm, heading_deg) since start; heading is counterclockwise."""
        with self._lock:
            return self._odometry

    def heading_source(self) -> str:
        """What the odometry's heading follows now: "gyro" or "wheels"."""
        with self._lock:
            return self._heading_source

    def has_gyro(self) -> bool:
        """Whether turns are measured by a gyro (fitted, perhaps still settling)."""
        return self._imu is not None and self._imu.fitted()

    def gyro_ready(self) -> bool:
        return self._imu is not None and self._imu.ready()

    def knock_since(self, since: float) -> float:
        """The hardest knock the IMU felt since then, in g (0 without one)."""
        return self._imu.knock_since(since) if self._imu is not None else 0.0

    def _upset(self) -> bool:
        return self._imu is not None and self._imu.upset()

    def set_supply(self, supply: Callable[[], float | None] | None, cells: int = 2) -> None:
        """Where to read the battery's voltage from (the power monitor)."""
        with self._lock:
            self._supply = supply
            self._cells = cells
            self._reference_volts = SUPPLY_REFERENCE_VOLTS_PER_CELL * cells
            self._supply_read_at = -math.inf

    def learned_boost(self) -> float:
        with self._lock:
            return self._learned_boost

    def set_learned_boost(self, boost: float) -> None:
        with self._lock:
            self._learned_boost = min(BOOST_LIMITS[1], max(BOOST_LIMITS[0], float(boost)))
            self._boost = self._learned_boost

    def _read_supply(self, now: float) -> None:
        """Refresh the battery gain now and then (outside the drive's lock)."""
        if self._supply is None or now - self._supply_read_at < SUPPLY_READ_SECONDS:
            return
        self._supply_read_at = now
        try:
            volts = self._supply()
        except Exception:
            volts = None
        low, high = SUPPLY_PLAUSIBLE_VOLTS_PER_CELL
        if volts is None or not low * self._cells <= volts <= high * self._cells:
            gain = 1.0
        else:
            gain = min(SUPPLY_GAIN_LIMITS[1], max(SUPPLY_GAIN_LIMITS[0], self._reference_volts / volts))
        with self._lock:
            self._supply_gain = gain

    def _check_start_locked(self, now: float) -> None:
        """Whether the car moves as told; if not, more effort."""
        if self._imu is None or self._start_check is None or self._start_confirmed:
            return
        if now - self._start_check < START_CHECK_SECONDS or not self._imu.ready():
            return
        throttle, strafe, steering = self._command
        turning = abs(steering) >= max(abs(throttle), abs(strafe))
        moving = abs(self._imu.rate_dps()) >= MOVING_TURN_DPS
        if not turning:
            moving = moving or self._imu.speed_cm_s() >= MOVING_SPEED_CM_S
        if moving:
            self._start_confirmed = True
            self._learned_boost += BOOST_LEARN * (self._boost - self._learned_boost)
            return
        if self._boost >= BOOST_LIMITS[1] - 1e-9:
            # As hard as it may push: keep pushing, but stop raising it.
            self._start_confirmed = True
            logger.warning("Not moving even at the most effort (%.2fx): stuck, or the battery is flat", self._boost)
            return
        self._boost = min(BOOST_LIMITS[1], self._boost * BOOST_STEP)
        self._boosts += 1
        self._start_check = now
        logger.info("Told to move but not moving: effort up to %.2fx", self._boost)

    @property
    def imu(self) -> "ImuController | None":
        return self._imu

    def measures_distance(self) -> bool:
        """Whether moves are measured by the accelerometer (its forward
        direction learned) rather than guessed from the wheels."""
        return self._imu is not None and self._imu.ready() and self._imu.forward_axis() is not None

    def begin_move(self) -> tuple[tuple[float, float, float], int] | None:
        """Start measuring a move with the accelerometer (the car at rest)."""
        if self._imu is None:
            return None
        move_id = self._imu.begin_move()
        return None if move_id is None else (self.odometry(), move_id)

    def finish_move(self, token, ahead_share: float, right_share: float) -> "tuple[MoveMeasure | None, bool]":
        """Finish measuring (the car at rest again). Learns the board's
        forward direction from moves going a known way (`ahead_share`,
        `right_share`: the commanded direction, 0, 0 if it turned too), and
        moves the odometry's position to where the accelerometer says when
        that is trustworthy. Returns the measure and whether it was used."""
        if token is None or self._imu is None:
            return None, False
        start, move_id = token
        measure = self._imu.end_move(move_id)
        if measure is None or measure.lost_samples or measure.end_speed_cm_s > MOVE_MAX_END_SPEED_CM_S:
            return measure, False
        if ahead_share or right_share:
            if measure.ahead_cm is None:
                self._imu.learn_forward(measure, ahead_share, right_share)
            else:
                went = math.degrees(math.atan2(measure.right_cm, measure.ahead_cm))
                meant = math.degrees(math.atan2(right_share, ahead_share))
                if abs((went - meant + 180) % 360 - 180) <= FORWARD_REFINE_DEGREES:
                    self._imu.learn_forward(measure, ahead_share, right_share)
        if measure.ahead_cm is None or measure.right_cm is None:
            return measure, False
        with self._lock:
            x, y, heading = self._odometry
            wheels = math.hypot(x - start[0], y - start[1])
            if math.hypot(measure.ahead_cm, measure.right_cm) > MOVE_MAX_OVER_WHEELS * wheels + MOVE_OVER_WHEELS_SLACK_CM:
                return measure, False
            # Where the move ended, in the frame it started in.
            radians = math.radians(start[2])
            self._odometry = (
                start[0] + math.cos(radians) * measure.right_cm - math.sin(radians) * measure.ahead_cm,
                start[1] + math.sin(radians) * measure.right_cm + math.cos(radians) * measure.ahead_cm,
                heading,
            )
        return measure, True

    def wait_until_stopped(self, timeout: float = MOTOR_RAMP_SECONDS + 0.1) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if not any(output != 0 for output in self._output):
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(MOTOR_UPDATE_INTERVAL_SECONDS, remaining))

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self._physical and not self._fault,
                "mode": "unavailable" if self._fault else "hardware" if self._physical else "simulation",
                "fault": self._fault,
                "throttle": self._command[0],
                "strafe": self._command[1],
                "steering": self._command[2],
                "max_duty": self._max_duty,
                "watchdog_timeout_seconds": self._timeout,
                "inhibited": self._inhibited,
                "output_error": str(self._output_error) if self._output_error else None,
                "heading_source": self._heading_source,
                "upset": self._upset(),
                "supply_gain": round(self._supply_gain, 3),
                "effort_boost": round(self._boost, 3),
                "learned_boost": round(self._learned_boost, 3),
                "boosts": self._boosts,
            }

    def _output_loop(self) -> None:
        try:
            while not self._closed.wait(MOTOR_UPDATE_INTERVAL_SECONDS):
                self._read_supply(time.monotonic())
                with self._lock:
                    now = time.monotonic()
                    previous_output = self._output
                    previous_gain = self._gain
                    self._check_start_locked(now)
                    if self._command != (0, 0, 0) and now - self._last_command > self._timeout:
                        logger.warning("Drive command expired; stopping motors")
                        self._command = (0, 0, 0)
                        self._target_output = (0, 0, 0, 0)
                        self._output = (0.0, 0.0, 0.0, 0.0)
                    elif self._command != (0, 0, 0) and self._upset():
                        logger.warning("The car is tipped over or lifted; stopping motors")
                        self._command = (0, 0, 0)
                        self._target_output = (0, 0, 0, 0)
                        self._output = (0.0, 0.0, 0.0, 0.0)
                    else:
                        elapsed = now - self._last_output_update
                        self._gain = self._supply_gain * self._boost
                        targets = tuple(
                            max(-FULL_MOTOR_DUTY, min(FULL_MOTOR_DUTY, target * self._gain)) for target in self._target_output
                        )
                        max_change = self._max_duty * self._gain * elapsed / MOTOR_RAMP_SECONDS
                        self._output = tuple(
                            self._approach(current, target, max_change)
                            for current, target in zip(self._output, targets)
                        )
                    # The wheels' estimate goes by the duty the command meant:
                    # the gain only makes up for lost voltage or grip.
                    self._integrate_odometry(
                        tuple(value / previous_gain for value in previous_output),
                        tuple(value / self._gain for value in self._output),
                        now - self._last_output_update,
                    )
                    self._last_output_update = now
                    output = tuple(round(value) for value in self._output)
                    if output != self._written_output:
                        self._driver.set_motor_model(*output)
                        self._written_output = output
        except Exception as exc:
            logger.exception("Drive output worker failed; stopping motors")
            with self._lock:
                self._output_error = exc
                self._command = (0, 0, 0)
                self._target_output = (0, 0, 0, 0)
                try:
                    self._stop_output_locked()
                except Exception:
                    logger.exception("Could not stop motors after drive output failure")

    def _stop_output_locked(self) -> None:
        self._driver.set_motor_model(0, 0, 0, 0)
        self._output = (0.0, 0.0, 0.0, 0.0)
        self._written_output = (0, 0, 0, 0)

    def _integrate_odometry(
        self,
        previous_output: tuple[float, ...],
        output: tuple[float, ...],
        elapsed: float,
    ) -> None:
        gyro = self._imu.heading_now() if self._imu is not None else None
        if gyro is None:
            self._gyro_heading = None
            self._heading_source = "wheels"
        moving = elapsed > 0 and (any(previous_output) or any(output))
        if not moving and gyro is None:
            return
        front_left, rear_left, front_right, rear_right = (
            self._wheel_speed_fraction((before + after) / 2) if moving else 0.0
            for before, after in zip(previous_output, output)
        )
        # Inverse of the mecanum mix in set_command.
        forward = (front_left + rear_left + front_right + rear_right) / 4
        strafe = (front_left - rear_left - front_right + rear_right) / 4
        turn = (-front_left - rear_left + front_right + rear_right) / 4
        x, y, heading = self._odometry
        if gyro is not None:
            # The gyro's turn since the last update, whether the motors made
            # it or not (a push counts too); picked up where it left off.
            turn_degrees = 0.0 if self._gyro_heading is None else gyro - self._gyro_heading
            self._gyro_heading = gyro
            self._heading_source = "gyro"
        else:
            turn_degrees = turn * TURN_DEGREES_PER_SECOND_AT_FULL_DUTY * elapsed
        if not moving:
            self._odometry = (x, y, heading + turn_degrees)
            return
        right = strafe * STRAFE_CM_PER_SECOND_AT_FULL_DUTY * elapsed
        ahead = forward * FORWARD_CM_PER_SECOND_AT_FULL_DUTY * elapsed
        middle = math.radians(heading + turn_degrees / 2)
        self._odometry = (
            x + math.cos(middle) * right - math.sin(middle) * ahead,
            y + math.sin(middle) * right + math.cos(middle) * ahead,
            heading + turn_degrees,
        )

    @staticmethod
    def _wheel_speed_fraction(duty: float) -> float:
        fraction = abs(duty) / FULL_MOTOR_DUTY
        effective = max(0.0, fraction - MOTOR_DEADBAND_FRACTION) / (1 - MOTOR_DEADBAND_FRACTION)
        return math.copysign(effective, duty)

    @staticmethod
    def _approach(current: float, target: int, max_change: float) -> float:
        if current < target:
            return min(current + max_change, target)
        if current > target:
            return max(current - max_change, target)
        return current

    def close(self) -> None:
        self._closed.set()
        self._output_worker.join(timeout=1)
        with self._lock:
            self._stop_output_locked()
        self._driver.close()
