import importlib
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)
MOTOR_RAMP_SECONDS = 0.25
MOTOR_UPDATE_INTERVAL_SECONDS = 0.01
FULL_MOTOR_DUTY = 4095
# Wheel odometry calibration. The forward speed is the earlier estimate of
# 30.48 cm per 0.25 s at full duty; the deadband, strafe and turn rates are
# first guesses. Mapping refines its own scale from sonar registration and camera yaw,
# but calibrate these on the floor for a better starting estimate.
MOTOR_DEADBAND_FRACTION = 0.1
FORWARD_CM_PER_SECOND_AT_FULL_DUTY = 121.9
STRAFE_CM_PER_SECOND_AT_FULL_DUTY = 97.5
TURN_DEGREES_PER_SECOND_AT_FULL_DUTY = 48.0


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
        self.pan = 90
        self.tilt = 90

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
    def __init__(self, driver: SimulatedPanTilt | FreenovePanTilt, physical: bool) -> None:
        self._driver = driver
        self._physical = physical
        self._lock = threading.Lock()
        self._inhibited = False
        self._pan, self._tilt = 90, 90
        self._driver.set_angles(self._pan, self._tilt)

    def set_angles(
        self,
        pan: int,
        tilt: int,
        *,
        override: bool = False,
    ) -> tuple[int, int]:
        if isinstance(pan, bool) or not isinstance(pan, int) or not 30 <= pan <= 150:
            raise ValueError("pan must be an integer between 30 and 150")
        if isinstance(tilt, bool) or not isinstance(tilt, int) or not 30 <= tilt <= 150:
            raise ValueError("tilt must be an integer between 30 and 150")
        with self._lock:
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
                "enabled": self._physical,
                "mode": "hardware" if self._physical else "simulation",
                "pan": self._pan,
                "tilt": self._tilt,
            }

    def close(self) -> None:
        self._driver.close()


class DriveController:
    def __init__(self, driver: MotorDriver, max_duty: int, command_timeout: float, physical: bool) -> None:
        self._driver = driver
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
            if self._output_error is not None:
                raise RuntimeError("Drive output worker failed") from self._output_error
            if self._inhibited and (throttle, strafe, steering) != (0, 0, 0):
                raise DriveInhibitedError("Drive controls are locked during a mapping scan")
            self._target_output = (
                round(front_left / peak * self._max_duty),
                round(rear_left / peak * self._max_duty),
                round(front_right / peak * self._max_duty),
                round(rear_right / peak * self._max_duty),
            )
            self._last_command = time.monotonic()
            self._command = (throttle, strafe, steering)
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

    def stop(self) -> None:
        with self._lock:
            self._command = (0, 0, 0)
            self._target_output = (0, 0, 0, 0)
            self._last_command = time.monotonic()
            if self._output_error is not None:
                # The output worker is gone, so nothing will ramp the motors down.
                self._stop_output_locked()

    def odometry(self) -> tuple[float, float, float]:
        """Dead-reckoned (x_cm, y_cm, heading_deg) since start; heading is counterclockwise."""
        with self._lock:
            return self._odometry

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
                "enabled": self._physical,
                "mode": "hardware" if self._physical else "simulation",
                "throttle": self._command[0],
                "strafe": self._command[1],
                "steering": self._command[2],
                "max_duty": self._max_duty,
                "watchdog_timeout_seconds": self._timeout,
                "inhibited": self._inhibited,
                "output_error": str(self._output_error) if self._output_error else None,
            }

    def _output_loop(self) -> None:
        try:
            while not self._closed.wait(MOTOR_UPDATE_INTERVAL_SECONDS):
                with self._lock:
                    now = time.monotonic()
                    previous_output = self._output
                    if self._command != (0, 0, 0) and now - self._last_command > self._timeout:
                        logger.warning("Drive command expired; stopping motors")
                        self._command = (0, 0, 0)
                        self._target_output = (0, 0, 0, 0)
                        self._output = (0.0, 0.0, 0.0, 0.0)
                    else:
                        elapsed = now - self._last_output_update
                        max_change = self._max_duty * elapsed / MOTOR_RAMP_SECONDS
                        self._output = tuple(
                            self._approach(current, target, max_change)
                            for current, target in zip(self._output, self._target_output)
                        )
                    self._integrate_odometry(previous_output, self._output, now - self._last_output_update)
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
        if elapsed <= 0 or not any(previous_output) and not any(output):
            return
        front_left, rear_left, front_right, rear_right = (
            self._wheel_speed_fraction((before + after) / 2)
            for before, after in zip(previous_output, output)
        )
        # Inverse of the mecanum mix in set_command.
        forward = (front_left + rear_left + front_right + rear_right) / 4
        strafe = (front_left - rear_left - front_right + rear_right) / 4
        turn = (-front_left - rear_left + front_right + rear_right) / 4
        x, y, heading = self._odometry
        turn_degrees = turn * TURN_DEGREES_PER_SECOND_AT_FULL_DUTY * elapsed
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
