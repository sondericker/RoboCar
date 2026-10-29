import importlib
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)
# gpiozero's DistanceSensor pings every 0.06 s in the background and reports
# the median of its last `queue_len` pings. Freenove leaves that at 9 (about
# half a second), so readings taken right after the head moved still mixed in
# earlier directions and smeared each sweep. Three pings keep a median but
# refresh within about 0.2 s.
GPIOZERO_PING_SECONDS = 0.06
RANGE_QUEUE_LENGTH = 3
GPIOZERO_DEFAULT_QUEUE_LENGTH = 9


class RangeSensor(Protocol):
    def get_distance(self) -> float | None: ...

    def close(self) -> None: ...


class FreenoveRangeSensor:
    def __init__(self, server_path: Path | None) -> None:
        if server_path is None:
            raise ValueError("FREENOVE_SERVER_PATH is required when hardware is enabled")
        if not server_path.is_dir():
            raise FileNotFoundError(f"Freenove server directory not found: {server_path}")
        sys.path.insert(0, str(server_path))
        try:
            module = importlib.import_module("ultrasonic")
            self._sensor = module.Ultrasonic()
        except Exception:
            sys.path.remove(str(server_path))
            raise
        self._server_path = str(server_path)
        self.queue_length = self._shorten_smoothing()

    def _shorten_smoothing(self) -> int:
        """Rebuild the kit's DistanceSensor with a short median; returns the queue length in use."""
        device = getattr(self._sensor, "sensor", None)
        try:
            gpiozero = importlib.import_module("gpiozero")
            pins = (self._sensor.echo_pin, self._sensor.trigger_pin, self._sensor.max_distance)
        except (ImportError, AttributeError):
            return GPIOZERO_DEFAULT_QUEUE_LENGTH
        if device is None:
            return GPIOZERO_DEFAULT_QUEUE_LENGTH
        echo, trigger, max_distance = pins
        try:
            device.close()
            self._sensor.sensor = gpiozero.DistanceSensor(
                echo=echo,
                trigger=trigger,
                max_distance=max_distance,
                queue_len=RANGE_QUEUE_LENGTH,
                partial=True,
            )
            return RANGE_QUEUE_LENGTH
        except Exception as exc:
            logger.warning("Could not shorten ultrasonic smoothing (%s); sweeps will wait longer", exc)
            try:
                self._sensor.sensor = gpiozero.DistanceSensor(echo=echo, trigger=trigger, max_distance=max_distance)
            except Exception:
                logger.exception("Could not restore the ultrasonic sensor")
            return GPIOZERO_DEFAULT_QUEUE_LENGTH

    @property
    def settle_seconds(self) -> float:
        """Time after pointing elsewhere until a reading reflects only the new direction."""
        return self.queue_length * GPIOZERO_PING_SECONDS + 0.02

    def get_distance(self) -> float | None:
        return self._sensor.get_distance()

    def close(self) -> None:
        try:
            self._sensor.close()
        finally:
            if self._server_path in sys.path:
                sys.path.remove(self._server_path)


class RangeSensorController:
    def __init__(
        self,
        driver: RangeSensor | None,
        physical: bool,
        sample_interval: float = 0.2,
        *,
        fault: str | None = None,
    ) -> None:
        if sample_interval <= 0:
            raise ValueError("sample_interval must be greater than 0")
        self._driver = driver
        self._physical = physical
        self._sample_interval = sample_interval
        self._lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._distance_cm: float | None = None
        self._fault = fault
        self._error: str | None = fault
        self._updated_at: float | None = None
        self._closed = threading.Event()
        self._sampler: threading.Thread | None = None
        if driver is not None:
            self._sampler = threading.Thread(
                target=self._sample_loop,
                name="range-sensor",
                daemon=True,
            )
            self._sampler.start()

    @property
    def settle_seconds(self) -> float:
        """How long readings lag a change of direction (0 for drivers that ping on request)."""
        return float(getattr(self._driver, "settle_seconds", 0.0))

    def status(self) -> dict[str, object]:
        with self._lock:
            unavailable = self._physical and self._driver is None
            return {
                "enabled": self._physical and not unavailable,
                "mode": "unavailable" if unavailable else "hardware" if self._physical else "disabled",
                "fault": self._fault,
                "distance_cm": self._distance_cm,
                "error": self._error,
                "updated_at": self._updated_at,
            }

    def sample_now(self) -> float | None:
        if self._driver is None:
            return None
        with self._read_lock:
            distance = self._driver.get_distance()
        if (
            distance is None
            or isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(distance)
            or distance < 0
        ):
            error = "No echo received" if distance is None else "Sensor returned an invalid distance"
            with self._lock:
                self._distance_cm = None
                self._error = error
                self._updated_at = time.time()
            return None
        value = float(distance)
        with self._lock:
            self._distance_cm = value
            self._error = None
            self._updated_at = time.time()
        return value

    def _sample_loop(self) -> None:
        assert self._driver is not None
        while not self._closed.is_set():
            error = None
            distance = None
            try:
                with self._read_lock:
                    distance = self._driver.get_distance()
                if distance is None:
                    error = "No echo received"
                elif isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 0:
                    error = "Sensor returned an invalid distance"
                    distance = None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    previous_error = self._error
                if error != previous_error:
                    logger.exception("Ultrasonic range reading failed")

            with self._lock:
                self._distance_cm = float(distance) if distance is not None else None
                self._error = error
                self._updated_at = time.time()
            self._closed.wait(self._sample_interval)

    def close(self) -> None:
        self._closed.set()
        if self._sampler is not None:
            self._sampler.join()
        if self._driver is not None:
            self._driver.close()
