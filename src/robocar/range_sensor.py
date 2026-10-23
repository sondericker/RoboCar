import importlib
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


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
    ) -> None:
        if sample_interval <= 0:
            raise ValueError("sample_interval must be greater than 0")
        self._driver = driver
        self._physical = physical
        self._sample_interval = sample_interval
        self._lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._distance_cm: float | None = None
        self._error: str | None = None
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

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self._physical,
                "mode": "hardware" if self._physical else "disabled",
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
