"""Battery and supply voltages: the car's battery and the Pi's own power.

The Freenove board measures its battery through an ADS7830 converter at I²C
address 0x48, channel 2, behind a voltage divider. Board version 1 uses a
3.3 V reference and divides by 3; version 2 a 5.2 V reference and divides by
2 (as the kit's own adc.py does). The version comes from the kit's
params.json. Channels 0 and 1 are the two light sensors.

The battery is two lithium cells in series. Its charge is estimated from a
resting discharge curve, judged only while the motors are idle: driving
pulls the voltage down for a while.

The Pi 5's power-management chip reports its input (the 5 V supply), its
internal rails and their currents through `vcgencmd pmic_read_adc`, and
`vcgencmd get_throttled` says whether the supply has sagged (under-voltage)
or the CPU was slowed. The CPU temperature comes from sysfs.

Every reading is optional: anything that cannot be read is reported as
unavailable with the reason, and nothing here can stop the rest of RoboCar.
"""

import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

ADC_ADDRESS = 0x48
ADS7830_COMMAND = 0x84
BATTERY_CHANNEL = 2
LIGHT_CHANNELS = (0, 1)
# Resting voltage of one lithium-ion cell against charge left, in percent.
CELL_CURVE = (
    (3.00, 0.0), (3.30, 2.0), (3.40, 5.0), (3.50, 10.0), (3.60, 20.0), (3.70, 35.0),
    (3.80, 50.0), (3.90, 65.0), (4.00, 80.0), (4.10, 90.0), (4.20, 100.0),
)
# Resting volts per cell: below about 27% charge "getting low" (charge
# soon), 10% "low" (charge now), and "critical" (no driving on its own).
GETTING_LOW_CELL_VOLTS = 3.65
LOW_CELL_VOLTS = 3.5
CRITICAL_CELL_VOLTS = 3.3
# The motors pull the battery down; readings this soon after driving are not resting.
RESTING_AFTER_SECONDS = 3.0
SAMPLE_INTERVAL_SECONDS = 2.0
# Median of this many readings, to ride over noise in the 8-bit converter.
BATTERY_SAMPLES = 5
PI_RAILS = ("EXT5V", "3V3_SYS", "1V8_SYS", "1V1_SYS", "VDD_CORE")
THROTTLE_FLAGS = {
    0: "under_voltage_now",
    1: "frequency_capped_now",
    2: "throttled_now",
    3: "temperature_limit_now",
    16: "under_voltage_since_boot",
    17: "frequency_capped_since_boot",
    18: "throttled_since_boot",
    19: "temperature_limit_since_boot",
}


def board_version(server_path: Path | None) -> int:
    """The Freenove board version from the kit's params.json (2 when unknown)."""
    if server_path is None:
        return 2
    try:
        version = json.loads((server_path / "params.json").read_text()).get("Pcb_Version")
    except (OSError, ValueError, AttributeError):
        return 2
    return version if version in (1, 2) else 2


def charge_percent(volts: float, cells: int) -> float:
    """Charge left from a resting battery voltage."""
    cell = volts / cells
    if cell <= CELL_CURVE[0][0]:
        return 0.0
    for (low_volts, low_percent), (high_volts, high_percent) in zip(CELL_CURVE, CELL_CURVE[1:]):
        if cell <= high_volts:
            return low_percent + (cell - low_volts) / (high_volts - low_volts) * (high_percent - low_percent)
    return 100.0


class BoardADC:
    """The ADS7830 on the Freenove board."""

    def __init__(self, version: int = 2, bus: int = 1) -> None:
        try:
            from smbus2 import SMBus
        except ImportError:
            from smbus import SMBus  # Raspberry Pi OS's python3-smbus
        self._bus = SMBus(bus)
        self._lock = threading.Lock()
        self.version = version
        self._reference = 3.3 if version == 1 else 5.2
        self._divider = 3.0 if version == 1 else 2.0
        # Fail now, not on the first sample, if the converter is not there.
        self.read_volts(BATTERY_CHANNEL)

    def read_volts(self, channel: int) -> float:
        """Voltage at a converter pin."""
        command = ADS7830_COMMAND | ((((channel << 2) | (channel >> 1)) & 0x07) << 4)
        with self._lock:
            self._bus.write_byte(ADC_ADDRESS, command)
            # Two equal reads in a row, as the kit does, but never forever.
            value = self._bus.read_byte(ADC_ADDRESS)
            for _ in range(10):
                again = self._bus.read_byte(ADC_ADDRESS)
                if again == value:
                    break
                value = again
        return value / 255.0 * self._reference

    def battery_volts(self) -> float:
        return self.read_volts(BATTERY_CHANNEL) * self._divider

    def close(self) -> None:
        self._bus.close()


def _vcgencmd(*arguments: str) -> str:
    return subprocess.run(
        ["vcgencmd", *arguments], capture_output=True, text=True, timeout=2.0, check=True
    ).stdout


def parse_pmic(text: str) -> dict[str, dict[str, float]]:
    """`vcgencmd pmic_read_adc` output as {rail: {"volts": .., "amps": ..}}."""
    rails: dict[str, dict[str, float]] = {}
    for name, kind, value in re.findall(r"(\S+)_([VA])\s+(?:volt|current)\(\d+\)=([-\d.]+)", text):
        rails.setdefault(name, {})["volts" if kind == "V" else "amps"] = float(value)
    return rails


def parse_throttled(text: str) -> dict[str, bool]:
    match = re.search(r"throttled=(0x[0-9a-fA-F]+)", text)
    if not match:
        raise ValueError(f"Unexpected get_throttled output: {text!r}")
    bits = int(match.group(1), 16)
    return {name: bool(bits >> bit & 1) for bit, name in THROTTLE_FLAGS.items()}


class PiPower:
    """The Raspberry Pi's own supply, rails, throttling and temperature."""

    def __init__(self, run: Callable[..., str] = _vcgencmd, thermal: Path = Path("/sys/class/thermal/thermal_zone0/temp")) -> None:
        self._run = run
        self._thermal = thermal

    def read(self) -> dict[str, object]:
        reading: dict[str, object] = {}
        try:
            rails = parse_pmic(self._run("pmic_read_adc"))
            reading["rails"] = {
                name: round(rails[name]["volts"], 3) for name in PI_RAILS if "volts" in rails.get(name, {})
            }
            # The PMIC rails (not USB or HATs, which draw from the 5 V input directly).
            reading["board_watts"] = round(
                sum(rail["volts"] * rail["amps"] for rail in rails.values() if "volts" in rail and "amps" in rail), 2
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            reading["rails_error"] = f"{type(exc).__name__}: {exc}"
        try:
            reading["flags"] = parse_throttled(self._run("get_throttled"))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            reading["flags_error"] = f"{type(exc).__name__}: {exc}"
        try:
            reading["cpu_celsius"] = round(int(self._thermal.read_text().strip()) / 1000.0, 1)
        except (OSError, ValueError):
            pass
        return reading


@dataclass
class _Battery:
    volts: float | None = None  # latest reading
    resting_volts: float | None = None  # latest reading with the motors idle a while
    error: str | None = None
    updated_at: float | None = None


class PowerMonitor:
    """Samples the battery and the Pi's supply in the background."""

    def __init__(
        self,
        adc: BoardADC | None,
        pi: PiPower | None,
        motors_running: Callable[[], bool],
        *,
        cells: int = 2,
        adc_fault: str | None = None,
        interval: float = SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._adc = adc
        self._pi = pi
        self._motors_running = motors_running
        self._cells = cells
        self._interval = interval
        self._lock = threading.Lock()
        self._battery = _Battery(error=adc_fault)
        self._pi_reading: dict[str, object] = {}
        self._last_motion = 0.0
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="power-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._closed.is_set():
            self.sample()
            self._closed.wait(self._interval)

    def sample(self) -> None:
        now = time.monotonic()
        try:
            moving = bool(self._motors_running())
        except Exception:
            moving = True  # unsure: do not take the reading as resting
        if moving:
            self._last_motion = now
        if self._adc is not None:
            try:
                readings = sorted(self._adc.battery_volts() for _ in range(BATTERY_SAMPLES))
                volts = readings[len(readings) // 2]
                resting = not moving and now - self._last_motion >= RESTING_AFTER_SECONDS
                with self._lock:
                    self._battery.volts = volts
                    if resting:
                        self._battery.resting_volts = volts
                    self._battery.error = None
                    self._battery.updated_at = time.time()
            except Exception as exc:
                with self._lock:
                    if self._battery.error is None:
                        logger.warning("Battery reading failed: %s", exc)
                    self._battery.error = f"{type(exc).__name__}: {exc}"
        if self._pi is not None:
            try:
                reading = self._pi.read()
            except Exception as exc:  # never let a reading take the monitor down
                reading = {"rails_error": f"{type(exc).__name__}: {exc}"}
            with self._lock:
                self._pi_reading = reading

    def battery_level(self) -> str | None:
        """"ok", "getting_low", "low" or "critical" from the resting voltage;
        None while unknown."""
        with self._lock:
            volts = self._battery.resting_volts
        if volts is None:
            return None
        cell = volts / self._cells
        if cell <= CRITICAL_CELL_VOLTS:
            return "critical"
        if cell <= LOW_CELL_VOLTS:
            return "low"
        return "getting_low" if cell <= GETTING_LOW_CELL_VOLTS else "ok"

    def status(self) -> dict[str, object]:
        level = self.battery_level()
        with self._lock:
            battery = self._battery
            resting = battery.resting_volts
            return {
                "battery": {
                    "available": self._adc is not None,
                    "volts": None if battery.volts is None else round(battery.volts, 2),
                    "resting_volts": None if resting is None else round(resting, 2),
                    "percent": None if resting is None else round(charge_percent(resting, self._cells)),
                    # Where the meter's warning marks go.
                    "getting_low_percent": round(charge_percent(GETTING_LOW_CELL_VOLTS * self._cells, self._cells)),
                    "low_percent": round(charge_percent(LOW_CELL_VOLTS * self._cells, self._cells)),
                    "level": level,
                    "cells": self._cells,
                    "under_load": battery.volts is not None and battery.volts != resting,
                    "error": battery.error,
                    "updated_at": battery.updated_at,
                },
                "pi": dict(self._pi_reading),
            }

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=5)
        if self._adc is not None:
            try:
                self._adc.close()
            except Exception:
                logger.debug("Closing the battery converter failed", exc_info=True)
