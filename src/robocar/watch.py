"""Eyes that never stop: continuous detection on the live camera feed.

During the hunt a background thread grabs frame after frame, runs object
detection on each (on the AI HAT+, about 12 ms a frame), and publishes what
it saw: the cat if any, whether a person stands close in front, where the
head pointed and where the car was when the frame was taken. Whoever needs
to know reads the latest result; nothing waits for a picture to be taken.

The moment a cat appears the watcher raises `spotted`, so the hunt can drop
whatever it is doing (a turn, a drive, a sonar sweep) and go after it.

Sharp frames while moving come from short exposures. The watcher runs its
own exposure control: it keeps the brightness near a target with exposure
time and gain, but caps the exposure while the car or head is moving (and
allows longer ones while still). The moving cap adapts: if frames taken on
the move come out clearly softer than still ones, it shortens (trading
blur for grain, which detection copes with far better); if they are about
as sharp, it lengthens again for less noise. Sharpness is measured
relative to contrast, and the cap is never shortened while the light runs
short (gain at its limit): a darker frame also scores as softer, and
chasing that once spiralled the exposure down to black frames. In dim
light the cap grows instead, trading a little blur for a visible cat.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from robocar.hunt import Sighting, cat_view_from
from robocar.localization import Pose

logger = logging.getLogger(__name__)

# Exposure while moving starts here and adapts between the limits; still
# frames may use up to STILL_CAP_US. Gain covers the rest, up to MAX_GAIN.
MOVING_CAP_US = 8_000
MOVING_CAP_LIMITS_US = (2_000, 16_000)
STILL_CAP_US = 33_000
MIN_EXPOSURE_US = 200
MAX_GAIN = 12.0
TARGET_BRIGHTNESS = 110.0
# Brightness corrections are damped and only sent when worth it.
EXPOSURE_DAMPING = 0.6
EXPOSURE_CHANGE_WORTH = 0.08
# Moving frames this much softer than still ones shorten the moving cap;
# this close to still ones lengthen it.
SOFT_MOVING_RATIO = 0.7
SHARP_MOVING_RATIO = 0.9
SHARPNESS_SAMPLES = 15
# Darker than this with the gain at its limit, moving frames need more light.
DIM_BRIGHTNESS = 0.6 * TARGET_BRIGHTNESS
# The dashboard picture is refreshed at most this often (encoding costs time).
PUBLISH_INTERVAL_SECONDS = 0.15
# Settings are sent again this often anyway: a sonar sweep's stop picture
# sets exposures of its own.
EXPOSURE_RESEND_SECONDS = 2.0
# Strip pictures for a panorama, one per this many degrees of pan.
PANORAMA_BIN_DEGREES = 10


def brightness_of(frame: np.ndarray) -> float:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = grey[::4, ::4]
    return float(small.mean())


def sharpness_of(frame: np.ndarray) -> float:
    """Detail relative to contrast: the Laplacian's variance over the
    picture's own, so a darker frame of the same scene scores the same."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = grey[::2, ::2]
    return float(cv2.Laplacian(small, cv2.CV_64F).var() / (float(small.var()) + 25.0))


@dataclass
class MotionExposure:
    """Exposure control that keeps moving frames sharp."""

    moving_cap_us: float = MOVING_CAP_US
    exposure_us: float = MOVING_CAP_US
    gain: float = 4.0
    moving_sharpness: deque = field(default_factory=lambda: deque(maxlen=SHARPNESS_SAMPLES))
    still_sharpness: deque = field(default_factory=lambda: deque(maxlen=SHARPNESS_SAMPLES))

    def settings(self) -> dict[str, float]:
        return {"exposure_us": int(round(self.exposure_us)), "gain": round(self.gain, 2)}

    def update(self, brightness: float, moving: bool) -> dict[str, float] | None:
        """New settings for this brightness and state, or None if not worth changing."""
        cap = self.moving_cap_us if moving else STILL_CAP_US
        wanted = (TARGET_BRIGHTNESS / max(brightness, 1.0)) ** EXPOSURE_DAMPING
        wanted = min(2.5, max(0.4, wanted))
        light = self.exposure_us * self.gain * wanted  # the exposure x gain product wanted
        # Exposure first (no noise) up to the cap, then gain.
        exposure = min(cap, max(MIN_EXPOSURE_US, light))
        gain = min(MAX_GAIN, max(1.0, light / exposure))
        changed = (
            abs(exposure - self.exposure_us) / self.exposure_us > EXPOSURE_CHANGE_WORTH
            or abs(gain - self.gain) / self.gain > EXPOSURE_CHANGE_WORTH
        )
        if not changed:
            return None
        self.exposure_us, self.gain = exposure, gain
        return self.settings()

    def light_limited(self) -> bool:
        """Whether the gain is at its limit, so a shorter exposure only darkens."""
        return self.gain >= MAX_GAIN * 0.95

    def note_sharpness(self, sharpness: float, moving: bool, brightness: float | None = None) -> None:
        """Learn how much softer moving frames are than still ones, and set
        the moving cap from it."""
        low, high = MOVING_CAP_LIMITS_US
        if moving and self.light_limited() and brightness is not None and brightness < DIM_BRIGHTNESS:
            # Too dark to see well: a longer exposure helps more than
            # sharpness does. Grow back, and forget comparisons made darker.
            self.moving_cap_us = min(high, self.moving_cap_us * 1.1)
            self.moving_sharpness.clear()
            return
        (self.moving_sharpness if moving else self.still_sharpness).append(sharpness)
        if len(self.moving_sharpness) < SHARPNESS_SAMPLES // 2 or len(self.still_sharpness) < SHARPNESS_SAMPLES // 2:
            return
        moving_typical = float(np.median(self.moving_sharpness))
        still_typical = float(np.median(self.still_sharpness))
        if still_typical <= 0:
            return
        ratio = moving_typical / still_typical
        if ratio < SOFT_MOVING_RATIO and not self.light_limited():
            self.moving_cap_us = max(low, self.moving_cap_us * 0.8)
            self.moving_sharpness.clear()
        elif ratio > SHARP_MOVING_RATIO:
            self.moving_cap_us = min(high, self.moving_cap_us * 1.1)
            self.moving_sharpness.clear()


class Watcher:
    """Continuous detection on a background thread."""

    def __init__(
        self,
        grab: Callable[[], np.ndarray | None],
        detect: Callable[[np.ndarray], list[dict[str, object]]],
        pan: Callable[[], int],
        odometry: Callable[[], Pose],
        moving: Callable[[], bool],
        *,
        publish: Callable[[np.ndarray, list[dict[str, object]]], None] | None = None,
        set_exposure: Callable[[dict[str, float]], None] | None = None,
        interrupt: threading.Event | None = None,
        on_sighting: Callable[[Sighting], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._grab = grab
        self._detect = detect
        self._pan = pan
        self._odometry = odometry
        self._moving = moving
        self._publish = publish
        self._set_exposure = set_exposure
        self._clock = clock
        self._on_sighting = on_sighting
        self._condition = threading.Condition()
        # The latest frame taken with the car and head still (for placing the car).
        self._still: tuple[Sighting, np.ndarray] | None = None
        self._latest: Sighting | None = None
        self._spotted: Sighting | None = None
        self._interrupts = False
        # Set when a cat is seen while interrupts are on.
        self.interrupt = interrupt if interrupt is not None else threading.Event()
        self.exposure = MotionExposure()
        self._frames = 0
        self._started_at = clock()
        self._fps = 0.0
        self._last_publish = 0.0
        self._last_exposure_sent = -float("inf")
        self._collecting: dict[int, bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # -- running ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._set_exposure is not None:
            self._set_exposure(self.exposure.settings())
        self._thread = threading.Thread(target=self._loop, name="hunt-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self._condition:
            self._condition.notify_all()
        if self._set_exposure is not None:
            # Hand the camera back on auto-exposure: the dashboard's live view
            # (or mapping, with one camera) should not keep a short exposure.
            try:
                self._set_exposure({"ev": 0.0})
            except Exception:
                logger.warning("Could not put the camera back on auto-exposure", exc_info=True)

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                self.step()
                failures = 0
            except Exception as exc:  # the watcher must never take the hunt down
                failures += 1
                self.error = f"{type(exc).__name__}: {exc}"
                if failures == 1:
                    logger.exception("The hunt's camera watcher failed; retrying")
                self._stop.wait(min(2.0, 0.1 * failures))

    def step(self) -> Sighting | None:
        """Grab, detect and publish one frame (the thread calls this in a loop)."""
        moving = bool(self._moving())
        pan = int(self._pan())
        odometry = self._odometry()
        captured = self._clock()
        frame = self._grab()
        if frame is None:
            self._stop.wait(0.05)
            return None
        self.error = None
        detections = list(self._detect(frame))
        look = cat_view_from(detections, pan, frame.shape[1], frame.shape[0])
        sighting = Sighting(captured, pan, odometry, look, moving, tuple(detections))
        self._frames += 1
        elapsed = self._clock() - self._started_at
        if elapsed > 0:
            self._fps = self._frames / elapsed
        with self._condition:
            self._latest = sighting
            if look.cat is not None:
                self._spotted = sighting
                if self._interrupts:
                    self.interrupt.set()
            if not moving:
                self._still = (sighting, frame)
            if self._collecting is not None:
                key = int(round(pan / PANORAMA_BIN_DEGREES) * PANORAMA_BIN_DEGREES)
                if key not in self._collecting:
                    encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if encoded:
                        self._collecting[key] = jpeg.tobytes()
            self._condition.notify_all()
        if self._on_sighting is not None:
            self._on_sighting(sighting)
        self._tune(frame, moving)
        if self._publish is not None and (look.cat is not None or captured - self._last_publish >= PUBLISH_INTERVAL_SECONDS):
            self._last_publish = captured
            self._publish(frame, detections)
        return sighting

    def _tune(self, frame: np.ndarray, moving: bool) -> None:
        brightness = brightness_of(frame)
        self.exposure.note_sharpness(sharpness_of(frame), moving, brightness)
        settings = self.exposure.update(brightness, moving)
        now = self._clock()
        if settings is None and now - self._last_exposure_sent >= EXPOSURE_RESEND_SECONDS:
            settings = self.exposure.settings()
        if settings is not None and self._set_exposure is not None:
            self._last_exposure_sent = now
            self._set_exposure(settings)

    # -- reading --------------------------------------------------------

    def latest(self) -> Sighting | None:
        with self._condition:
            return self._latest

    def wait_for(self, after: float, timeout: float) -> Sighting | None:
        """The first frame captured after `after` (time.monotonic()), waiting
        up to `timeout` seconds for it."""
        deadline = self._clock() + timeout
        with self._condition:
            while self._latest is None or self._latest.at < after:
                remaining = deadline - self._clock()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._condition.wait(min(remaining, 0.1))
            return self._latest

    def still_frame(self) -> tuple[Sighting, np.ndarray] | None:
        """The latest frame taken while the car and head were still, with what it showed."""
        with self._condition:
            return self._still

    def set_interrupts(self, enabled: bool) -> None:
        """While on, a cat sighting sets `interrupt` (and a stale one is forgotten)."""
        with self._condition:
            self._interrupts = enabled
            if enabled:
                self._spotted = None
            self.interrupt.clear()

    def spotted(self) -> Sighting | None:
        """The latest cat sighting since interrupts were last turned on or cleared."""
        with self._condition:
            return self._spotted

    def clear_spotted(self) -> None:
        with self._condition:
            self._spotted = None
            self.interrupt.clear()

    def collect(self, enabled: bool) -> dict[int, bytes]:
        """Start keeping one picture per pan step (for a panorama), or stop
        and hand back what was kept."""
        with self._condition:
            kept = self._collecting or {}
            self._collecting = {} if enabled else None
            return kept

    def status(self) -> dict[str, object]:
        latest = self.latest()
        return {
            "fps": round(self._fps, 1),
            "exposure_ms": round(self.exposure.exposure_us / 1000, 1),
            "gain": round(self.exposure.gain, 1),
            "moving_cap_ms": round(self.exposure.moving_cap_us / 1000, 1),
            "last_frame_age_seconds": None if latest is None else round(self._clock() - latest.at, 2),
            "error": self.error,
        }

