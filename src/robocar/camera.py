import logging
import threading
import time
from importlib import import_module
from typing import Any

import cv2

from robocar.detector import ObjectDetector

logger = logging.getLogger(__name__)
# Frames thrown away after opening the camera or changing its exposure, so
# auto-exposure (or the new controls) has settled before a picture is kept.
# Without this, a camera opened just for one snapshot returned its first,
# badly underexposed frame.
OPEN_SETTLE_FRAMES = 8
EXPOSURE_SETTLE_FRAMES = 4
# Let auto-exposure use frames up to 100 ms long in dim rooms.
LONGEST_FRAME_US = 100_000


class CameraService:
    def __init__(
        self,
        backend: str,
        camera_index: int,
        width: int,
        height: int,
        detector: ObjectDetector,
        enabled: bool = False,
    ) -> None:
        self._backend = backend
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._detector = detector
        self._camera: Any = None
        self._lock = threading.Lock()
        self._enabled = enabled
        self._held = False
        self._exposure: dict[str, float] = {}
        self._settle_frames = 0
        self.error: str | None = None
        self.last_detections: list[dict[str, object]] = []

    @property
    def detector_status(self) -> str:
        return self._detector.status

    def _open(self) -> None:
        if self._camera is not None:
            return
        if self._backend == "picamera2":
            picamera2 = import_module("picamera2")
            camera = picamera2.Picamera2()
            camera.configure(
                camera.create_video_configuration(
                    main={"format": "BGR888", "size": (self._width, self._height)}
                )
            )
            camera.start()
            self._camera = camera
            self._apply_exposure()
            self._settle_frames = OPEN_SETTLE_FRAMES
        else:
            camera = cv2.VideoCapture(self._camera_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if not camera.isOpened():
                camera.release()
                raise RuntimeError(f"Could not open camera index {self._camera_index}")
            self._camera = camera
            self._apply_exposure()
            self._settle_frames = OPEN_SETTLE_FRAMES
        self.error = None

    def _apply_exposure(self) -> None:
        """Send the requested exposure to an open camera (best effort for OpenCV)."""
        settings = self._exposure
        try:
            if self._backend == "picamera2":
                if "exposure_us" in settings:
                    exposure = int(settings["exposure_us"])
                    controls = {
                        "AeEnable": False,
                        "ExposureTime": exposure,
                        "AnalogueGain": float(settings.get("gain", 1.0)),
                        "FrameDurationLimits": (exposure + 1000, max(exposure + 1000, LONGEST_FRAME_US)),
                    }
                else:
                    controls = {
                        "AeEnable": True,
                        "ExposureValue": float(settings.get("ev", 0.0)),
                        "FrameDurationLimits": (33_333, LONGEST_FRAME_US),
                    }
                self._camera.set_controls(controls)
            elif "exposure_us" in settings:
                # V4L2 manual exposure, in units of 100 microseconds.
                self._camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                self._camera.set(cv2.CAP_PROP_EXPOSURE, float(settings["exposure_us"]) / 100)
            else:
                self._camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        except Exception as exc:  # cameras differ in which controls they accept
            logger.warning("Camera did not accept exposure settings %s: %s", settings, exc)

    def set_exposure(self, settings: dict[str, float]) -> None:
        """Use these exposure settings for the next snapshots: {"ev": ...} keeps
        auto-exposure with compensation; {"exposure_us": ..., "gain": ...} is manual."""
        with self._lock:
            if dict(settings) == self._exposure:
                return
            self._exposure = dict(settings)
            if self._camera is not None:
                self._apply_exposure()
                self._settle_frames = max(self._settle_frames, EXPOSURE_SETTLE_FRAMES)

    def hold(self, held: bool) -> None:
        """Keep the camera open between snapshots (during mapping) so exposure stays settled."""
        with self._lock:
            self._held = held
            if not held and not self._enabled:
                self._release()

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            if not self._enabled:
                return None
            try:
                frame = self._capture_frame()
                frame, self.last_detections = self._detector.annotate(frame)
                encoded, jpeg = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise RuntimeError("OpenCV could not encode a camera frame")
                self.error = None
                return jpeg.tobytes()
            except (ImportError, OSError, RuntimeError) as exc:
                self.error = str(exc)
                logger.error("Camera or vision pipeline failed: %s", exc)
                self._release()
                return None

    def capture_snapshot(self) -> bytes | None:
        """Capture a still without enabling the annotated dashboard stream."""
        with self._lock:
            try:
                frame = self._capture_frame()
                while self._settle_frames > 0:
                    self._settle_frames -= 1
                    frame = self._capture_frame()
                encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not encoded:
                    raise RuntimeError("OpenCV could not encode a mapping snapshot")
                self.error = None
                return jpeg.tobytes()
            except (ImportError, OSError, RuntimeError) as exc:
                self.error = str(exc)
                logger.error("Mapping camera capture failed: %s", exc)
                self._release()
                return None
            finally:
                if not self._enabled and not self._held:
                    self._release()

    def _capture_frame(self):
        self._open()
        if self._backend == "picamera2":
            return self._camera.capture_array()
        ok, frame = self._camera.read()
        if not ok:
            raise RuntimeError("Camera stopped returning frames")
        return frame

    def _release(self) -> None:
        if self._camera is not None:
            if self._backend == "picamera2":
                self._camera.stop()
                self._camera.close()
            else:
                self._camera.release()
            self._camera = None

    def close(self) -> None:
        with self._lock:
            self._enabled = False
            self._release()

    def set_enabled(self, enabled: bool) -> bool:
        if not isinstance(enabled, bool):
            raise ValueError("Camera enabled state must be a boolean")
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self.last_detections = []
                self._release()
        return self._enabled

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "backend": self._backend,
                "enabled": self._enabled,
                "connected": self._camera is not None,
                "error": self.error,
                "detector": self.detector_status,
                "detections": self.last_detections,
            }

    def stream(self):
        while True:
            jpeg = self.get_jpeg()
            if jpeg is None:
                time.sleep(0.25 if not self.status()["enabled"] else 0.5)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
