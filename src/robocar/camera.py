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
# Sensor modes reading at least this share of the widest one's area count
# as showing the whole view; of those, the smallest that still runs this fast.
FULL_VIEW_SHARE = 0.95
MIN_SENSOR_FPS = 30.0
PREFERRED_BIT_DEPTH = 10


def connected_models(backend: str) -> list[str]:
    """Sensor names of the cameras connected (lower case, in the order
    Picamera2 numbers them); [] when they cannot be listed."""
    if backend != "picamera2":
        return []
    try:
        info = import_module("picamera2").Picamera2.global_camera_info()
    except Exception as exc:  # no picamera2, no libcamera: no camera list
        logger.info("Could not list cameras: %s", exc)
        return []
    return [str(entry.get("Model", "")).lower() for entry in info]


def full_view_mode(modes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The sensor mode to use: the whole field of view (binned, not cropped),
    at a usable frame rate, as small as possible so frames come fast. On the
    HQ camera (IMX477) that is 2028x1520, about 54 frames a second; the
    OV5647's 640x480 mode also reads nearly its full width."""
    usable = [mode for mode in modes if mode.get("crop_limits") and mode.get("size")]
    if not usable:
        return None
    area = lambda mode: mode["crop_limits"][2] * mode["crop_limits"][3]
    widest = max(area(mode) for mode in usable)
    full = [mode for mode in usable if area(mode) >= FULL_VIEW_SHARE * widest]
    fast = [mode for mode in full if float(mode.get("fps", 0.0)) >= MIN_SENSOR_FPS] or full
    smallest = min(mode["size"][0] * mode["size"][1] for mode in fast)
    candidates = [mode for mode in fast if mode["size"][0] * mode["size"][1] == smallest]
    preferred = [mode for mode in candidates if mode.get("bit_depth") == PREFERRED_BIT_DEPTH]
    return (preferred or sorted(candidates, key=lambda mode: -float(mode.get("fps", 0.0))))[0]


class CameraService:
    def __init__(
        self,
        backend: str,
        camera_index: int,
        width: int,
        height: int,
        detector: ObjectDetector,
        enabled: bool = False,
        model: str | None = None,
    ) -> None:
        """With `model` (a sensor name such as "imx477"), that camera is used
        whichever number it has; otherwise camera number `camera_index`."""
        self._backend = backend
        self._camera_index = camera_index
        self.model = model.lower() if model else None
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
        self.sensor_mode: dict[str, object] | None = None

    @property
    def detector_status(self) -> str:
        return self._detector.status

    def _open(self) -> None:
        if self._camera is not None:
            return
        if self._backend == "picamera2":
            picamera2 = import_module("picamera2")
            camera = picamera2.Picamera2(self._picamera_number(picamera2))
            # Picamera2's "RGB888" is stored B, G, R: the order OpenCV
            # expects. ("BGR888" swapped red and blue in every frame.)
            main = {"format": "RGB888", "size": (self._width, self._height)}
            mode = full_view_mode(list(getattr(camera, "sensor_modes", []) or []))
            configuration = None
            if mode is not None:
                try:
                    configuration = camera.create_video_configuration(
                        main=main, sensor={"output_size": tuple(mode["size"]), "bit_depth": mode["bit_depth"]}
                    )
                except (TypeError, KeyError):  # an older picamera2 without sensor selection
                    configuration = None
            camera.configure(configuration or camera.create_video_configuration(main=main))
            self.sensor_mode = None if mode is None else {"size": list(mode["size"]), "fps": round(float(mode["fps"]), 1)}
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

    def _picamera_number(self, picamera2: Any) -> int:
        if self.model is None:
            return self._camera_index
        for number, entry in enumerate(picamera2.Picamera2.global_camera_info()):
            if self.model in str(entry.get("Model", "")).lower():
                return int(entry.get("Num", number))
        raise RuntimeError(f"No {self.model} camera is connected")

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

    def capture_frame(self):
        """A raw BGR frame for continuous processing (no settling, no JPEG)."""
        with self._lock:
            try:
                frame = self._capture_frame()
                self.error = None
                return frame
            except (ImportError, OSError, RuntimeError) as exc:
                self.error = str(exc)
                logger.error("Camera frame capture failed: %s", exc)
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
                "model": self.model,
                "enabled": self._enabled,
                "connected": self._camera is not None,
                "sensor_mode": self.sensor_mode,
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
