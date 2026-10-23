import logging
import threading
import time
from importlib import import_module
from typing import Any

import cv2

from robocar.detector import ObjectDetector

logger = logging.getLogger(__name__)


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
        else:
            camera = cv2.VideoCapture(self._camera_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if not camera.isOpened():
                camera.release()
                raise RuntimeError(f"Could not open camera index {self._camera_index}")
            self._camera = camera
        self.error = None

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
                if not self._enabled:
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
