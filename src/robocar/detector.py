"""Object detection: YOLO on the Raspberry Pi AI HAT+ (Hailo), or on the CPU.

The Hailo backend runs YOLOv8s with non-maximum suppression done on the
chip, in about 12 ms a frame without loading the CPU. The ONNX backend runs
YOLO11n through ONNX Runtime on the CPU. With both configured the HAT is
used, and the CPU model takes over if the HAT is missing or stops working;
the HAT is retried now and then, so a hiccup does not leave detection on the
CPU for good. Either way callers see the same detections:
{"label", "confidence", "box": [x, y, width, height]} in frame pixels.
"""

import logging
import shutil
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol
from urllib.request import urlopen

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

DEFAULT_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
# Models the hailo-models package installs, best first for each chip.
HAILO_MODEL_CANDIDATES = (
    Path("/usr/share/hailo-models/yolov8s_h8.hef"),
    Path("/usr/share/hailo-models/yolov8s_h8l.hef"),
    Path("/usr/share/hailo-models/yolov6n_h8.hef"),
    Path("/usr/share/hailo-models/yolov6n_h8l.hef"),
)
# After the HAT fails, wait this long before trying it again.
HAILO_RETRY_SECONDS = 30.0
LETTERBOX_FILL = 114


def _explain_hailo_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "OUT_OF_PHYSICAL_DEVICES" in text:
        # HailoRT lets one program use the chip at a time.
        return "in use by another program (HAILO_OUT_OF_PHYSICAL_DEVICES); retrying every 30 s"
    if "No module named" in text:
        return "HailoRT is not installed for this Python"
    return text


def default_hailo_model() -> Path | None:
    return next((path for path in HAILO_MODEL_CANDIDATES if path.is_file()), None)


def _download_default_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(DEFAULT_MODEL_URL, timeout=60) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary_path.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded YOLO model is unexpectedly small")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def letterbox(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """Scale a frame to fit (width, height) keeping its shape, padding the rest grey."""
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    resized_width, resized_height = round(frame_width * scale), round(frame_height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (width - resized_width) // 2, (height - resized_height) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        height - resized_height - pad_y,
        pad_x,
        width - resized_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(LETTERBOX_FILL, LETTERBOX_FILL, LETTERBOX_FILL),
    )
    return padded, scale, pad_x, pad_y


def _to_frame_box(
    left: float, top: float, right: float, bottom: float, scale: float, pad_x: int, pad_y: int, width: int, height: int
) -> list[int] | None:
    """A box in letterboxed-input pixels as [x, y, w, h] in frame pixels, or None if empty."""
    x1 = max(0, min(width, round((left - pad_x) / scale)))
    y1 = max(0, min(height, round((top - pad_y) / scale)))
    x2 = max(0, min(width, round((right - pad_x) / scale)))
    y2 = max(0, min(height, round((bottom - pad_y) / scale)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


class Backend(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[dict[str, object]]: ...

    def close(self) -> None: ...


class HailoBackend:
    """YOLO on the AI HAT+, with non-maximum suppression done on the chip."""

    def __init__(self, hef_path: Path, labels: list[str], confidence: float) -> None:
        from hailo_platform import (  # imported here: only present where HailoRT is installed
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        self._labels = labels
        self._confidence = confidence
        self._stack = ExitStack()
        try:
            hef = HEF(str(hef_path))
            self._vdevice = self._stack.enter_context(VDevice(VDevice.create_params()))
            params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            group = self._vdevice.configure(hef, params)[0]
            self._pipeline = self._stack.enter_context(
                InferVStreams(
                    group,
                    InputVStreamParams.make(group, format_type=FormatType.UINT8),
                    OutputVStreamParams.make(group, format_type=FormatType.FLOAT32),
                )
            )
            self._stack.enter_context(group.activate(group.create_params()))
            info = hef.get_input_vstream_infos()[0]
            self._input_name = info.name
            self._height, self._width = int(info.shape[0]), int(info.shape[1])
            self._output_name = hef.get_output_vstream_infos()[0].name
        except Exception:
            self._stack.close()
            raise
        self.name = f"Hailo AI HAT+ ({hef_path.stem})"

    def detect(self, frame: np.ndarray) -> list[dict[str, object]]:
        height, width = frame.shape[:2]
        padded, scale, pad_x, pad_y = letterbox(frame, self._width, self._height)
        rgb = np.ascontiguousarray(padded[:, :, ::-1])
        output = self._pipeline.infer({self._input_name: rgb[None]})[self._output_name][0]
        return decode_hailo_nms(output, self._labels, self._confidence, self._width, self._height, scale, pad_x, pad_y, width, height)

    def close(self) -> None:
        self._stack.close()


def decode_hailo_nms(
    per_class: list,
    labels: list[str],
    confidence: float,
    input_width: int,
    input_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, object]]:
    """Hailo "NMS by class" output: for each class, rows of [y_min, x_min,
    y_max, x_max, score] as fractions of the (letterboxed) input."""
    detections = []
    for class_id, boxes in enumerate(per_class):
        if class_id >= len(labels):
            break
        for y_min, x_min, y_max, x_max, score in np.asarray(boxes, dtype=float).reshape(-1, 5):
            if score < confidence:
                continue
            box = _to_frame_box(
                x_min * input_width,
                y_min * input_height,
                x_max * input_width,
                y_max * input_height,
                scale,
                pad_x,
                pad_y,
                frame_width,
                frame_height,
            )
            if box is not None:
                detections.append({"label": labels[class_id], "confidence": round(float(score), 3), "box": box})
    detections.sort(key=lambda item: -float(item["confidence"]))
    return detections


class OnnxBackend:
    """YOLO ONNX inference on the CPU through ONNX Runtime."""

    def __init__(self, model_path: Path, labels: list[str], confidence: float, nms_threshold: float) -> None:
        import onnxruntime as ort

        if not model_path.is_file() and model_path.name == "yolo11n.onnx":
            _download_default_model(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Object-detection model not found: {model_path}")
        self._labels = labels
        self._confidence = confidence
        self._nms_threshold = nms_threshold
        available = ort.get_available_providers()
        providers = [
            provider
            for provider in ("XnnpackExecutionProvider", "DnnlExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.name = f"CPU ({model_path.name}, {providers[0]})"

    def detect(self, frame: np.ndarray) -> list[dict[str, object]]:
        padded, scale, pad_x, pad_y = letterbox(frame, 640, 640)
        tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        predictions = np.asarray(self._session.run(None, {self._input_name: tensor})[0])
        if predictions.ndim == 3:
            predictions = predictions[0]
        return parse_yolo_predictions(
            predictions, self._labels, self._confidence, self._nms_threshold, frame.shape[1], frame.shape[0], scale, pad_x, pad_y
        )

    def close(self) -> None:
        self._session = None


def parse_yolo_predictions(
    predictions: np.ndarray,
    labels: list[str],
    confidence: float,
    nms_threshold: float,
    frame_width: int,
    frame_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> list[dict[str, object]]:
    """Raw YOLO output (either orientation, or [x1, y1, x2, y2, score, class]
    rows) to detections, with per-class non-maximum suppression."""
    if predictions.ndim != 2:
        raise ValueError(f"Expected two-dimensional YOLO output, got {predictions.shape}")
    if predictions.shape[1] == 6:
        rows = predictions
        xyxy_output = True
    elif predictions.shape[0] == len(labels) + 4:
        rows = predictions.T
        xyxy_output = False
    elif predictions.shape[1] == len(labels) + 4:
        rows = predictions
        xyxy_output = False
    else:
        raise ValueError(f"Unsupported YOLO output shape {predictions.shape} for {len(labels)} labels")

    by_class: dict[int, list[tuple[list[int], float]]] = {}
    for row in rows:
        if xyxy_output:
            score = float(row[4])
            class_id = int(row[5])
            left, top, right, bottom = (float(value) for value in row[:4])
        else:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            center_x, center_y, box_width, box_height = (float(value) for value in row[:4])
            left, top = center_x - box_width / 2, center_y - box_height / 2
            right, bottom = center_x + box_width / 2, center_y + box_height / 2
        if score < confidence or not 0 <= class_id < len(labels):
            continue
        box = _to_frame_box(left, top, right, bottom, scale, pad_x, pad_y, frame_width, frame_height)
        if box is not None:
            by_class.setdefault(class_id, []).append((box, score))

    detections: list[dict[str, object]] = []
    for class_id, candidates in by_class.items():
        indices = cv2.dnn.NMSBoxes(
            [box for box, _ in candidates], [score for _, score in candidates], confidence, nms_threshold
        )
        for raw_index in indices:
            box, score = candidates[int(raw_index)]
            detections.append({"label": labels[class_id], "confidence": round(score, 3), "box": box})
    return detections


class ObjectDetector:
    """Detects objects with the best backend available, falling back and
    recovering on its own. Safe to call from several threads."""

    def __init__(
        self,
        model_path: Path | None,
        labels_path: Path | None,
        confidence: float,
        nms_threshold: float,
        hailo_model: Path | None = None,
    ) -> None:
        self._labels = list(COCO_LABELS)
        if labels_path is not None:
            if not labels_path.is_file():
                raise FileNotFoundError(f"Object-detection labels not found: {labels_path}")
            self._labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not self._labels:
                raise ValueError(f"Object-detection labels file is empty: {labels_path}")
        self._confidence = confidence
        self._nms_threshold = nms_threshold
        self._model_path = model_path
        self._hailo_model = hailo_model
        self._lock = threading.Lock()
        self._backend: Backend | None = None
        self._on_hailo = False  # whether self._backend is the HAT
        self._cpu: Backend | None = None
        self._cpu_failed: str | None = None
        self._hailo_failed_at: float | None = None
        self._hailo_error: str | None = None
        self._latency_ms: float | None = None
        self._frames = 0
        self.status = "Model not configured"
        if hailo_model is None and model_path is None:
            return
        with self._lock:
            self._choose_backend_locked()

    # -- backends -----------------------------------------------------------

    def _choose_backend_locked(self) -> None:
        if self._hailo_model is not None and self._hailo_failed_at is None:
            try:
                self._backend = HailoBackend(self._hailo_model, self._labels, self._confidence)
                self._on_hailo = True
                self._hailo_error = None
                self._update_status_locked()
                logger.info("Object detection on %s", self._backend.name)
                return
            except Exception as exc:  # no HAT, driver trouble, device busy...
                self._hailo_failed_at = time.monotonic()
                self._hailo_error = _explain_hailo_error(exc)
                logger.warning("AI HAT+ unavailable (%s); using the CPU model", self._hailo_error)
        self._backend = self._cpu_backend_locked()
        self._on_hailo = False
        self._update_status_locked()

    def _cpu_backend_locked(self) -> Backend | None:
        if self._cpu is None and self._cpu_failed is None and self._model_path is not None:
            try:
                self._cpu = OnnxBackend(self._model_path, self._labels, self._confidence, self._nms_threshold)
            except Exception as exc:
                self._cpu_failed = f"{type(exc).__name__}: {exc}"
                logger.error("CPU object detection unavailable: %s", self._cpu_failed)
        return self._cpu

    def _maybe_retry_hailo_locked(self) -> None:
        """While on the CPU after a HAT failure, try the HAT again now and then."""
        if (
            self._hailo_model is None
            or self._hailo_failed_at is None
            or time.monotonic() - self._hailo_failed_at < HAILO_RETRY_SECONDS
        ):
            return
        self._hailo_failed_at = None
        self._choose_backend_locked()

    def _update_status_locked(self) -> None:
        if self._backend is None:
            reasons = [reason for reason in (self._hailo_error, self._cpu_failed) if reason]
            self.status = "Object detection unavailable" + (f": {'; '.join(reasons)}" if reasons else "")
            return
        timing = f", {self._latency_ms:.0f} ms" if self._latency_ms is not None else ""
        fallback = f" (AI HAT+ unavailable: {self._hailo_error})" if self._hailo_error else ""
        self.status = f"{self._backend.name}, {len(self._labels)} labels{timing}{fallback}"

    # -- detection ----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def on_accelerator(self) -> bool:
        return self._on_hailo

    def detect(self, frame: Any) -> list[dict[str, object]]:
        """Detections in a BGR frame; [] when no backend is available."""
        with self._lock:
            self._maybe_retry_hailo_locked()
            backend = self._backend
            if backend is None:
                return []
            started = time.monotonic()
            try:
                detections = backend.detect(frame)
            except Exception as exc:
                logger.exception("Object detection failed on %s", backend.name)
                if self._on_hailo:
                    self._hailo_failed_at = time.monotonic()
                    self._hailo_error = _explain_hailo_error(exc)
                    try:
                        backend.close()
                    except Exception:
                        logger.debug("Closing the failed AI HAT+ session failed too", exc_info=True)
                    self._backend = self._cpu_backend_locked()
                    self._on_hailo = False
                else:
                    self._cpu_failed = f"{type(exc).__name__}: {exc}"
                    self._cpu = None
                    self._backend = None
                self._update_status_locked()
                return []
            elapsed = (time.monotonic() - started) * 1000
            self._latency_ms = elapsed if self._latency_ms is None else 0.9 * self._latency_ms + 0.1 * elapsed
            self._frames += 1
            if self._frames % 30 == 1:
                self._update_status_locked()
            return detections

    def annotate(self, frame: Any) -> tuple[Any, list[dict[str, object]]]:
        detections = self.detect(frame)
        draw_detections(frame, detections)
        return frame, detections

    def info(self) -> dict[str, object]:
        with self._lock:
            return {
                "backend": self._backend.name if self._backend else None,
                "accelerator": self._on_hailo,
                "latency_ms": None if self._latency_ms is None else round(self._latency_ms, 1),
                "accelerator_error": self._hailo_error,
                "status": self.status,
            }

    def close(self) -> None:
        with self._lock:
            for backend in {id(b): b for b in (self._backend, self._cpu) if b is not None}.values():
                try:
                    backend.close()
                except Exception:
                    logger.debug("Closing a detection backend failed", exc_info=True)
            self._backend = self._cpu = None
            self._on_hailo = False

    # Kept for callers and tests that parse raw ONNX output directly.
    def _parse_predictions(
        self, predictions: np.ndarray, frame_width: int, frame_height: int, scale: float, pad_x: int, pad_y: int
    ) -> list[dict[str, object]]:
        return parse_yolo_predictions(
            predictions, self._labels, self._confidence, self._nms_threshold, frame_width, frame_height, scale, pad_x, pad_y
        )


def draw_detections(frame: np.ndarray, detections: list[dict[str, object]]) -> None:
    for detection in detections:
        x, y, box_width, box_height = detection["box"]
        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (45, 220, 90), 2)
        cv2.putText(
            frame,
            f"{detection['label']} {float(detection['confidence']):.0%}",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (45, 220, 90),
            2,
        )
