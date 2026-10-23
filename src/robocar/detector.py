from pathlib import Path
import shutil
from typing import Any
from urllib.request import urlopen

import cv2
import numpy as np
import onnxruntime as ort

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


class ObjectDetector:
    """YOLO ONNX inference through ONNX Runtime, with OpenCV box rendering."""

    def __init__(
        self,
        model_path: Path | None,
        labels_path: Path | None,
        confidence: float,
        nms_threshold: float,
    ) -> None:
        self._session: Any = None
        self._labels = list(COCO_LABELS)
        self._confidence = confidence
        self._nms_threshold = nms_threshold
        self.status = "Model not configured"
        if model_path is None:
            return
        if not model_path.is_file() and model_path.name == "yolo11n.onnx":
            _download_default_model(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Object-detection model not found: {model_path}")
        if labels_path is not None:
            if not labels_path.is_file():
                raise FileNotFoundError(f"Object-detection labels not found: {labels_path}")
            self._labels = [
                line.strip()
                for line in labels_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not self._labels:
                raise ValueError(f"Object-detection labels file is empty: {labels_path}")

        available = ort.get_available_providers()
        providers = [
            provider
            for provider in ("XnnpackExecutionProvider", "DnnlExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.status = f"Loaded {model_path.name} ({len(self._labels)} labels; {providers[0]})"

    def annotate(self, frame: Any) -> tuple[Any, list[dict[str, object]]]:
        if self._session is None:
            return frame, []

        input_tensor, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        predictions = np.asarray(outputs[0])
        if predictions.ndim == 3:
            predictions = predictions[0]
        detections = self._parse_predictions(predictions, frame.shape[1], frame.shape[0], scale, pad_x, pad_y)
        for detection in detections:
            x, y, box_width, box_height = detection["box"]
            label = str(detection["label"])
            score = float(detection["confidence"])
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (45, 220, 90), 2)
            cv2.putText(
                frame,
                f"{label} {score:.0%}",
                (x, max(y - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (45, 220, 90),
                2,
            )
        return frame, detections

    @staticmethod
    def _preprocess(frame: Any) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (640 - resized_width) // 2, (640 - resized_height) // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_y,
            640 - resized_height - pad_y,
            pad_x,
            640 - resized_width - pad_x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor[None, ...], scale, pad_x, pad_y

    def _parse_predictions(
        self,
        predictions: np.ndarray,
        frame_width: int,
        frame_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[dict[str, object]]:
        if predictions.ndim != 2:
            raise ValueError(f"Expected two-dimensional YOLO output, got {predictions.shape}")
        if predictions.shape[1] == 6:
            rows = predictions
            xyxy_output = True
        elif predictions.shape[0] == len(self._labels) + 4:
            rows = predictions.T
            xyxy_output = False
        elif predictions.shape[1] == len(self._labels) + 4:
            rows = predictions
            xyxy_output = False
        else:
            raise ValueError(
                f"Unsupported YOLO output shape {predictions.shape} for {len(self._labels)} labels"
            )

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
            if score < self._confidence or not 0 <= class_id < len(self._labels):
                continue
            x1 = round((left - pad_x) / scale)
            y1 = round((top - pad_y) / scale)
            x2 = round((right - pad_x) / scale)
            y2 = round((bottom - pad_y) / scale)
            x1, x2 = max(0, min(frame_width, x1)), max(0, min(frame_width, x2))
            y1, y2 = max(0, min(frame_height, y1)), max(0, min(frame_height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            by_class.setdefault(class_id, []).append(([x1, y1, x2 - x1, y2 - y1], score))

        detections: list[dict[str, object]] = []
        for class_id, candidates in by_class.items():
            indices = cv2.dnn.NMSBoxes(
                [box for box, _ in candidates],
                [score for _, score in candidates],
                self._confidence,
                self._nms_threshold,
            )
            for raw_index in indices:
                box, score = candidates[int(raw_index)]
                detections.append(
                    {"label": self._labels[class_id], "confidence": round(score, 3), "box": box}
                )
        return detections
