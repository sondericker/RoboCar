import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from robocar import detector as detector_module
from robocar.detector import COCO_LABELS, ObjectDetector, decode_hailo_nms, default_hailo_model, letterbox


class DetectorTests(unittest.TestCase):
    def test_disabled_detector_leaves_frame_unchanged(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        detector = ObjectDetector(None, None, 0.35, 0.45)
        annotated, detections = detector.annotate(frame)
        self.assertIs(annotated, frame)
        self.assertEqual(detections, [])
        np.testing.assert_array_equal(annotated, original)

    def test_yolo_output_boxes_are_scaled_and_class_labeled(self):
        detector = ObjectDetector(None, None, 0.35, 0.45)
        predictions = np.zeros((len(COCO_LABELS) + 4, 2), dtype=np.float32)
        predictions[:4, 0] = (320, 320, 100, 100)
        predictions[4, 0] = 0.9
        predictions[:4, 1] = (160, 240, 80, 60)
        predictions[19, 1] = 0.8

        results = detector._parse_predictions(predictions, 640, 480, 1.0, 0, 80)

        self.assertEqual(
            results,
            [
                {"label": "person", "confidence": 0.9, "box": [270, 190, 100, 100]},
                {"label": "cat", "confidence": 0.8, "box": [120, 130, 80, 60]},
            ],
        )


class FakeBackend:
    def __init__(self, name, detections=None, fail=False):
        self.name = name
        self.detections = detections or []
        self.fail = fail
        self.closed = False

    def detect(self, frame):
        if self.fail:
            raise RuntimeError(f"{self.name} broke")
        return list(self.detections)

    def close(self):
        self.closed = True


class HailoDecodingTests(unittest.TestCase):
    def test_on_chip_nms_boxes_map_back_through_the_letterbox(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = letterbox(frame, 640, 640)
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertEqual((scale, pad_x, pad_y), (1.0, 0, 80))
        per_class = [[] for _ in COCO_LABELS]
        # A person filling the middle of the 640x480 frame, as fractions of the 640x640 input.
        per_class[0] = np.array([[(80 + 120) / 640, 160 / 640, (80 + 360) / 640, 480 / 640, 0.92]])
        per_class[56] = np.array([[0.5, 0.1, 0.6, 0.2, 0.2]])  # a chair below the confidence bar
        detections = decode_hailo_nms(per_class, list(COCO_LABELS), 0.35, 640, 640, scale, pad_x, pad_y, 640, 480)
        self.assertEqual(detections, [{"label": "person", "confidence": 0.92, "box": [160, 120, 320, 240]}])


class BackendChoiceTests(unittest.TestCase):
    def test_the_hat_is_used_when_present(self):
        hailo = FakeBackend("hailo", [{"label": "cat", "confidence": 0.9, "box": [1, 2, 3, 4]}])
        with patch.object(detector_module, "HailoBackend", lambda *args: hailo):
            detector = ObjectDetector(None, None, 0.35, 0.45, hailo_model=Path("model.hef"))
        self.assertEqual(detector.detect(np.zeros((480, 640, 3), np.uint8))[0]["label"], "cat")
        self.assertIn("hailo", detector.status)

    def test_without_the_hat_the_cpu_model_takes_over(self):
        def missing(*args):
            raise RuntimeError("no Hailo device")

        cpu = FakeBackend("cpu", [{"label": "dog", "confidence": 0.8, "box": [1, 2, 3, 4]}])
        with patch.object(detector_module, "HailoBackend", missing), patch.object(detector_module, "OnnxBackend", lambda *args: cpu):
            detector = ObjectDetector(Path("model.onnx"), None, 0.35, 0.45, hailo_model=Path("model.hef"))
        self.assertEqual(detector.detect(np.zeros((480, 640, 3), np.uint8))[0]["label"], "dog")
        self.assertIn("AI HAT+ unavailable", detector.status)
        self.assertFalse(detector.info()["accelerator"])

    def test_a_hat_failure_falls_back_then_retries_the_hat(self):
        hailo = FakeBackend("hailo", fail=True)
        cpu = FakeBackend("cpu", [{"label": "dog", "confidence": 0.8, "box": [1, 2, 3, 4]}])
        healthy = FakeBackend("hailo again", [{"label": "cat", "confidence": 0.9, "box": [1, 2, 3, 4]}])
        backends = iter([hailo, healthy])
        frame = np.zeros((480, 640, 3), np.uint8)

        class Hailo(FakeBackend):
            def __new__(cls, *args):
                return next(backends)

        with (
            patch.object(detector_module, "HailoBackend", Hailo),
            patch.object(detector_module, "OnnxBackend", lambda *args: cpu),
            patch.object(detector_module, "HAILO_RETRY_SECONDS", 0.0),
        ):
            detector = ObjectDetector(Path("model.onnx"), None, 0.35, 0.45, hailo_model=Path("model.hef"))
            self.assertEqual(detector.detect(frame), [])  # the frame that broke it
            self.assertTrue(hailo.closed)
            # Straight after, the HAT is retried (retry delay 0 here) and works again.
            self.assertEqual(detector.detect(frame)[0]["label"], "cat")

    def test_nothing_available_means_no_detections_not_a_crash(self):
        def missing(*args):
            raise RuntimeError("no Hailo device")

        with patch.object(detector_module, "HailoBackend", missing):
            detector = ObjectDetector(None, None, 0.35, 0.45, hailo_model=Path("model.hef"))
        self.assertFalse(detector.available)
        self.assertEqual(detector.annotate(np.zeros((480, 640, 3), np.uint8))[1], [])
        self.assertIn("unavailable", detector.status)


class HatHardwareTests(unittest.TestCase):
    """Runs on a Pi with the AI HAT+ and HailoRT installed; skipped elsewhere."""

    def test_the_hat_finds_the_people_and_the_bus(self):
        try:
            import hailo_platform  # noqa: F401
        except ImportError:
            self.skipTest("HailoRT is not installed in this environment")
        sample = Path.home() / "robot_models" / "bus.jpg"
        model = default_hailo_model()
        if model is None or not sample.is_file() or not Path("/dev/hailo0").exists():
            self.skipTest("no AI HAT+, model or sample picture here")
        detector = ObjectDetector(None, None, 0.35, 0.45, hailo_model=model)
        try:
            if not detector.on_accelerator and "OUT_OF_PHYSICAL_DEVICES" in detector.status:
                self.skipTest("the AI HAT+ is in use by another program (the RoboCar service?)")
            self.assertTrue(detector.on_accelerator, detector.status)
            frame = cv2.resize(cv2.imread(str(sample)), (640, 480))
            labels = [detection["label"] for detection in detector.detect(frame)]
            self.assertIn("bus", labels)
            self.assertGreaterEqual(labels.count("person"), 3)
        finally:
            detector.close()


if __name__ == "__main__":
    unittest.main()
