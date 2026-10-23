import unittest

import numpy as np

from robocar.detector import COCO_LABELS, ObjectDetector


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


if __name__ == "__main__":
    unittest.main()
