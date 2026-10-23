import unittest

import numpy as np

from robocar.camera import CameraService


class DetectorStub:
    status = "stub detector"

    def annotate(self, frame):
        raise AssertionError("Detector should not run while the camera is paused")


class CameraServiceTests(unittest.TestCase):
    def test_paused_camera_does_not_open_or_process_frames(self):
        camera = CameraService("opencv", 0, 640, 480, DetectorStub())
        camera._open = lambda: self.fail("Camera should not open while paused")
        camera.set_enabled(False)
        try:
            self.assertIsNone(camera.get_jpeg())
            self.assertFalse(camera.status()["enabled"])
            self.assertFalse(camera.status()["connected"])
        finally:
            camera.close()

    def test_camera_starts_paused_by_default(self):
        camera = CameraService("opencv", 0, 640, 480, DetectorStub())
        try:
            self.assertFalse(camera.status()["enabled"])
            self.assertIsNone(camera.get_jpeg())
        finally:
            camera.close()

    def test_mapping_snapshot_skips_detector_and_keeps_camera_paused(self):
        camera = CameraService("opencv", 0, 640, 480, DetectorStub())
        camera._capture_frame = lambda: np.zeros((16, 16, 3), dtype=np.uint8)
        try:
            snapshot = camera.capture_snapshot()
            self.assertIsNotNone(snapshot)
            self.assertTrue(snapshot.startswith(b"\xff\xd8"))
            self.assertFalse(camera.status()["enabled"])
            self.assertFalse(camera.status()["connected"])
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
