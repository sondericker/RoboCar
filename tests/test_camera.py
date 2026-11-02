import unittest
import unittest.mock

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


class CameraHoldAndSettleTests(unittest.TestCase):
    def make_camera(self):
        camera = CameraService("opencv", 0, 640, 480, DetectorStub())
        device = unittest.mock.Mock()
        camera._camera = device
        frames = []

        def capture_frame():
            frames.append(1)
            return np.zeros((16, 16, 3), dtype=np.uint8)

        camera._capture_frame = capture_frame
        return camera, device, frames

    def test_held_camera_stays_open_between_snapshots(self):
        camera, device, _ = self.make_camera()
        camera.hold(True)
        self.assertIsNotNone(camera.capture_snapshot())
        self.assertIs(camera._camera, device)
        camera.hold(False)
        device.release.assert_called_once()
        self.assertIsNone(camera._camera)

    def test_frames_are_discarded_until_exposure_settles(self):
        camera, _, frames = self.make_camera()
        camera.hold(True)
        camera.set_exposure({"ev": 2.0})
        self.assertIsNotNone(camera.capture_snapshot())
        # One frame plus the settling frames after the exposure change.
        self.assertEqual(len(frames), 1 + 4)
        frames.clear()
        camera.set_exposure({"ev": 2.0})  # unchanged: no extra settling
        camera.capture_snapshot()
        self.assertEqual(len(frames), 1)
        camera.close()


IMX477_MODES = [
    {"size": (1332, 990), "fps": 120.5, "crop_limits": (704, 544, 2664, 1980), "bit_depth": 10},
    {"size": (2028, 1080), "fps": 74.74, "crop_limits": (8, 456, 4056, 2160), "bit_depth": 10},
    {"size": (2028, 1520), "fps": 53.77, "crop_limits": (8, 16, 4056, 3040), "bit_depth": 10},
    {"size": (4056, 3040), "fps": 14.0, "crop_limits": (8, 16, 4056, 3040), "bit_depth": 10},
    {"size": (2028, 1520), "fps": 45.19, "crop_limits": (8, 16, 4056, 3040), "bit_depth": 12},
    {"size": (2028, 1520), "fps": 66.38, "crop_limits": (8, 16, 4056, 3040), "bit_depth": 8},
]
OV5647_MODES = [
    {"size": (640, 480), "fps": 62.5, "crop_limits": (16, 0, 2560, 1920), "bit_depth": 10},
    {"size": (1296, 972), "fps": 46.34, "crop_limits": (0, 0, 2592, 1944), "bit_depth": 10},
    {"size": (1920, 1080), "fps": 32.81, "crop_limits": (348, 434, 1928, 1080), "bit_depth": 10},
    {"size": (2592, 1944), "fps": 15.63, "crop_limits": (0, 0, 2592, 1944), "bit_depth": 10},
]


class SensorModeTests(unittest.TestCase):
    def test_the_hq_camera_reads_its_whole_view_binned_and_fast(self):
        from robocar.camera import full_view_mode

        mode = full_view_mode(IMX477_MODES)
        self.assertEqual(mode["size"], (2028, 1520))
        self.assertEqual(mode["bit_depth"], 10)

    def test_the_ov5647_keeps_its_nearly_full_width_small_mode(self):
        from robocar.camera import full_view_mode

        self.assertEqual(full_view_mode(OV5647_MODES)["size"], (640, 480))

    def test_no_modes_no_choice(self):
        from robocar.camera import full_view_mode

        self.assertIsNone(full_view_mode([]))

    def test_the_chosen_mode_is_asked_for_and_reported(self):
        import sys

        configured = {}

        class FakePicamera2:
            sensor_modes = IMX477_MODES

            def __init__(self, index):
                configured["index"] = index

            def create_video_configuration(self, **kwargs):
                return kwargs

            def configure(self, configuration):
                configured["configuration"] = configuration

            def start(self):
                pass

            def set_controls(self, controls):
                pass

            def capture_array(self):
                return np.zeros((480, 640, 3), np.uint8)

            def stop(self):
                pass

            def close(self):
                pass

        module = unittest.mock.MagicMock(Picamera2=FakePicamera2)
        with unittest.mock.patch.dict(sys.modules, {"picamera2": module}):
            camera = CameraService("picamera2", 0, 640, 480, DetectorStub())
            self.assertIsNotNone(camera.capture_frame())
        self.assertEqual(configured["configuration"]["sensor"], {"output_size": (2028, 1520), "bit_depth": 10})
        self.assertEqual(camera.status()["sensor_mode"], {"size": [2028, 1520], "fps": 53.8})


class CameraRoleTests(unittest.TestCase):
    """Which camera does which job, by the cameras connected."""

    def roles(self, connected):
        from robocar.config import Config
        from robocar.web import _open_cameras

        with unittest.mock.patch("robocar.web.connected_models", return_value=connected):
            forward, head, on_head = _open_cameras(Config(hardware_enabled=True), DetectorStub())
        return forward, head, on_head

    def test_both_cameras_split_the_jobs(self):
        forward, head, on_head = self.roles(["imx477", "ov5647"])
        self.assertEqual(forward.model, "imx477")
        self.assertEqual(head.model, "ov5647")
        self.assertTrue(on_head)

    def test_one_camera_does_both_jobs(self):
        forward, head, on_head = self.roles(["imx477"])
        self.assertIs(forward, head)
        self.assertEqual(forward.model, "imx477")
        self.assertFalse(on_head)  # the HQ camera is fixed facing forward
        forward, head, on_head = self.roles(["ov5647"])
        self.assertIs(forward, head)
        self.assertTrue(on_head)

    def test_outside_hardware_mode_no_real_camera_is_looked_up(self):
        from robocar.config import Config
        from robocar.web import _open_cameras

        with unittest.mock.patch("robocar.web.connected_models", side_effect=AssertionError("looked up")):
            forward, head, _ = _open_cameras(Config(hardware_enabled=False), DetectorStub())
        self.assertIs(forward, head)
        self.assertIsNone(forward.model)

    def test_no_named_camera_falls_back_to_the_camera_number(self):
        forward, head, on_head = self.roles([])
        self.assertIs(forward, head)
        self.assertIsNone(forward.model)

    def test_a_camera_is_opened_by_its_sensor_name(self):
        import sys

        opened = {}

        class FakePicamera2:
            sensor_modes = []

            @staticmethod
            def global_camera_info():
                return [{"Model": "ov5647", "Num": 0}, {"Model": "imx477", "Num": 1}]

            def __init__(self, number):
                opened["number"] = number

            def create_video_configuration(self, **kwargs):
                return kwargs

            def configure(self, configuration):
                pass

            def start(self):
                pass

            def set_controls(self, controls):
                pass

            def capture_array(self):
                return np.zeros((480, 640, 3), np.uint8)

            def stop(self):
                pass

            def close(self):
                pass

        module = unittest.mock.MagicMock(Picamera2=FakePicamera2)
        with unittest.mock.patch.dict(sys.modules, {"picamera2": module}):
            self.assertIsNotNone(CameraService("picamera2", 0, 640, 480, DetectorStub(), model="imx477").capture_frame())
            self.assertEqual(opened["number"], 1)
            missing = CameraService("picamera2", 0, 640, 480, DetectorStub(), model="imx219")
            self.assertIsNone(missing.capture_frame())
            self.assertIn("No imx219 camera", missing.status()["error"])
