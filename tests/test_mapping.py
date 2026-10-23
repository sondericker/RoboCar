import json
import math
import tempfile
import time
import stat
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from robocar.drive import (
    DriveController,
    DriveInhibitedError,
    PanTiltController,
    SimulatedDriver,
    SimulatedPanTilt,
)
from robocar.mapping import (
    AUTO_MAX_SEARCH_TURNS,
    AUTO_MAX_TURN_DEGREES,
    SCAN_ANGLES,
    CameraMotionEstimate,
    MappingController,
    MappingStore,
    direction_clearances,
    plan_explore_step,
    estimate_camera_yaw_change,
    forward_clearance,
)
from robocar.localization import relative
from robocar.range_sensor import RangeSensorController
from camera_scene import PanoramaScene
from sonar_room import SonarRoom


class FakeCamera:
    def capture_snapshot(self) -> bytes:
        return b"jpeg"


class FakeRangeSensor:
    def __init__(self, distance: float | None = 80.0) -> None:
        self.distance = distance

    def get_distance(self) -> float | None:
        return self.distance

    def close(self) -> None:
        pass


class RecordingDriver(SimulatedDriver):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[tuple[int, int, int, int]] = []

    def set_motor_model(self, left_front: int, left_rear: int, right_front: int, right_rear: int) -> None:
        super().set_motor_model(left_front, left_rear, right_front, right_rear)
        self.commands.append(self.command)


def fast_sweeps(test: unittest.TestCase) -> None:
    for name, value in (
        ("SCAN_FIRST_SETTLE_SECONDS", 0.01),
        ("SCAN_STEP_SETTLE_SECONDS", 0.002),
        ("SCAN_SAMPLE_INTERVAL_SECONDS", 0.001),
    ("CAMERA_SETTLE_SECONDS", 0.0),
    ):
        patcher = patch(f"robocar.mapping.{name}", value)
        patcher.start()
        test.addCleanup(patcher.stop)


class MappingTests(unittest.TestCase):
    def setUp(self):
        fast_sweeps(self)

    def test_store_commits_observations_across_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "mapping.sqlite3"
            store = MappingStore(database)
            self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
            observation_id = store.save(
                b"image",
                [{"angle_deg": 0, "distance_cm": 80.0, "samples": 3, "error": None}],
                "test-run",
                "explore",
                (1.0, 2.0, 3.0),
            )
            store.close()

            reopened = MappingStore(database)
            try:
                self.assertEqual(reopened.count(), 1)
                self.assertEqual(reopened.image(observation_id), b"image")
                self.assertEqual(
                    reopened.recent()[0]["scan"],
                    [{"angle_deg": 0, "distance_cm": 80.0, "samples": 3, "error": None}],
                )
                self.assertEqual(reopened.recent()[0]["run_id"], "test-run")
                self.assertEqual(reopened.recent()[0]["pose_x_cm"], 1.0)
            finally:
                reopened.close()

    def test_store_reset_removes_observations_and_restarts_ids(self):
        store = MappingStore(Path(":memory:"))
        try:
            store.save(b"image", [], "run", "survey", None)
            store.reset()
            self.assertEqual(store.count(), 0)
            self.assertIsNone(store.latest())
            self.assertEqual(store.save(b"new", [], "run", "survey", None), 1)
        finally:
            store.close()

    def test_store_migrates_existing_survey_database_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "mapping.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute(
                """
                CREATE TABLE observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    image_jpeg BLOB NOT NULL,
                    scan_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO observations (captured_at, image_jpeg, scan_json) VALUES (?, ?, ?)",
                ("2026-01-01T00:00:00+00:00", b"old-image", "[]"),
            )
            connection.commit()
            connection.close()

            store = MappingStore(database)
            try:
                self.assertEqual(store.count(), 1)
                old_observation = store.recent()[0]
                self.assertEqual(old_observation["mode"], "survey")
                self.assertIsNone(old_observation["pose_x_cm"])
                self.assertEqual(store.image(old_observation["id"]), b"old-image")
            finally:
                store.close()

    def test_camera_image_comparison_estimates_direction_of_yaw(self):
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        for y in range(20, 480, 40):
            for x in range(20, 640, 40):
                cv2.circle(image, (x, y), 6, int((x * 3 + y * 7) % 256), 2)
        focal_length = 640 / (2 * np.tan(np.radians(62.2) / 2))
        camera_matrix = np.array(
            [[focal_length, 0, 320], [0, focal_length, 240], [0, 0, 1]],
            dtype=np.float64,
        )
        yaw_radians = np.radians(10)
        rotation = np.array(
            [
                [np.cos(yaw_radians), 0, np.sin(yaw_radians)],
                [0, 1, 0],
                [-np.sin(yaw_radians), 0, np.cos(yaw_radians)],
            ]
        )
        homography = camera_matrix @ rotation @ np.linalg.inv(camera_matrix)
        rotated = cv2.warpPerspective(image, homography, (640, 480))
        _, previous_jpeg = cv2.imencode(".jpg", image)
        _, current_jpeg = cv2.imencode(".jpg", rotated)

        estimated_yaw = estimate_camera_yaw_change(
            [previous_jpeg.tobytes()] * 3,
            [current_jpeg.tobytes()] * 3,
        )

        self.assertIsNotNone(estimated_yaw)
        self.assertGreater(estimated_yaw, 0)
        self.assertAlmostEqual(estimated_yaw, 10.0, delta=2.0)

    def test_camera_yaw_estimate_ignores_sideways_image_translation(self):
        rng = np.random.default_rng(456)
        image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        for y in range(20, 480, 40):
            for x in range(20, 640, 40):
                cv2.circle(image, (x, y), 6, int((x * 3 + y * 7) % 256), 2)
        strafed = cv2.warpAffine(
            image,
            np.float32([[1, 0, -50], [0, 1, 0]]),
            (640, 480),
        )
        _, previous_jpeg = cv2.imencode(".jpg", image)
        _, current_jpeg = cv2.imencode(".jpg", strafed)

        estimated_yaw = estimate_camera_yaw_change(
            [previous_jpeg.tobytes()] * 3,
            [current_jpeg.tobytes()] * 3,
        )

        self.assertIsNotNone(estimated_yaw)
        self.assertAlmostEqual(estimated_yaw, 0.0, delta=1.0)

    def test_agreeing_camera_views_overrule_slipping_wheels(self):
        drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(
            FakeCamera(),
            drive,
            pan_tilt,
            range_sensor,
            MappingStore(Path(":memory:")),
        )
        mapping._run_id = "slip-run"
        odometry = iter([(0.0, 0.0, 0.0), (0.0, 0.0, 12.0), (-6.27, 29.82, 12.0)])
        try:
            with patch.object(drive, "odometry", side_effect=lambda: next(odometry)):
                mapping._capture_observation(mode="explore")
                turned = [((90, 90), CameraMotionEstimate(30.0, None))] * 3
                with patch("robocar.mapping.estimate_view_motion", return_value=turned):
                    mapping._capture_observation(mode="explore")
                self.assertAlmostEqual(mapping._pose[2], 30.0, delta=0.5)

                straight = [((90, 90), CameraMotionEstimate(0.0, 0.0))] * 3
                with patch("robocar.mapping.estimate_view_motion", return_value=straight):
                    mapping._capture_observation(mode="explore")
            x, y, heading = mapping._pose
            self.assertAlmostEqual(heading, 30.0, delta=0.5)
            # About 30 cm ahead along a heading 30 degrees to the left.
            self.assertAlmostEqual(x, -30.48 * np.sin(np.radians(30)), delta=1.5)
            self.assertAlmostEqual(y, 30.48 * np.cos(np.radians(30)), delta=1.5)
            estimate = mapping.observations()[-1]["estimate"]
            self.assertTrue(estimate["visual_used"])
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def test_consecutive_explore_snapshots_save_vision_corrected_pose(self):
        rng = np.random.default_rng(321)
        previous_image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        focal_length = 640 / (2 * np.tan(np.radians(62.2) / 2))
        camera_matrix = np.array(
            [[focal_length, 0, 320], [0, focal_length, 240], [0, 0, 1]],
            dtype=np.float64,
        )
        yaw_radians = np.radians(10)
        rotation = np.array(
            [
                [np.cos(yaw_radians), 0, np.sin(yaw_radians)],
                [0, 1, 0],
                [-np.sin(yaw_radians), 0, np.cos(yaw_radians)],
            ]
        )
        current_image = cv2.warpPerspective(
            previous_image,
            camera_matrix @ rotation @ np.linalg.inv(camera_matrix),
            (640, 480),
        )
        _, previous_jpeg = cv2.imencode(".jpg", previous_image)
        _, current_jpeg = cv2.imencode(".jpg", current_image)

        class ImageSequenceCamera:
            def __init__(self) -> None:
                self.images = iter(
                    (
                        previous_jpeg.tobytes(),
                        previous_jpeg.tobytes(),
                        previous_jpeg.tobytes(),
                        current_jpeg.tobytes(),
                        current_jpeg.tobytes(),
                        current_jpeg.tobytes(),
                    )
                )

            def capture_snapshot(self) -> bytes:
                return next(self.images)

        drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        range_sensor = RangeSensorController(None, physical=False)
        store = MappingStore(Path(":memory:"))
        mapping = MappingController(ImageSequenceCamera(), drive, pan_tilt, range_sensor, store)
        mapping._run_id = "visual-run"
        odometry = iter([(0.0, 0.0, 0.0), (0.0, 0.0, 12.0)])
        try:
            with patch.object(drive, "odometry", side_effect=lambda: next(odometry)):
                mapping._capture_observation(mode="explore")
                mapping._capture_observation(mode="explore")

            observations = mapping.observations()
            self.assertEqual(len(observations), 2)
            self.assertAlmostEqual(observations[1]["heading_deg"], 10.0, delta=2.0)
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def test_stationary_mapping_sweeps_sensor_and_inhibits_drive(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = SimulatedDriver()
            drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=False)
            pan_tilt_driver = SimulatedPanTilt()
            pan_tilt = PanTiltController(pan_tilt_driver, physical=False)
            range_sensor = RangeSensorController(FakeRangeSensor(), physical=True, sample_interval=0.02)
            store = MappingStore(Path(directory) / "mapping.sqlite3")
            mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, store)
            try:
                mapping.start()
                deadline = time.monotonic() + 5
                while mapping.status()["state"] != "capturing" and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(mapping.status()["state"], "capturing")
                with self.assertRaises(DriveInhibitedError):
                    drive.set_command(1, 0, 0)

                deadline = time.monotonic() + 5
                while mapping.status()["observation_count"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                observation = mapping.observations()[0]
                self.assertEqual(
                    [item["angle_deg"] for item in observation["scan"]],
                    list(range(-57, 61, 3)),
                )
                self.assertTrue(all(item["distance_cm"] == 80 for item in observation["scan"]))
                self.assertEqual(pan_tilt_driver.pan, 90)
                self.assertFalse(drive.status()["inhibited"])
            finally:
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()

    def test_autonomous_mode_uses_short_bursts_then_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = SimulatedDriver()
            drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=True)
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
            range_sensor = RangeSensorController(FakeRangeSensor(), physical=True, sample_interval=0.02)
            mapping = MappingController(
                FakeCamera(),
                drive,
                pan_tilt,
                range_sensor,
                MappingStore(Path(directory) / "mapping.sqlite3"),
            )
            try:
                mapping.start_autonomous(attended=True)
                deadline = time.monotonic() + 7
                while mapping.status()["move_count"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertGreaterEqual(mapping.status()["move_count"], 1)
                self.assertIn(driver.command, ((0, 0, 0, 0), (1000, 1000, 1000, 1000)))
            finally:
                mapping.stop()
                self.assertEqual(driver.command, (0, 0, 0, 0))
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()

    def test_move_burst_refreshes_command_past_watchdog_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = RecordingDriver()
            drive = DriveController(driver, max_duty=1000, command_timeout=0.1, physical=True)
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
            range_sensor = RangeSensorController(FakeRangeSensor(), physical=True, sample_interval=0.02)
            mapping = MappingController(
                FakeCamera(),
                drive,
                pan_tilt,
                range_sensor,
                MappingStore(Path(directory) / "mapping.sqlite3"),
            )
            try:
                started_at = time.monotonic()
                self.assertTrue(mapping._move_burst(1.0, 0.25))
                self.assertGreaterEqual(time.monotonic() - started_at, 0.25 + 0.2)
                self.assertTrue(
                    any(
                        all(0 < duty < 1000 for duty in command)
                        for command in driver.commands
                    )
                )
                self.assertEqual(driver.command, (0, 0, 0, 0))
            finally:
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()

    def test_autonomous_mode_backs_up_once_then_stops_when_still_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = RecordingDriver()
            drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=True)
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
            range_sensor = RangeSensorController(
                FakeRangeSensor(distance=20.0),
                physical=True,
                sample_interval=0.02,
            )
            mapping = MappingController(
                FakeCamera(),
                drive,
                pan_tilt,
                range_sensor,
                MappingStore(Path(directory) / "mapping.sqlite3"),
            )
            blocked_scan = [
                {"angle_deg": angle, "distance_cm": 20.0, "samples": 3, "error": None}
                for angle in (-60, -30, 0, 30, 60)
            ]
            try:
                with patch.object(mapping, "_capture_observation", return_value=blocked_scan):
                    mapping.start_autonomous(attended=True)
                    deadline = time.monotonic() + 3
                    while mapping.status()["enabled"] and time.monotonic() < deadline:
                        time.sleep(0.02)
                self.assertEqual(mapping.status()["state"], "blocked")
                self.assertEqual(mapping.status()["move_count"], 1)
                self.assertGreater(max(abs(command[0]) for command in driver.commands), 900)
                self.assertEqual(driver.command, (0, 0, 0, 0))
            finally:
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()

    def test_autonomous_mode_fails_closed_on_missing_forward_echo(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = SimulatedDriver()
            drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=True)
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
            range_sensor = RangeSensorController(
                FakeRangeSensor(distance=None),
                physical=True,
                sample_interval=0.02,
            )
            mapping = MappingController(
                FakeCamera(),
                drive,
                pan_tilt,
                range_sensor,
                MappingStore(Path(directory) / "mapping.sqlite3"),
            )
            try:
                mapping.start_autonomous(attended=True)
                deadline = time.monotonic() + 5
                while mapping.status()["enabled"] and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(mapping.status()["state"], "error")
                self.assertIn("No valid ultrasonic readings", mapping.status()["error"])
                self.assertEqual(driver.command, (0, 0, 0, 0))
            finally:
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()

    def test_autonomous_mode_turns_toward_a_clear_side_when_forward_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = RecordingDriver()
            drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=True)
            pan_driver = SimulatedPanTilt()
            pan_tilt = PanTiltController(pan_driver, physical=True)

            class SideClearSensor(FakeRangeSensor):
                def get_distance(self) -> float:
                    return 30.0 if pan_driver.pan == 90 else 110.0

            range_sensor = RangeSensorController(
                SideClearSensor(),
                physical=True,
                sample_interval=0.02,
            )
            mapping = MappingController(
                FakeCamera(),
                drive,
                pan_tilt,
                range_sensor,
                MappingStore(Path(directory) / "mapping.sqlite3"),
            )
            try:
                mapping.start_autonomous(attended=True)
                deadline = time.monotonic() + 5
                while not any(command[0] != command[2] for command in driver.commands) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(
                    any(command[0] != command[2] for command in driver.commands),
                    f"state={mapping.status()}, pan={pan_driver.pan}, commands={driver.commands}",
                )
                self.assertEqual(mapping.status()["mode"], "explore")
                # A closed-loop turn toward the opening takes over a second.
                deadline = time.monotonic() + 5
                while mapping.status()["move_count"] == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertGreaterEqual(mapping.status()["move_count"], 1)
            finally:
                mapping.stop()
                self.assertEqual(driver.command, (0, 0, 0, 0))
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()

    def test_autonomous_mode_periodically_turns_when_a_side_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = RecordingDriver()
            drive = DriveController(driver, max_duty=1000, command_timeout=1, physical=True)
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
            range_sensor = RangeSensorController(
                FakeRangeSensor(distance=100.0),
                physical=True,
                sample_interval=0.02,
            )
            mapping = MappingController(
                FakeCamera(),
                drive,
                pan_tilt,
                range_sensor,
                MappingStore(Path(directory) / "mapping.sqlite3"),
            )
            clear_scan = [
                {"angle_deg": angle, "distance_cm": 100.0, "samples": 3, "error": None}
                for angle in (-60, -30, 0, 30, 60)
            ]
            moves: list[tuple[str, float]] = []
            original_move_burst = mapping._move_burst
            original_turn_by = mapping._turn_by

            def track_move_burst(
                throttle: float,
                duration: float,
                *,
                steering: float = 0,
            ) -> bool:
                moves.append(("drive", throttle))
                return original_move_burst(throttle, duration, steering=steering)

            def track_turn_by(degrees: float) -> bool:
                moves.append(("turn", degrees))
                return original_turn_by(degrees)

            try:
                with (
                    patch.object(mapping, "_capture_observation", return_value=clear_scan),
                    patch.object(mapping, "_move_burst", side_effect=track_move_burst),
                    patch.object(mapping, "_turn_by", side_effect=track_turn_by),
                ):
                    mapping.start_autonomous(attended=True)
                    deadline = time.monotonic() + 5
                    while not any(
                        command[0] != command[2] and command != (0, 0, 0, 0)
                        for command in driver.commands
                    ) and time.monotonic() < deadline:
                        time.sleep(0.01)

                    movement_commands = [
                        command for command in driver.commands if command != (0, 0, 0, 0)
                    ]
                    self.assertGreaterEqual(len(movement_commands), 4)
                    self.assertTrue(
                        any(
                            all(0 < duty < 1000 for duty in command)
                            for command in movement_commands
                        )
                    )
                    self.assertTrue(
                        any(
                            command[0] != command[2]
                            for command in movement_commands
                        )
                    )
                    # Three bursts ahead, then a look-around turn toward the
                    # open side, kept to one small step.
                    self.assertEqual(
                        moves[:4],
                        [("drive", 1.0), ("drive", 1.0), ("drive", 1.0), ("turn", -AUTO_MAX_TURN_DEGREES)],
                    )
                    self.assertEqual(mapping.status()["mode"], "explore")
                    self.assertGreaterEqual(mapping.status()["move_count"], 3)
            finally:
                mapping.stop()
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()

    def test_exploration_requires_attendance_and_hardware(self):
        drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=False)
        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(
            FakeCamera(),
            drive,
            pan_tilt,
            range_sensor,
            MappingStore(Path(":memory:")),
        )
        try:
            with self.assertRaises(ValueError):
                mapping.start_autonomous(attended=False)
            with self.assertRaises(RuntimeError):
                mapping.start_autonomous(attended=True)
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def test_sweep_reaches_every_reading_from_the_same_side(self):
        class RecordingPanTilt(SimulatedPanTilt):
            def __init__(self) -> None:
                super().__init__()
                self.history: list[int] = []

            def set_angles(self, pan: int, tilt: int) -> None:
                super().set_angles(pan, tilt)
                self.history.append(pan)

        drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
        pan_driver = RecordingPanTilt()
        pan_tilt = PanTiltController(pan_driver, physical=False)
        range_sensor = RangeSensorController(FakeRangeSensor(), physical=True, sample_interval=0.5)
        mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        try:
            pan_driver.history.clear()
            scan, _ = mapping._sweep(90, 90, (54, 126))
            sweep = pan_driver.history[1:-1]
            self.assertEqual(pan_driver.history[0], 30)
            self.assertEqual(sweep, sorted(sweep))
            self.assertEqual(sweep, list(SCAN_ANGLES))
            self.assertEqual(pan_driver.history[-1], 90)
            self.assertEqual(len(scan), len(SCAN_ANGLES))
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def test_forward_clearance_covers_the_corridor_and_side_needs_width(self):
        scan = [
            {"angle_deg": angle, "distance_cm": 200.0, "samples": 3, "error": None}
            for angle in range(-60, 61, 6)
        ]
        scan[12]["distance_cm"] = 45.0  # 12 degrees left of centre
        self.assertEqual(forward_clearance(scan), 45.0)
        scan[10]["distance_cm"] = None
        self.assertIsNone(forward_clearance(scan))

        sides = [
            {"angle_deg": angle, "distance_cm": 50.0, "samples": 3, "error": None}
            for angle in range(-60, 61, 6)
        ]
        sides[0]["distance_cm"] = 300.0  # a single narrow gap on the right
        for index in range(16, 21):
            sides[index]["distance_cm"] = 150.0  # a car-wide opening on the left
        clearances = direction_clearances(sides)
        self.assertEqual(clearances[-60.0], 50.0)
        self.assertEqual(clearances[48.0], 150.0)
        step = plan_explore_step(sides, (0.0, 0.0, 0.0), [], 0, 0, 0)
        # Toward the opening at 48 degrees, one small step at a time.
        self.assertEqual((step.action, step.degrees), ("turn", AUTO_MAX_TURN_DEGREES))

    def test_boxed_in_corner_turns_to_search_instead_of_shuttling(self):
        # The first real sweep from the passage run: walls 48-66 cm all round.
        data = json.loads((Path(__file__).parent / "data" / "passage_run.json").read_text())
        corner = data["observations"][0]["scan"]
        forward = plan_explore_step(corner, (0.0, 0.0, 0.0), [], 0, 0, 0)
        self.assertEqual(forward.action, "forward")  # 64 cm ahead is still clear
        closer = [
            {**item, "distance_cm": item["distance_cm"] - 9.0 if item["distance_cm"] else None}
            for item in corner
        ]
        first = plan_explore_step(closer, (0.0, 9.0, 0.0), [(0.0, 0.0, 0.0)], 1, 0, 0)
        self.assertEqual(first.action, "search")
        second = plan_explore_step(
            closer, (0.0, 9.0, 45.0), [(0.0, 0.0, 0.0)], 0, 1, 0, search_sign=math.copysign(1, first.degrees)
        )
        self.assertEqual(second.degrees, first.degrees)  # keeps turning the same way
        exhausted = plan_explore_step(closer, (0.0, 9.0, 0.0), [], 0, AUTO_MAX_SEARCH_TURNS, 0)
        self.assertEqual(exhausted.action, "reverse")
        self.assertEqual(
            plan_explore_step(closer, (0.0, 9.0, 0.0), [], 0, AUTO_MAX_SEARCH_TURNS, 1).action,
            "blocked",
        )

    def test_exploration_prefers_openings_away_from_where_it_has_been(self):
        scan = [
            {"angle_deg": angle, "distance_cm": 40.0 if abs(angle) < 30 else 150.0, "samples": 3, "error": None}
            for angle in range(-57, 61, 3)
        ]
        # Having come from the left, the car should look right.
        visited = [(-60.0, 40.0, 0.0), (-40.0, 20.0, 0.0), (0.0, 0.0, 0.0)]
        step = plan_explore_step(scan, (0.0, 0.0, 0.0), visited, 0, 0, 0)
        self.assertEqual(step.action, "turn")
        self.assertLess(step.degrees, 0)

    def test_strip_of_pictures_measures_the_turn_actually_made(self):
        scene = PanoramaScene()
        for planned, actual in ((25, 25), (25, 33), (25, 70), (-25, -55)):
            sign = 1 if planned >= 0 else -1
            strip = {90 + sign * step: (scene.view(sign * step),) for step in range(0, 61, 10)}
            current = (scene.view(actual), scene.view(actual))
            self.assertAlmostEqual(MappingController._measure_turn(strip, current, 90), actual, delta=1.0)

    def test_camera_aligns_a_turn_when_odometry_under_reads_it(self):
        scene = PanoramaScene()
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_driver = SimulatedPanTilt()
        pan_tilt = PanTiltController(pan_driver, physical=True)

        def true_heading() -> float:
            # The real car turns 2.8 times as far as its odometry says,
            # as when a planned 25-degree turn came out near 70 degrees.
            return 2.8 * drive.odometry()[2]

        class SceneCamera:
            def capture_snapshot(self) -> bytes:
                return scene.view(true_heading() + pan_driver.pan - 90)

        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(SceneCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        try:
            with patch("robocar.mapping.AUTO_ALIGN_SETTLE_SECONDS", 0.001):
                self.assertTrue(mapping._turn_and_align(25.0))
            self.assertAlmostEqual(true_heading(), 25.0, delta=4.0)
            self.assertAlmostEqual(mapping._verified_turn, true_heading(), delta=1.5)
            self.assertEqual(pan_driver.pan, 90)
            # The turn taught odometry that turns run long.
            self.assertGreater(mapping._odometry_scale[1], 1.2)
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def test_turn_without_usable_camera_views_relies_on_odometry(self):
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        turns: list[float] = []
        try:
            with (
                patch("robocar.mapping.AUTO_ALIGN_SETTLE_SECONDS", 0.001),
                patch.object(mapping, "_turn_by", side_effect=lambda degrees, *_: turns.append(degrees) or True),
            ):
                self.assertTrue(mapping._turn_and_align(-25.0))
            self.assertEqual(turns, [-25.0])
            self.assertIsNone(mapping._verified_turn)
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def test_turn_by_rotates_close_to_the_requested_angle(self):
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        try:
            for degrees in (45.0, -30.0):
                before = drive.odometry()[2]
                self.assertTrue(mapping._turn_by(degrees))
                self.assertAlmostEqual(drive.odometry()[2] - before, degrees, delta=6.0)
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def test_surface_map_survives_restart_by_replaying_the_latest_run(self):
        room = SonarRoom()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "mapping.sqlite3"
            store = MappingStore(database)
            for pose in [(0.0, 0.0, 0.0), (0.0, 30.0, 0.0), (0.0, 30.0, 35.0)]:
                store.save(b"jpeg", room.scan(pose), "replayed", "explore", pose, {"pose_sigma": [1.0, 0.5]})
            store.close()

            drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
            pan_tilt = PanTiltController(SimulatedPanTilt(), physical=False)
            range_sensor = RangeSensorController(None, physical=False)
            mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(database))
            try:
                surface_map = mapping.surface_map()
                self.assertEqual(surface_map["run_id"], "replayed")
                self.assertEqual(surface_map["scan_count"], 3)
                self.assertEqual(len(surface_map["path"]), 3)
                front = max(surface_map["surfaces"], key=lambda surface: surface["length_cm"])
                self.assertAlmostEqual(front["start"][1], 120.0, delta=1.5)
                self.assertAlmostEqual(front["end"][1], 120.0, delta=1.5)
            finally:
                mapping.close()
                drive.close()
                pan_tilt.close()
                range_sensor.close()


class SlippingRoomTests(unittest.TestCase):
    """Drive a simulated car whose wheels cover 20% less ground than odometry claims."""

    def setUp(self):
        fast_sweeps(self)

    def test_sonar_registration_corrects_slipping_odometry(self):
        room = SonarRoom()
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_driver = SimulatedPanTilt()
        pan_tilt = PanTiltController(pan_driver, physical=True)

        def true_pose():
            x, y, heading = drive.odometry()
            return (0.8 * x, 0.8 * y, 0.8 * heading)

        class RoomSensor(FakeRangeSensor):
            def get_distance(self) -> float | None:
                return room.distance(true_pose(), pan_driver.pan - 90)

        range_sensor = RangeSensorController(RoomSensor(), physical=True, sample_interval=0.5)
        mapping = MappingController(
            FakeCamera(),
            drive,
            pan_tilt,
            range_sensor,
            MappingStore(Path(":memory:")),
        )
        mapping._run_id = "slipping-room"
        try:
            mapping._capture_observation(mode="explore")
            for throttle, steering in ((1.0, 0), (1.0, 0), (0, 1.0), (1.0, 0)):
                self.assertTrue(mapping._move_burst(throttle, 0.25, steering=steering))
                mapping._capture_observation(mode="explore")

            truth = true_pose()
            odometry = drive.odometry()
            estimate = mapping._pose
            odometry_error = np.hypot(odometry[0] - truth[0], odometry[1] - truth[1])
            estimate_error = np.hypot(estimate[0] - truth[0], estimate[1] - truth[1])
            self.assertGreater(odometry_error, 10.0)
            self.assertLess(estimate_error, odometry_error / 2)
            self.assertLess(abs(relative(truth, estimate)[2]), abs(relative(truth, odometry)[2]) + 0.5)
            self.assertTrue(
                any(observation["estimate"]["registration"]["accepted"] for observation in mapping.observations())
            )
            self.assertLess(mapping.status()["odometry_scale"][0], 1.0)

            surface_map = mapping.surface_map()
            front = max(surface_map["surfaces"], key=lambda surface: surface["length_cm"])
            self.assertAlmostEqual((front["start"][1] + front["end"][1]) / 2, 120.0, delta=4.0)
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()


if __name__ == "__main__":
    unittest.main()
