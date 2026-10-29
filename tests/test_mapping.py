import json
from dataclasses import replace
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
    AUTO_LOOK_AROUND_TURNS,
    AUTO_MAX_TURN_PER_STOP_DEGREES,
    AUTO_STEP_CM,
    SCAN_ANGLES,
    CameraMotionEstimate,
    assess_picture,
    MappingController,
    MappingStore,
    direction_clearances,
    plan_explore_step,
    estimate_camera_yaw_change,
    forward_clearance,
)
from robocar.exploration import CoverageGrid, Frontier, Waypoint
from robocar.localization import compose, relative
from robocar.vision_debug import render_matches
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
            """The previous view at the first stop, the turned view at the second."""

            def __init__(self) -> None:
                self.view = previous_jpeg.tobytes()

            def capture_snapshot(self) -> bytes:
                return self.view

        drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=True)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=True)
        range_sensor = RangeSensorController(None, physical=False)
        store = MappingStore(Path(":memory:"))
        camera = ImageSequenceCamera()
        mapping = MappingController(camera, drive, pan_tilt, range_sensor, store)
        mapping._run_id = "visual-run"
        odometry = iter([(0.0, 0.0, 0.0), (0.0, 0.0, 12.0)])
        try:
            with patch.object(drive, "odometry", side_effect=lambda: next(odometry)):
                mapping._capture_observation(mode="explore")
                camera.view = current_jpeg.tobytes()
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
                    list(range(57, -61, -3)),  # pan 33 (left) to 150 (right)
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
                # Stopped, or a straight move or turn with every wheel at one speed.
                self.assertEqual(len({abs(duty) for duty in driver.command}), 1)
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
                FakeRangeSensor(distance=15.0),
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
                {"angle_deg": angle, "distance_cm": 15.0, "samples": 3, "error": None}
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

    def corridor_grid(self, pose, headings, walls=None):
        # A corridor whose far end (at 290 cm) is beyond where one sweep clears space.
        walls = walls or (
            ((-40.0, -50.0), (-40.0, 290.0)),
            ((40.0, -50.0), (40.0, 290.0)),
            ((-40.0, -50.0), (40.0, -50.0)),
            ((-40.0, 290.0), (40.0, 290.0)),
        )
        room = SonarRoom(walls=walls, posts=())
        grid = CoverageGrid()
        for heading in headings:
            looking = (pose[0], pose[1], heading)
            grid.integrate(room.scan(looking), looking, 25.0)
        return room, grid

    def test_a_long_reading_off_to_the_left_draws_the_car_that_way(self):
        # Walls close ahead and to the right, but nothing echoes on the left:
        # that is where the unknown is, so that is where it goes to look.
        scan = [
            {"angle_deg": angle, "distance_cm": 300.0 if angle >= 30 else 70.0, "samples": 3, "error": None}
            for angle in range(57, -61, -3)
        ]
        grid = CoverageGrid()
        pose = (0.0, 0.0, 0.0)
        grid.integrate(scan, pose, 25.0)
        step = plan_explore_step(scan, pose, grid, 0, 0)
        self.assertEqual(step.action, "turn")
        self.assertGreater(step.degrees, 20.0)

    def test_it_drives_toward_the_unexplored_far_end_of_a_corridor(self):
        pose = (0.0, 0.0, 0.0)
        room, grid = self.corridor_grid(pose, (0.0, -90.0, 180.0, 90.0))
        step = plan_explore_step(room.scan(pose), pose, grid, 0, 0)
        self.assertEqual(step.action, "forward")
        self.assertGreater(step.distance_cm, AUTO_STEP_CM)

    def test_when_the_sonar_blocks_the_way_it_does_not_drive(self):
        pose = (0.0, 0.0, 0.0)
        room, grid = self.corridor_grid(pose, (0.0, -90.0, 180.0, 90.0))
        blocked = [
            {**item, "distance_cm": 30.0 if abs(item["angle_deg"]) <= 15 else item["distance_cm"]}
            for item in room.scan(pose)
        ]
        step = plan_explore_step(blocked, pose, grid, 0, 0)
        self.assertEqual(step.action, "turn")

    def test_a_fully_seen_room_is_explored(self):
        walls = (((-60.0, -60.0), (60.0, -60.0)), ((60.0, -60.0), (60.0, 60.0)), ((60.0, 60.0), (-60.0, 60.0)), ((-60.0, 60.0), (-60.0, -60.0)))
        pose = (0.0, 0.0, 0.0)
        room, grid = self.corridor_grid(pose, range(0, 360, 30), walls=walls)
        scan = room.scan(pose)
        # It looks round first, then reports the room explored.
        self.assertEqual(plan_explore_step(scan, pose, grid, 0, 0).action, "turn")
        self.assertEqual(plan_explore_step(scan, pose, grid, AUTO_LOOK_AROUND_TURNS, 0).action, "done")

    def test_turns_go_the_short_way_in_one_stop(self):
        class BehindLeft(CoverageGrid):
            def next_waypoint(self, pose, avoid_within_deg=None, previous_target=None, learner=None):
                return Waypoint(160.0, 80.0, Frontier((0.0, 0.0), (-30.0, -70.0), 10, 90.0, 500.0))

        scan = [{"angle_deg": angle, "distance_cm": 90.0, "samples": 3, "error": None} for angle in range(-57, 61, 3)]
        step = plan_explore_step(scan, (0.0, 0.0, 0.0), BehindLeft(), 0, 0)
        self.assertEqual((step.action, step.degrees), ("turn", AUTO_MAX_TURN_PER_STOP_DEGREES))

    def test_strip_of_pictures_measures_the_turn_actually_made(self):
        scene = PanoramaScene()
        for planned, actual in ((25, 25), (25, 33), (25, 70), (-25, -55)):
            sign = 1 if planned >= 0 else -1
            strip = {90 - sign * step: (scene.view(sign * step),) for step in range(0, 61, 10)}
            current = (scene.view(actual), scene.view(actual))
            self.assertAlmostEqual(MappingController._measure_turn(strip, current, 90), actual, delta=1.0)

    def test_a_single_strip_reading_is_not_trusted(self):
        scene = PanoramaScene()
        strip = {90: (scene.view(0),), 30: (scene.view(60),)}
        # Only the 60-degree picture overlaps a 70-degree turn: no consensus.
        self.assertIsNone(MappingController._measure_turn(strip, (scene.view(70), scene.view(70)), 90))

    def test_camera_aligns_a_turn_when_odometry_under_reads_it(self):
        scene = PanoramaScene()
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
        pan_driver = SimulatedPanTilt()
        pan_tilt = PanTiltController(pan_driver, physical=True)

        def true_heading() -> float:
            # The real car turns 1.6 times as far as its odometry says.
            return 1.6 * drive.odometry()[2]

        class SceneCamera:
            def capture_snapshot(self) -> bytes:
                return scene.view(true_heading() + 90 - pan_driver.pan)

        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(SceneCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        try:
            with patch("robocar.mapping.AUTO_ALIGN_SETTLE_SECONDS", 0.001):
                self.assertTrue(mapping._turn_and_align(25.0))
            self.assertAlmostEqual(true_heading(), 25.0, delta=4.0)
            self.assertAlmostEqual(mapping._verified_turn, true_heading(), delta=1.5)
            self.assertEqual(pan_driver.pan, 90)
            # The turn taught odometry that turns run long.
            self.assertGreater(mapping._odometry_scale[1], 1.1)
        finally:
            mapping.close()
            drive.close()
            pan_tilt.close()
            range_sensor.close()

    def turning_rig(self):
        """A car over a panorama whose real turns run 1.6 times what odometry
        says, with every head position recorded."""
        scene = PanoramaScene()
        drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)

        class RecordingPanTilt(SimulatedPanTilt):
            def __init__(self) -> None:
                super().__init__()
                self.visited: list[int] = []

            def set_angles(self, pan: int, tilt: int) -> None:
                super().set_angles(pan, tilt)
                self.visited.append(pan)

        pan_driver = RecordingPanTilt()
        pan_tilt = PanTiltController(pan_driver, physical=True)

        def true_heading() -> float:
            return 1.6 * drive.odometry()[2]

        class SceneCamera:
            def capture_snapshot(self) -> bytes:
                return scene.view(true_heading() + 90 - pan_driver.pan)

        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(SceneCamera(), drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        for resource in (range_sensor, pan_tilt, drive, mapping):
            self.addCleanup(resource.close)
        return mapping, drive, pan_driver, true_heading

    def strip_pans(self):
        return [90 + offset for offset in range(-60, 61, 10)]

    def test_a_turn_reuses_the_strip_taken_during_the_sweep(self):
        mapping, drive, pan_driver, true_heading = self.turning_rig()
        with patch("robocar.mapping.AUTO_ALIGN_SETTLE_SECONDS", 0.001):
            mapping._sweep(100, 90, (), self.strip_pans())
            mapping._sweep_strip = replace(mapping._sweep_strip, odometry=drive.odometry())
            self.assertEqual(sorted(mapping._sweep_strip.pictures), self.strip_pans())
            pan_driver.visited.clear()
            self.assertTrue(mapping._turn_and_align(25.0))
        # No second pass of the head: only the forward view after the turn.
        self.assertEqual(set(pan_driver.visited), {90, 84})
        # Measured and corrected just as well as with a strip of its own.
        self.assertAlmostEqual(true_heading(), 25.0, delta=4.0)
        self.assertAlmostEqual(mapping._verified_turn, true_heading(), delta=1.5)
        self.assertIsNone(mapping._sweep_strip)  # used once: the car has turned

    def test_after_moving_the_sweep_strip_is_not_trusted(self):
        mapping, drive, pan_driver, true_heading = self.turning_rig()
        with patch("robocar.mapping.AUTO_ALIGN_SETTLE_SECONDS", 0.001):
            mapping._sweep(100, 90, (), self.strip_pans())
            mapping._sweep_strip = replace(mapping._sweep_strip, odometry=drive.odometry())
            self.assertTrue(mapping._move_burst(1.0, 0.2))
            pan_driver.visited.clear()
            self.assertTrue(mapping._turn_and_align(25.0))
        self.assertIn(40, pan_driver.visited)  # it took a fresh strip
        self.assertAlmostEqual(mapping._verified_turn, true_heading(), delta=1.5)

    def test_only_exploring_takes_strip_pictures_during_the_sweep(self):
        mapping, drive, pan_driver, true_heading = self.turning_rig()
        mapping._run_id = "survey"
        mapping._capture_observation(mode="survey")
        self.assertIsNone(mapping._sweep_strip)
        mapping._capture_observation(mode="explore")
        self.assertEqual(sorted(mapping._sweep_strip.pictures), self.strip_pans())
        self.assertEqual(mapping._sweep_strip.odometry, drive.odometry())

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
                return room.distance(true_pose(), 90 - pan_driver.pan)

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

    def test_a_new_run_after_a_restart_continues_the_existing_map(self):
        room = SonarRoom()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "mapping.sqlite3"
            parked = (0.0, 0.0, 0.0)
            for run_id, moves in (("first", ((1.0, 0), (0, 1.0))), ("second", ((1.0, 0),))):
                # Each run is a fresh service: odometry restarts at zero, but
                # the car is still where the last run left it.
                drive = DriveController(SimulatedDriver(), max_duty=4095, command_timeout=1, physical=True)
                pan_driver = SimulatedPanTilt()
                pan_tilt = PanTiltController(pan_driver, physical=True)
                start = parked

                def true_pose(drive=drive, start=start):
                    return compose(start, drive.odometry())

                class RoomSensor(FakeRangeSensor):
                    def get_distance(self, true_pose=true_pose, pan_driver=pan_driver) -> float | None:
                        return room.distance(true_pose(), 90 - pan_driver.pan)

                range_sensor = RangeSensorController(RoomSensor(), physical=True, sample_interval=0.5)
                mapping = MappingController(FakeCamera(), drive, pan_tilt, range_sensor, MappingStore(database))
                try:
                    mapping._run_id = run_id
                    mapping._continue_map(run_id)
                    mapping._capture_observation(mode="explore")
                    if run_id == "second":
                        # The first stop of the new run lands where the car
                        # really is, not back at the origin.
                        self.assertLess(math.dist(mapping._pose[:2], true_pose()[:2]), 5.0)
                    for throttle, steering in moves:
                        self.assertTrue(mapping._move_burst(throttle, 0.25, steering=steering))
                        mapping._capture_observation(mode="explore")
                    parked = true_pose()
                    estimate = mapping._pose
                    surface_map = mapping.surface_map()
                finally:
                    mapping.close()
                    drive.close()
                    pan_tilt.close()
                    range_sensor.close()
            # Both runs' sweeps refine one map, and the pose stays on track.
            self.assertEqual(surface_map["scan_count"], 5)
            self.assertLess(math.dist(estimate[:2], parked[:2]), 5.0)
            self.assertLess(abs(relative(parked, estimate)[2]), 3.0)


if __name__ == "__main__":
    unittest.main()


class ExposureCamera:
    """A camera whose pictures get brighter with exposure, over a fixed scene."""

    def __init__(self, scene_brightness: float) -> None:
        rng = np.random.default_rng(5)
        pattern = cv2.GaussianBlur(rng.random((480, 640)).astype(np.float32), (0, 0), 3)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        for _ in range(400):
            centre = (int(rng.integers(0, 640)), int(rng.integers(0, 480)))
            cv2.circle(pattern, centre, int(rng.integers(3, 12)), float(rng.random()), 2)
        self.pattern = pattern
        self.scene_brightness = scene_brightness
        self.setting: dict[str, float] = {"ev": 0.0}
        self.shots: list[dict[str, float]] = []

    def set_exposure(self, setting: dict[str, float]) -> None:
        self.setting = dict(setting)

    def capture_snapshot(self) -> bytes:
        if "exposure_us" in self.setting:
            gain = self.setting["exposure_us"] / 33_000 * self.setting.get("gain", 1.0)
        else:
            gain = 2.0 ** self.setting.get("ev", 0.0)
        self.shots.append(dict(self.setting))
        image = np.clip(self.pattern * self.scene_brightness * gain * 2, 0, 255).astype(np.uint8)
        return cv2.imencode(".jpg", image)[1].tobytes()


class ExposureTests(unittest.TestCase):
    def controller(self, camera):
        drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=False)
        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(camera, drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        self.addCleanup(range_sensor.close)
        self.addCleanup(pan_tilt.close)
        self.addCleanup(drive.close)
        self.addCleanup(mapping.close)
        return mapping

    def test_a_dark_scene_is_retaken_brighter_and_the_setting_remembered(self):
        camera = ExposureCamera(scene_brightness=12.0)
        mapping = self.controller(camera)
        jpeg, info = mapping._capture_best_picture()
        quality = assess_picture(jpeg)
        self.assertTrue(quality.decent, info)
        self.assertGreater(info["attempts"], 1)
        self.assertGreater(camera.setting.get("ev", 99.0), 0.0)
        # The next picture starts from what worked and needs a single shot.
        camera.shots.clear()
        _, again = mapping._capture_best_picture()
        self.assertEqual(again["attempts"], 1)
        self.assertEqual(camera.shots[0], info["setting"])

    def test_strip_and_calibration_pictures_get_the_exposure_check_too(self):
        camera = ExposureCamera(scene_brightness=12.0)
        mapping = self.controller(camera)
        views = mapping._capture_views([60, 90, 120], 100, 90, frames=2)
        self.assertEqual(sorted(views), [60, 90, 120])
        for frames in views.values():
            self.assertEqual(len(frames), 2)
            self.assertTrue(all(assess_picture(frame).decent for frame in frames))
        self.assertGreater(camera.setting.get("ev", 99.0), 0.0)

    def test_a_bright_scene_is_retaken_darker(self):
        camera = ExposureCamera(scene_brightness=250.0)
        mapping = self.controller(camera)
        _, info = mapping._capture_best_picture()
        self.assertLess(info["setting"].get("ev", 0.0), 0.0)

    def test_the_best_picture_is_kept_when_no_setting_is_decent(self):
        camera = ExposureCamera(scene_brightness=0.5)  # far too dark for any setting
        mapping = self.controller(camera)
        jpeg, info = mapping._capture_best_picture()
        self.assertIsNotNone(jpeg)
        self.assertFalse(info["decent"])
        self.assertLessEqual(info["attempts"], 5)
        # Of the pictures tried, the brightest (least underexposed) is kept.
        self.assertEqual(info["setting"], {"ev": 4.0})

    def test_a_camera_without_exposure_control_takes_one_snapshot(self):
        mapping = self.controller(FakeCamera())
        jpeg, info = mapping._capture_best_picture()
        self.assertEqual(jpeg, b"jpeg")
        self.assertIsNone(info)

    def test_matching_pictures_are_published_for_the_dashboard(self):
        mapping = self.controller(FakeCamera())
        scene = PanoramaScene()
        first, second = scene.view(0), scene.view(10)
        mapping._set_vision("motion", "Stop to stop", ["test"], lambda: render_matches(first, second, "Stop to stop", ["test"]))
        panels = mapping.vision_panels()
        self.assertEqual([panel["name"] for panel in panels], ["motion"])
        picture = cv2.imdecode(np.frombuffer(mapping.vision_image("motion"), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(picture.shape[1], 640)
        self.assertIsNone(mapping.vision_image("place"))


class SteadinessTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("robocar.mapping.CAMERA_STEADY_INTERVAL_SECONDS", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def controller(self, camera):
        drive = DriveController(SimulatedDriver(), max_duty=1000, command_timeout=1, physical=False)
        pan_tilt = PanTiltController(SimulatedPanTilt(), physical=False)
        range_sensor = RangeSensorController(None, physical=False)
        mapping = MappingController(camera, drive, pan_tilt, range_sensor, MappingStore(Path(":memory:")))
        for resource in (range_sensor, pan_tilt, drive, mapping):
            self.addCleanup(resource.close)
        return mapping

    def test_pictures_wait_until_the_view_stops_moving(self):
        scene = PanoramaScene()

        class SwayingCamera(ExposureCamera):
            """Still swinging for the first few frames after a turn."""

            def __init__(self):
                super().__init__(scene_brightness=60.0)
                self.calls = 0

            def capture_snapshot(self):
                self.calls += 1
                return scene.view(max(0, 5 - self.calls) * 3.0)

        camera = SwayingCamera()
        mapping = self.controller(camera)
        self.assertTrue(mapping._wait_until_steady())
        self.assertGreaterEqual(camera.calls, 5)

    def test_a_blurred_picture_is_retaken(self):
        sharp_camera = ExposureCamera(scene_brightness=60.0)
        sharp = sharp_camera.capture_snapshot()
        blurred_image = cv2.blur(cv2.imdecode(np.frombuffer(sharp, np.uint8), cv2.IMREAD_GRAYSCALE), (25, 1))
        blurred = cv2.imencode(".jpg", blurred_image)[1].tobytes()
        self.assertTrue(assess_picture(blurred).blurred)

        class ShakyCamera(ExposureCamera):
            def __init__(self):
                super().__init__(scene_brightness=60.0)
                self.calls = 0

            def capture_snapshot(self):
                self.calls += 1
                # The steadiness check sees matching frames, but the kept
                # picture comes out smeared once before the car settles.
                return blurred if self.calls == 3 else sharp

        mapping = self.controller(ShakyCamera())
        jpeg, info = mapping._capture_best_picture()
        self.assertFalse(assess_picture(jpeg).blurred)
        self.assertGreater(info["sharpness"], 20.0)

    def test_sensor_noise_alone_does_not_hold_up_pictures(self):
        rng = np.random.default_rng(3)
        base = rng.integers(60, 120, (480, 640)).astype(np.float32)

        class NoisyCamera(ExposureCamera):
            """A still but dim scene whose frames differ only by sensor noise."""

            def __init__(self):
                super().__init__(scene_brightness=60.0)
                self.calls = 0

            def capture_snapshot(self):
                self.calls += 1
                noisy = np.clip(base + rng.normal(0, 40, base.shape), 0, 255).astype(np.uint8)
                return cv2.imencode(".jpg", noisy)[1].tobytes()

        camera = NoisyCamera()
        mapping = self.controller(camera)
        self.assertTrue(mapping._wait_until_steady(timeout=5.0))
        self.assertLessEqual(camera.calls, 8)
