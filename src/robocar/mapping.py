import json
import logging
import math
import os
import shutil
import sqlite3
import statistics
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

from robocar.drive import MOTOR_RAMP_SECONDS, TURN_DEGREES_PER_SECOND_AT_FULL_DUTY, DriveController
from robocar.localization import (
    MotionPrediction,
    Pose,
    VisualMotion,
    pose_sigma,
    predict_motion,
    propagate,
    register_scan,
    relative,
    wrap_degrees,
)
from robocar.places import (
    CAMERA_HORIZONTAL_FOV_DEGREES,
    Keyframe,
    PlaceMemory,
    detect_features,
    make_keyframe,
)
from robocar.range_sensor import RangeSensorController
from robocar.surfaces import IntegrationReport, SurfaceMap, direction, extract_echoes, pose_rotation

logger = logging.getLogger(__name__)
# Pan positions for each sweep: 40 directions 3 degrees apart, overlapping the
# ultrasonic beam so surfaces show up as continuous range profiles. The head
# first parks at the lead-in position and then only ever steps upward, so
# every reading is reached from the same side and servo backlash cancels out.
SCAN_LEAD_IN_PAN = 30
SCAN_ANGLES = tuple(range(33, 151, 3))
SCAN_FIRST_SETTLE_SECONDS = 0.3
SCAN_STEP_SETTLE_SECONDS = 0.07
SCAN_SAMPLES_PER_ANGLE = 3
SCAN_SAMPLE_INTERVAL_SECONDS = 0.05
# Extra camera views taken during the sweep when the forward view cannot be
# trusted to measure motion.
SIDE_VIEW_PANS = (54, 126)
TEXTURE_POOR_SCORE = 120.0
FORWARD_CORRIDOR_DEGREES = 12
SIDE_MIN_ANGLE_DEGREES = 30
STOP_SETTLE_SECONDS = 0.75
MINIMUM_FREE_BYTES = 256 * 1024 * 1024
AUTO_MAX_SECONDS = 120
AUTO_MAX_MOVES = 30
AUTO_MIN_CLEARANCE_CM = 60.0
AUTO_FORWARD_COMMAND = 1.0
AUTO_FORWARD_SECONDS = 0.25
AUTO_TURN_COMMAND = 1.0
AUTO_FORWARD_BURSTS_PER_TURN = 3
AUTO_MAX_BACKUP_BURSTS = 1
# Turns are closed-loop on wheel odometry and kept small; a larger change of
# direction happens over several stops, with a fresh sweep after each.
AUTO_MAX_TURN_DEGREES = 25.0
AUTO_MAX_TURN_SECONDS = 3.0
# Before each turn the camera photographs a strip across the pan, from
# straight ahead to past the destination. Afterwards the new forward view is
# matched against every strip picture to measure the turn actually made, and
# small corrective turns remove the remaining error.
AUTO_STRIP_STEP_DEGREES = 10
# The full pan range on the turn side, so even a large overshoot is measured.
AUTO_STRIP_REACH_DEGREES = 60
AUTO_STRIP_MAX_YAW_DEGREES = 20.0
AUTO_ALIGN_TOLERANCE_DEGREES = 3.0
AUTO_ALIGN_MAX_CORRECTIONS = 2
AUTO_ALIGN_FRAMES = 2
AUTO_ALIGN_SETTLE_SECONDS = 0.15
PAN_LIMITS = (30, 150)
# When everything ahead and to the sides is close, rotate in place to look
# for a way out (always the same way) before ever backing up blind.
AUTO_SEARCH_TURN_DEGREES = 25.0
AUTO_MAX_SEARCH_TURNS = 8
AUTO_TURN_MIN_CLEARANCE_CM = 25.0
# Prefer directions that lead away from where the car has already been.
AUTO_NOVELTY_LOOKAHEAD_CM = 80.0
AUTO_NOVELTY_WEIGHT = 0.5
CAMERA_MOTION_FRAME_COUNT = 3
# Let the car stop rocking and the head stop moving before the first picture.
CAMERA_SETTLE_SECONDS = 0.3
CAMERA_MOTION_FRAME_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True)
class CameraMotionEstimate:
    yaw_deg: float
    travel_bearing_deg: float | None


def estimate_camera_yaw_change(
    previous_images: Sequence[bytes],
    current_images: Sequence[bytes],
) -> float | None:
    estimates = [
        estimate_pair_camera_yaw_change(previous_jpeg, current_jpeg)
        for previous_jpeg, current_jpeg in zip(previous_images, current_images)
    ]
    valid_estimates = [estimate for estimate in estimates if estimate is not None]
    return statistics.median(valid_estimates) if valid_estimates else None


def estimate_pair_camera_yaw_change(
    previous_jpeg: bytes,
    current_jpeg: bytes,
) -> float | None:
    estimate = estimate_pair_camera_motion(previous_jpeg, current_jpeg)
    return estimate.yaw_deg if estimate is not None else None


def estimate_pair_camera_motion(
    previous_jpeg: bytes,
    current_jpeg: bytes,
) -> CameraMotionEstimate | None:
    """Estimate camera yaw (counterclockwise) and unscaled travel direction."""
    previous = detect_features(previous_jpeg, 1000)
    current = detect_features(current_jpeg, 1000)
    if previous is None or current is None or previous[2] != current[2]:
        return None
    previous_points, previous_descriptors, (image_height, image_width) = previous
    current_points, current_descriptors, _ = current
    if previous_descriptors is None or current_descriptors is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        previous_descriptors,
        current_descriptors,
        k=2,
    )
    good_matches = [
        first
        for pair in matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]
    if len(good_matches) < 12:
        return None

    previous_xy = np.float32(
        [previous_points[match.queryIdx].pt for match in good_matches]
    )
    current_xy = np.float32(
        [current_points[match.trainIdx].pt for match in good_matches]
    )
    focal_length_px = image_width / (
        2 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEGREES) / 2)
    )
    camera_matrix = np.array(
        [
            [focal_length_px, 0, image_width / 2],
            [0, focal_length_px, image_height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    essential, inliers = cv2.findEssentialMat(
        previous_xy,
        current_xy,
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5,
    )
    if essential is None or inliers is None:
        return None
    inlier_mask = inliers.ravel().astype(bool)
    inlier_previous = previous_xy[inlier_mask]
    if len(inlier_previous) < 8:
        return None
    if (
        np.ptp(inlier_previous[:, 0]) < image_width * 0.15
        or np.ptp(inlier_previous[:, 1]) < image_height * 0.15
    ):
        return None

    recovered_count, rotation, translation, pose_inliers = cv2.recoverPose(
        essential,
        previous_xy,
        current_xy,
        camera_matrix,
        mask=inliers.copy(),
    )
    minimum_pose_inliers = max(8, int(inlier_mask.sum() * 0.15))
    if recovered_count >= minimum_pose_inliers and pose_inliers is not None:
        # The camera centre moved to -R^T t in the first camera's frame
        # (x right, z forward); a bearing to the left is positive.
        centre = -rotation.T @ translation.ravel()
        bearing = (
            math.degrees(math.atan2(-centre[0], centre[2]))
            if math.hypot(centre[0], centre[2]) > 1e-6
            else None
        )
        return CameraMotionEstimate(
            math.degrees(math.atan2(rotation[0, 2], rotation[2, 2])),
            bearing,
        )

    homography, homography_inliers = cv2.findHomography(
        previous_xy,
        current_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if homography is None or homography_inliers is None or homography_inliers.sum() < 12:
        return None
    _, rotations, translations, _ = cv2.decomposeHomographyMat(
        homography,
        camera_matrix,
    )
    rotation_only = [
        candidate
        for candidate, translation in zip(rotations, translations)
        if np.linalg.norm(translation) < 0.02
    ]
    if not rotation_only:
        return None
    angles = [
        math.degrees(math.atan2(candidate[0, 2], candidate[2, 2]))
        for candidate in rotation_only
    ]
    return CameraMotionEstimate(statistics.median(angles), None)


ViewKey = tuple[int, int]


def estimate_view_motion(
    previous_views: dict[ViewKey, Sequence[bytes]],
    current_views: dict[ViewKey, Sequence[bytes]],
) -> list[tuple[ViewKey, CameraMotionEstimate]]:
    """Compare frames taken at the same pan/tilt at the previous and current stop."""
    estimates: list[tuple[ViewKey, CameraMotionEstimate]] = []
    for key, current_images in current_views.items():
        for previous_jpeg, current_jpeg in zip(previous_views.get(key, ()), current_images):
            try:
                estimate = estimate_pair_camera_motion(previous_jpeg, current_jpeg)
            except cv2.error:
                logger.warning("Could not estimate camera motion from consecutive mapping images")
                continue
            if estimate is not None:
                estimates.append((key, estimate))
    return estimates


def combine_view_estimates(
    estimates: Sequence[tuple[ViewKey, CameraMotionEstimate]],
) -> VisualMotion | None:
    if not estimates:
        return None
    yaws = [estimate.yaw_deg for _, estimate in estimates]
    # A panned camera sees the same yaw change, but its travel direction is
    # measured relative to where it was pointing.
    bearings = [
        math.radians(estimate.travel_bearing_deg + key[0] - 90)
        for key, estimate in estimates
        if estimate.travel_bearing_deg is not None
    ]
    bearing = (
        math.degrees(math.atan2(sum(map(math.sin, bearings)), sum(map(math.cos, bearings))))
        if bearings
        else None
    )
    return VisualMotion(statistics.median(yaws), bearing, len(yaws), max(yaws) - min(yaws))


def image_texture_score(jpeg: bytes) -> float:
    """Score how well an image can anchor feature matching (keypoints x coverage)."""
    detected = detect_features(jpeg, 500)
    if detected is None or not detected[0]:
        return 0.0
    keypoints, _, (height, width) = detected
    points = np.float32([keypoint.pt for keypoint in keypoints])
    coverage = min(1.0, float(np.ptp(points[:, 0])) / (0.6 * width)) * min(
        1.0, float(np.ptp(points[:, 1])) / (0.6 * height)
    )
    return len(keypoints) * coverage


class SnapshotCamera(Protocol):
    def capture_snapshot(self) -> bytes | None: ...


class PanTilt(Protocol):
    def set_angles(
        self,
        pan: int,
        tilt: int,
        *,
        override: bool = False,
    ) -> tuple[int, int]: ...

    def set_inhibited(self, inhibited: bool) -> None: ...

    def status(self) -> dict[str, object]: ...


class MappingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False, timeout=10)
        if str(path) != ":memory:":
            os.chmod(path, 0o600)
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                image_jpeg BLOB NOT NULL,
                scan_json TEXT NOT NULL,
                run_id TEXT,
                mode TEXT NOT NULL DEFAULT 'survey',
                pose_x_cm REAL,
                pose_y_cm REAL,
                heading_deg REAL,
                estimate_json TEXT
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS keyframes (
                observation_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                pan_offset_deg REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                points BLOB NOT NULL,
                descriptors BLOB NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS place_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                dx_cm REAL NOT NULL,
                dy_cm REAL NOT NULL,
                dheading_deg REAL NOT NULL,
                sigma_json TEXT NOT NULL,
                kind TEXT NOT NULL,
                inliers INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(observations)").fetchall()
        }
        for column, definition in (
            ("run_id", "TEXT"),
            ("mode", "TEXT NOT NULL DEFAULT 'survey'"),
            ("pose_x_cm", "REAL"),
            ("pose_y_cm", "REAL"),
            ("heading_deg", "REAL"),
            ("estimate_json", "TEXT"),
        ):
            if column not in columns:
                self._connection.execute(
                    f"ALTER TABLE observations ADD COLUMN {column} {definition}"
                )
        self._connection.commit()

    def save(
        self,
        image_jpeg: bytes,
        scan: list[dict[str, object]],
        run_id: str,
        mode: str,
        pose: tuple[float, float, float] | None,
        estimate: dict[str, object] | None = None,
    ) -> int:
        self.ensure_capacity()
        captured_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO observations (
                    captured_at, image_jpeg, scan_json, run_id, mode,
                    pose_x_cm, pose_y_cm, heading_deg, estimate_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    captured_at,
                    image_jpeg,
                    json.dumps(scan, separators=(",", ":")),
                    run_id,
                    mode,
                    pose[0] if pose else None,
                    pose[1] if pose else None,
                    pose[2] if pose else None,
                    json.dumps(estimate, separators=(",", ":")) if estimate is not None else None,
                ),
            )
            return int(cursor.lastrowid)

    def ensure_capacity(self) -> None:
        if str(self._path) != ":memory:":
            free_bytes = shutil.disk_usage(self._path.parent).free
            if free_bytes < MINIMUM_FREE_BYTES:
                raise OSError(
                    f"Mapping paused: less than {MINIMUM_FREE_BYTES // (1024 * 1024)} MB "
                    "of free space remains"
                )

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM observations").fetchone()
        return int(row[0])

    def latest(self) -> tuple[int, str] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT id, captured_at FROM observations ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return (int(row[0]), str(row[1])) if row else None

    def latest_run_id(self) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT run_id FROM observations WHERE run_id IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return str(row[0]) if row else None

    def recent(self, limit: int = 100) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, captured_at, scan_json, run_id, mode,
                       pose_x_cm, pose_y_cm, heading_deg, estimate_json
                FROM observations ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row(row) for row in reversed(rows)]

    def run_observations(self, run_id: str) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, captured_at, scan_json, run_id, mode,
                       pose_x_cm, pose_y_cm, heading_deg, estimate_json
                FROM observations WHERE run_id = ? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: Sequence[object]) -> dict[str, object]:
        return {
            "id": int(row[0]),
            "captured_at": row[1],
            "scan": json.loads(row[2]),
            "run_id": row[3],
            "mode": row[4],
            "pose_x_cm": row[5],
            "pose_y_cm": row[6],
            "heading_deg": row[7],
            "estimate": json.loads(row[8]) if row[8] else None,
        }

    def image(self, observation_id: int) -> bytes | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT image_jpeg FROM observations WHERE id = ?",
                (observation_id,),
            ).fetchone()
        return bytes(row[0]) if row else None

    def place_observations(self) -> list[dict[str, object]]:
        """Every stop of every run, without images, oldest first."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, captured_at, scan_json, run_id, mode,
                       pose_x_cm, pose_y_cm, heading_deg, estimate_json
                FROM observations ORDER BY id
                """
            ).fetchall()
        return [self._row(row) for row in rows]

    def save_keyframe(self, keyframe: Keyframe) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO keyframes (
                    observation_id, run_id, pan_offset_deg, width, height, points, descriptors
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    keyframe.observation_id,
                    keyframe.run_id,
                    keyframe.pan_offset_deg,
                    keyframe.width,
                    keyframe.height,
                    keyframe.points.astype(np.float32).tobytes(),
                    keyframe.descriptors.astype(np.uint8).tobytes(),
                ),
            )

    def keyframes(self) -> list[Keyframe]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT observation_id, run_id, pan_offset_deg, width, height, points, descriptors
                FROM keyframes ORDER BY observation_id
                """
            ).fetchall()
        return [
            Keyframe(
                int(row[0]),
                str(row[1]),
                float(row[2]),
                int(row[3]),
                int(row[4]),
                np.frombuffer(row[5], dtype=np.float32).reshape(-1, 2),
                np.frombuffer(row[6], dtype=np.uint8).reshape(-1, 32),
            )
            for row in rows
        ]

    def save_link(
        self,
        source: int,
        target: int,
        delta: tuple[float, float, float],
        sigma: Sequence[float],
        kind: str,
        inliers: int,
    ) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO place_links (
                    source, target, dx_cm, dy_cm, dheading_deg, sigma_json, kind, inliers, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source,
                    target,
                    delta[0],
                    delta[1],
                    delta[2],
                    json.dumps([round(float(value), 3) for value in sigma]),
                    kind,
                    inliers,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def delete_link(self, link_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM place_links WHERE id = ?", (link_id,))

    def links(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, source, target, dx_cm, dy_cm, dheading_deg, sigma_json, kind, inliers
                FROM place_links ORDER BY id
                """
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "source": int(row[1]),
                "target": int(row[2]),
                "delta": (float(row[3]), float(row[4]), float(row[5])),
                "sigma": json.loads(row[6]),
                "kind": row[7],
                "inliers": int(row[8]),
            }
            for row in rows
        ]

    def reset(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM observations")
            self._connection.execute("DELETE FROM keyframes")
            self._connection.execute("DELETE FROM place_links")
            self._connection.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('observations', 'place_links')"
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def forward_clearance(scan: Sequence[dict[str, object]]) -> float | None:
    """Nearest valid range in the corridor ahead; None without a straight-ahead echo."""
    if not any(item["angle_deg"] == 0 and item["distance_cm"] is not None for item in scan):
        return None
    return min(
        float(item["distance_cm"])
        for item in scan
        if item["distance_cm"] is not None
        and abs(float(item["angle_deg"])) <= FORWARD_CORRIDOR_DEGREES
    )


def direction_clearances(scan: Sequence[dict[str, object]]) -> dict[float, float]:
    """Clearance in each swept direction, taken across a car-wide corridor."""
    valid = {
        float(item["angle_deg"]): float(item["distance_cm"])
        for item in scan
        if item["distance_cm"] is not None
    }
    return {
        angle: min(
            distance for other, distance in valid.items() if abs(other - angle) <= FORWARD_CORRIDOR_DEGREES
        )
        for angle in valid
    }


@dataclass(frozen=True)
class ExploreStep:
    action: str  # "forward", "turn", "search", "reverse" or "blocked"
    degrees: float = 0.0
    reason: str = ""


def plan_explore_step(
    scan: Sequence[dict[str, object]],
    pose: Sequence[float],
    visited: Sequence[Sequence[float]],
    forward_bursts_since_turn: int,
    search_turns: int,
    backup_bursts: int,
    search_sign: float = 0.0,
) -> ExploreStep:
    """Decide the next autonomous move from the latest sweep (counterclockwise degrees)."""
    ahead = forward_clearance(scan)
    clearances = direction_clearances(scan)
    ahead_clear = ahead is not None and ahead >= AUTO_MIN_CLEARANCE_CM

    def novelty(angle: float) -> float:
        target = np.array(pose[:2], dtype=float) + AUTO_NOVELTY_LOOKAHEAD_CM * direction(float(pose[2]) + angle)
        if not visited:
            return AUTO_NOVELTY_LOOKAHEAD_CM
        nearest = min(float(np.linalg.norm(target - np.array(point[:2], dtype=float))) for point in visited)
        return min(AUTO_NOVELTY_LOOKAHEAD_CM, nearest)

    if ahead_clear and forward_bursts_since_turn < AUTO_FORWARD_BURSTS_PER_TURN:
        return ExploreStep("forward", reason="path ahead is clear")
    openings = [
        (min(clearance, 200.0) + AUTO_NOVELTY_WEIGHT * novelty(angle), angle)
        for angle, clearance in sorted(clearances.items())
        if abs(angle) >= SIDE_MIN_ANGLE_DEGREES and clearance >= AUTO_MIN_CLEARANCE_CM + 20
    ]
    if openings:
        _, angle = max(openings, key=lambda opening: opening[0])
        degrees = max(-AUTO_MAX_TURN_DEGREES, min(AUTO_MAX_TURN_DEGREES, angle))
        return ExploreStep("turn", degrees, f"turning {degrees:+.0f}° toward open space")
    if ahead_clear:
        return ExploreStep("forward", reason="no side opening; continuing ahead")
    if (
        search_turns < AUTO_MAX_SEARCH_TURNS
        and clearances
        and min(clearances.values()) >= AUTO_TURN_MIN_CLEARANCE_CM
    ):
        if search_sign == 0.0:
            left = [clearance for angle, clearance in clearances.items() if angle > 0]
            right = [clearance for angle, clearance in clearances.items() if angle < 0]
            search_sign = 1.0 if (np.mean(left) if left else 0.0) >= (np.mean(right) if right else 0.0) else -1.0
        degrees = search_sign * AUTO_SEARCH_TURN_DEGREES
        return ExploreStep("search", degrees, f"blocked; turning {degrees:+.0f}° to look for a way out")
    if backup_bursts < AUTO_MAX_BACKUP_BURSTS:
        return ExploreStep("reverse", reason="boxed in; backing up once")
    return ExploreStep("blocked", reason="no safe path after searching")


class MappingController:
    def __init__(
        self,
        camera: SnapshotCamera,
        drive: DriveController,
        pan_tilt: PanTilt,
        range_sensor: RangeSensorController,
        store: MappingStore,
    ) -> None:
        self._camera = camera
        self._drive = drive
        self._pan_tilt = pan_tilt
        self._range_sensor = range_sensor
        self._store = store
        self._lock = threading.Lock()
        self._map_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._state = "idle"
        self._error: str | None = None
        latest = self._store.latest()
        self._last_id = latest[0] if latest else None
        self._last_observation = latest[1] if latest else None
        self._run_id: str | None = None
        self._mode = "idle"
        self._move_count = 0
        self._plan: str | None = None
        self._reset_localization()
        self._surface_map = SurfaceMap()
        self._path: list[Pose] = []
        self._history: list[list[Pose]] = []
        self._links: list[tuple[Pose, Pose]] = []
        self._map_run_id: str | None = None
        # The place memory is only used from the mapping thread (and at start
        # up); the dashboard reads the snapshots kept below instead.
        self._places_lock = threading.Lock()
        self._places = PlaceMemory(self._store)
        self._places.load()
        self._place_counts = self._places.counts()
        self._last_place: dict[str, object] | None = None
        self._replay_latest_run()

    def _reset_localization(self) -> None:
        self._pose: Pose = (0.0, 0.0, 0.0)
        self._pose_covariance = np.zeros((3, 3))
        self._last_odometry: Pose | None = None
        self._last_camera_views: dict[ViewKey, tuple[bytes, ...]] | None = None
        self._odometry_scale = (1.0, 1.0)
        self._last_localization: dict[str, object] | None = None
        # Camera-verified rotation since the last stop, for turns too large
        # for stop-to-stop image comparison.
        self._verified_turn: float | None = None
        self._last_node: int | None = None

    def start(self) -> None:
        self._start("survey", self._run_survey)

    def start_autonomous(self, attended: bool) -> None:
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before exploring")
        drive_status = self._drive.status()
        if drive_status["mode"] != "hardware":
            raise RuntimeError("Autonomous exploration requires hardware mode")
        if any(
            abs(float(drive_status[axis])) > 0.01
            for axis in ("throttle", "strafe", "steering")
        ):
            raise RuntimeError("Stop the car before starting autonomous exploration")
        if not self._range_sensor.status()["enabled"]:
            raise RuntimeError("Autonomous exploration requires an enabled ultrasonic sensor")
        self._start("explore", self._run_autonomous)

    def _start(self, mode: str, target: Callable[[], None]) -> None:
        with self._lock:
            if self._enabled or (self._thread is not None and self._thread.is_alive()):
                raise RuntimeError("A mapping session is already active")
            self._enabled = True
            self._mode = mode
            self._state = "waiting_for_stop" if mode == "survey" else "scanning"
            self._error = None
            self._run_id = str(uuid.uuid4())
            self._move_count = 0
            self._plan = None
            self._reset_localization()
            with self._map_lock:
                self._surface_map = SurfaceMap()
                self._path = []
                self._history = []
                self._links = []
                self._map_run_id = self._run_id
            self._last_place = None
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=target,
                name=f"mapping-{mode}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            if self._state != "error":
                self._state = "idle"
            self._stop_event.set()
            thread = self._thread
        self._drive.inhibit_and_stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._drive.set_inhibited(False)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def status(self) -> dict[str, object]:
        with self._lock:
            position_sigma, heading_sigma = pose_sigma(self._pose_covariance)
            status = {
                "enabled": self._enabled,
                "state": self._state,
                "error": self._error,
                "observation_count": self._store.count(),
                "last_observation": self._last_observation,
                "last_id": self._last_id,
                "mode": self._mode,
                "run_id": self._run_id,
                "pose_model": "wheel odometry + camera yaw/travel + sonar surface registration",
                "move_count": self._move_count,
                "plan": self._plan,
                "pose": [round(value, 1) for value in self._pose],
                "pose_sigma_cm": round(position_sigma, 1),
                "heading_sigma_deg": round(heading_sigma, 1),
                "odometry_scale": [round(value, 3) for value in self._odometry_scale],
                "localization": self._last_localization,
                "places": dict(self._place_counts),
                "last_place": self._last_place,
            }
        with self._map_lock:
            status["surface_count"] = len(self._surface_map.walls())
            status["feature_count"] = len(self._surface_map.features())
        return status

    def observations(self) -> list[dict[str, object]]:
        return self._store.recent()

    def surface_map(self) -> dict[str, object]:
        with self._lock:
            position_sigma, heading_sigma = pose_sigma(self._pose_covariance)
        with self._map_lock:
            data = self._surface_map.to_dict()
            data["run_id"] = self._map_run_id
            data["path"] = [[round(value, 1) for value in pose] for pose in self._path]
            data["history"] = [[[round(value, 1) for value in pose] for pose in path] for path in self._history]
            data["links"] = [
                [[round(start[0], 1), round(start[1], 1)], [round(end[0], 1), round(end[1], 1)]]
                for start, end in self._links
            ]
        data["pose_sigma_cm"] = round(position_sigma, 1)
        data["heading_sigma_deg"] = round(heading_sigma, 1)
        return data

    def image(self, observation_id: int) -> bytes | None:
        return self._store.image(observation_id)

    def reset(self) -> None:
        with self._lock:
            if self._enabled or (self._thread is not None and self._thread.is_alive()):
                raise RuntimeError("Stop mapping before resetting its database")
            self._store.reset()
            self._last_id = None
            self._last_observation = None
            self._run_id = None
            self._mode = "idle"
            self._state = "idle"
            self._error = None
            self._move_count = 0
            self._plan = None
            self._reset_localization()
            with self._places_lock:
                self._places = PlaceMemory(self._store)
                self._place_counts = self._places.counts()
            self._last_place = None
            with self._map_lock:
                self._surface_map = SurfaceMap()
                self._path = []
                self._history = []
                self._links = []
                self._map_run_id = None

    def close(self) -> None:
        self.stop()
        self._store.close()

    def _replay_latest_run(self) -> None:
        """Rebuild the newest run's map, joined with every session it recognized."""
        run_id = self._store.latest_run_id()
        with self._places_lock:
            nodes = [node for node, run in self._places.graph.runs.items() if run == run_id]
        if not nodes:
            return
        latest = max(nodes)
        self._rebuild_map(latest, run_id)
        with self._places_lock:
            pose = self._places.graph.nodes[latest]
            position_sigma, heading_sigma = self._places.stop(latest).pose_sigma
        self._pose = pose
        self._pose_covariance = np.diag(
            [position_sigma**2, position_sigma**2, math.radians(heading_sigma) ** 2]
        )

    def _rebuild_map(self, node: int, run_id: str) -> IntegrationReport | None:
        """Replay every sweep in the node's connected sessions at their optimized poses."""
        with self._places_lock:
            component = self._places.graph.component(node)
            stops = [
                (other, self._places.stop(other), self._places.graph.nodes[other])
                for other in sorted(component)
            ]
            links = [
                (self._places.graph.nodes[source], self._places.graph.nodes[target])
                for source, target, _ in self._places.loop_links(component)
            ]
        surface_map = SurfaceMap()
        path: list[Pose] = []
        history: dict[str, list[Pose]] = {}
        report = None
        for other, stop, pose in stops:
            report = surface_map.integrate(stop.scan, pose, *stop.pose_sigma)
            if stop.run_id == run_id:
                path.append(pose)
            else:
                history.setdefault(stop.run_id, []).append(pose)
        with self._map_lock:
            self._surface_map = surface_map
            self._path = path
            self._history = list(history.values())
            self._links = links
            self._map_run_id = run_id
        return report

    def _run_survey(self) -> None:
        capture_pending = True
        stopped_since: float | None = None
        try:
            while not self._stop_event.is_set():
                drive_status = self._drive.status()
                moving = any(
                    abs(float(drive_status[axis])) > 0.01
                    for axis in ("throttle", "strafe", "steering")
                )
                if moving:
                    capture_pending = True
                    stopped_since = None
                    self._set_state("waiting_for_stop")
                elif capture_pending:
                    if stopped_since is None:
                        stopped_since = time.monotonic()
                    elif time.monotonic() - stopped_since >= STOP_SETTLE_SECONDS:
                        if self._drive.set_inhibited(True):
                            self._set_state("capturing")
                            self._capture_observation()
                            self._drive.set_inhibited(False)
                            capture_pending = False
                            stopped_since = None
                            self._set_state("waiting_for_stop")
                self._stop_event.wait(0.1)
        except Exception as exc:
            logger.exception("Mapping survey stopped after an error")
            self._drive.stop()
            with self._lock:
                self._enabled = False
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._drive.set_inhibited(False)

    def _run_autonomous(self) -> None:
        started_at = time.monotonic()
        forward_bursts_since_turn = 0
        backup_bursts = 0
        search_turns = 0
        search_sign = 0.0
        try:
            while not self._stop_event.is_set():
                if time.monotonic() - started_at >= AUTO_MAX_SECONDS:
                    self._set_state("time_limit")
                    break
                with self._lock:
                    if self._move_count >= AUTO_MAX_MOVES:
                        self._state = "move_limit"
                        break
                if not self._drive.set_inhibited(True):
                    raise RuntimeError("Drive was not stopped before a range scan")
                self._set_state("scanning")
                scan = self._capture_observation(mode="explore")
                if self._stop_event.is_set():
                    break
                if not any(item["distance_cm"] is not None for item in scan):
                    self._set_state("sensor_fault")
                    raise RuntimeError("No valid ultrasonic readings; motors remain stopped")
                # A missing straight-ahead echo (often a smooth wall at a steep
                # angle) counts as blocked: the planner turns rather than drives.
                with self._lock:
                    pose = self._pose
                    last_node = self._last_node
                # Every remembered stop in this frame, from earlier sessions too.
                with self._places_lock:
                    visited = (
                        list(self._places.component_poses(last_node).values()) if last_node is not None else []
                    )
                step = plan_explore_step(
                    scan,
                    pose,
                    visited,
                    forward_bursts_since_turn,
                    search_turns,
                    backup_bursts,
                    search_sign,
                )
                with self._lock:
                    self._plan = step.reason
                if step.action == "blocked":
                    self._set_state("blocked")
                    break

                self._drive.set_inhibited(False)
                if step.action == "forward":
                    self._set_state("moving")
                    if not self._move_burst(AUTO_FORWARD_COMMAND, AUTO_FORWARD_SECONDS):
                        break
                    forward_bursts_since_turn += 1
                    backup_bursts = search_turns = 0
                    search_sign = 0.0
                elif step.action in ("turn", "search"):
                    self._set_state("turning")
                    if not self._turn_and_align(step.degrees):
                        break
                    forward_bursts_since_turn = 0
                    if step.action == "search":
                        search_turns += 1
                        search_sign = math.copysign(1.0, step.degrees)
                else:
                    self._set_state("backing_up")
                    if not self._move_burst(-AUTO_FORWARD_COMMAND, AUTO_FORWARD_SECONDS):
                        break
                    backup_bursts += 1
                    search_turns = 0
                self._count_move()
                self._drive.set_inhibited(True)
                self._set_state("scanning")
                if self._stop_event.wait(0.1):
                    break
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("Autonomous mapping stopped after an error")
            self._drive.stop()
            if not self._stop_event.is_set():
                with self._lock:
                    self._enabled = False
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._drive.stop()
            if not self._stop_event.is_set():
                self._drive.set_inhibited(False)
            self._set_enabled_false()

    def _turn_and_align(self, degrees: float) -> bool:
        """Turn by `degrees`, then use the camera to measure and correct the turn."""
        pan_status = self._pan_tilt.status()
        centre_pan, tilt = int(pan_status["pan"]), int(pan_status["tilt"])
        sign = 1 if degrees >= 0 else -1
        offsets = [
            sign * step
            for step in range(0, AUTO_STRIP_REACH_DEGREES + 1, AUTO_STRIP_STEP_DEGREES)
            if PAN_LIMITS[0] <= centre_pan + sign * step <= PAN_LIMITS[1]
        ]
        strip = self._capture_views([centre_pan + offset for offset in offsets], tilt, centre_pan, frames=1)
        odometry_before = self._drive.odometry()[2]
        if not self._turn_by(degrees):
            return False
        actual: float | None = None
        for attempt in range(AUTO_ALIGN_MAX_CORRECTIONS + 1):
            current = self._capture_views([centre_pan], tilt, centre_pan, frames=AUTO_ALIGN_FRAMES)
            actual = self._measure_turn(strip, current.get(centre_pan), centre_pan)
            if actual is None:
                break
            # Positive means the car went too far counterclockwise.
            error = actual - degrees
            if abs(error) <= AUTO_ALIGN_TOLERANCE_DEGREES or attempt == AUTO_ALIGN_MAX_CORRECTIONS:
                break
            # Correct with the turn ratio this very turn showed, so a badly
            # calibrated turn rate cannot make the correction overshoot too.
            odometry_turned = self._drive.odometry()[2] - odometry_before
            ratio = actual / odometry_turned if abs(odometry_turned) > 2.0 else None
            self._set_state("aligning")
            if not self._turn_by(-error, ratio if ratio is not None and 0.2 <= ratio <= 5.0 else None):
                return False
        if actual is None:
            return True
        odometry_turned = self._drive.odometry()[2] - odometry_before
        with self._lock:
            self._verified_turn = (self._verified_turn or 0.0) + actual
            translation_scale, rotation_scale = self._odometry_scale
            if abs(odometry_turned) > 5.0 and 0.3 <= actual / odometry_turned <= 3.0:
                rotation_scale = min(2.0, max(0.5, 0.7 * rotation_scale + 0.3 * actual / odometry_turned))
                self._odometry_scale = (translation_scale, rotation_scale)
        return True

    @staticmethod
    def _measure_turn(
        strip: dict[int, tuple[bytes, ...]],
        current: tuple[bytes, ...] | None,
        centre_pan: int,
    ) -> float | None:
        """Combine the turn each strip picture implies: its pan offset plus the
        rotation still left between it and the current forward view."""
        if not current:
            return None
        readings: list[tuple[float, float]] = []
        for pan, frames in strip.items():
            key = (pan - centre_pan, 0)
            visual = combine_view_estimates(estimate_view_motion({key: frames}, {key: current}))
            if visual is not None and abs(visual.yaw_deg) <= AUTO_STRIP_MAX_YAW_DEGREES:
                # Pictures the car ended up nearly facing overlap most.
                readings.append((key[0] + visual.yaw_deg, 1.0 / (1.0 + abs(visual.yaw_deg))))
        if not readings:
            return None
        total = sum(weight for _, weight in readings)
        return sum(turn * weight for turn, weight in readings) / total

    def _capture_views(
        self,
        pans: Sequence[int],
        tilt: int,
        return_pan: int,
        frames: int,
    ) -> dict[int, tuple[bytes, ...]]:
        """Photograph at each pan in ascending order, approaching every position
        from below so servo backlash is the same for every picture."""
        views: dict[int, tuple[bytes, ...]] = {}
        if not pans:
            return views
        ordered = sorted(pans)
        self._pan_tilt.set_inhibited(True)
        try:
            self._pan_tilt.set_angles(max(PAN_LIMITS[0], ordered[0] - 6), tilt, override=True)
            for pan in ordered:
                self._pan_tilt.set_angles(pan, tilt, override=True)
                if self._stop_event.wait(AUTO_ALIGN_SETTLE_SECONDS):
                    return views
                pictures = [self._camera.capture_snapshot() for _ in range(frames)]
                if all(picture is not None for picture in pictures):
                    views[pan] = tuple(pictures)
            return views
        finally:
            try:
                if ordered[-1] != return_pan or len(ordered) > 1:
                    self._pan_tilt.set_angles(max(PAN_LIMITS[0], return_pan - 6), tilt, override=True)
                    self._pan_tilt.set_angles(return_pan, tilt, override=True)
            finally:
                self._pan_tilt.set_inhibited(False)

    def _turn_by(self, degrees: float, rotation_scale: float | None = None) -> bool:
        """Rotate in place by about `degrees` (counterclockwise), closed loop on wheel odometry."""
        if self._stop_event.is_set():
            return False
        if rotation_scale is None:
            with self._lock:
                rotation_scale = self._odometry_scale[1]
        # Odometry under-reads or over-reads turns by the learned scale.
        target = abs(degrees) / max(rotation_scale, 1e-3)
        steering = math.copysign(AUTO_TURN_COMMAND, degrees)
        peak_rate = TURN_DEGREES_PER_SECOND_AT_FULL_DUTY * DriveController._wheel_speed_fraction(
            float(self._drive.status()["max_duty"]) * AUTO_TURN_COMMAND
        )
        start_heading = self._drive.odometry()[2]
        started = time.monotonic()
        previous_time, previous_heading = started, start_heading
        try:
            self._drive.set_command(0, 0, steering)
            while True:
                if self._stop_event.wait(0.01):
                    return False
                now = time.monotonic()
                heading = self._drive.odometry()[2]
                rate = abs(heading - previous_heading) / max(now - previous_time, 1e-3)
                previous_time, previous_heading = now, heading
                # The motors ramp down from their current speed after the
                # stop, adding rotation in proportion to how fast it is turning.
                coast = rate * MOTOR_RAMP_SECONDS * min(1.0, rate / max(peak_rate, 1e-3)) / 2
                if (
                    abs(heading - start_heading) + coast >= target
                    or now - started >= AUTO_MAX_TURN_SECONDS
                ):
                    return True
                self._drive.set_command(0, 0, steering)
        finally:
            self._drive.stop()
            if not self._drive.wait_until_stopped():
                raise RuntimeError("Drive output did not stop after the ramp-down interval")

    def _move_burst(
        self,
        throttle: float,
        duration: float,
        *,
        steering: float = 0,
    ) -> bool:
        if self._stop_event.is_set():
            return False
        try:
            self._drive.set_command(throttle, 0, steering)
            deadline = time.monotonic() + duration
            refresh_interval = min(
                float(self._drive.status()["watchdog_timeout_seconds"]) / 2,
                0.1,
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return True
                if self._stop_event.wait(min(refresh_interval, remaining)):
                    return False
                if time.monotonic() < deadline:
                    self._drive.set_command(throttle, 0, steering)
        finally:
            self._drive.stop()
            if not self._drive.wait_until_stopped():
                raise RuntimeError("Drive output did not stop after the ramp-down interval")

    def _capture_observation(self, mode: str = "survey") -> list[dict[str, object]]:
        self._store.ensure_capacity()
        if self._stop_event.wait(CAMERA_SETTLE_SECONDS):
            return []
        pan_status = self._pan_tilt.status()
        original_pan = int(pan_status["pan"])
        original_tilt = int(pan_status["tilt"])
        image_jpeg = self._camera.capture_snapshot()
        if image_jpeg is None:
            raise RuntimeError("Camera snapshot failed; see camera status for details")
        centre_frames = [image_jpeg]
        for _ in range(CAMERA_MOTION_FRAME_COUNT - 1):
            if self._stop_event.wait(CAMERA_MOTION_FRAME_INTERVAL_SECONDS):
                return []
            frame = self._camera.capture_snapshot()
            if frame is None:
                logger.warning("Extra camera frame unavailable for motion comparison")
                continue
            centre_frames.append(frame)
        centre_key = (original_pan, original_tilt)
        views: dict[ViewKey, tuple[bytes, ...]] = {centre_key: tuple(centre_frames)}

        odometry = self._drive.odometry()
        with self._lock:
            previous_views = self._last_camera_views
            previous_odometry = self._last_odometry
            translation_scale, rotation_scale = self._odometry_scale
        raw_odometry = relative(previous_odometry, odometry) if previous_odometry else (0.0, 0.0, 0.0)
        odometry_delta = (
            raw_odometry[0] * translation_scale,
            raw_odometry[1] * translation_scale,
            raw_odometry[2] * rotation_scale,
        )
        view_estimates = estimate_view_motion(previous_views or {}, views)
        texture = image_texture_score(image_jpeg)
        # Take side pictures during the sweep when the forward view is too
        # bland, failed to measure motion, or disagrees with the wheels, and
        # keep taking them while the previous stop needed them too.
        need_side_views = texture < TEXTURE_POOR_SCORE or (
            previous_views is not None
            and (
                not view_estimates
                or predict_motion(odometry_delta, combine_view_estimates(view_estimates)).visual_rejected
                or any(key != centre_key for key in previous_views)
            )
        )
        side_pans = tuple(pan for pan in SIDE_VIEW_PANS if pan != original_pan) if need_side_views else ()
        scan, side_views = self._sweep(original_tilt, original_pan, side_pans)
        if scan is None:
            return []
        if side_views:
            views.update(side_views)
            view_estimates += estimate_view_motion(previous_views or {}, side_views)
        visual = combine_view_estimates(view_estimates)
        with self._lock:
            verified_turn, self._verified_turn = self._verified_turn, None
        if visual is None and verified_turn is not None:
            # The turn was too large for the views to overlap, but the camera
            # already measured it against the destination view.
            visual = VisualMotion(verified_turn, None, AUTO_ALIGN_FRAMES, 0.0)
        prediction = predict_motion(odometry_delta, visual)

        with self._lock:
            run_id = self._run_id
            previous_pose, previous_covariance = self._pose, self._pose_covariance
        if run_id is None:
            raise RuntimeError("Mapping run ID was not initialized")
        if previous_odometry is None:
            prior_pose, prior_covariance = previous_pose, previous_covariance
        else:
            prior_pose, prior_covariance = propagate(previous_pose, previous_covariance, prediction)

        with self._map_lock:
            echoes = [echo for run in extract_echoes(scan, self._surface_map.beam_width_deg) for echo in run]
            registration = register_scan(echoes, prior_pose, prior_covariance, self._surface_map)
        pose, covariance = registration.pose, registration.covariance
        position_sigma, heading_sigma = pose_sigma(covariance)
        scale = self._learned_odometry_scale(raw_odometry, previous_pose, pose, covariance, prediction, registration.accepted)

        with self._map_lock:
            counts = (len(self._surface_map.walls()), len(self._surface_map.features()))
        estimate = {
            "odometry_delta": [round(value, 2) for value in raw_odometry],
            "odometry_scale": [round(value, 3) for value in scale],
            "visual": None
            if visual is None
            else {
                "yaw_deg": round(visual.yaw_deg, 2),
                "travel_bearing_deg": None
                if visual.travel_bearing_deg is None
                else round(visual.travel_bearing_deg, 1),
                "estimates": visual.estimates,
                "spread_deg": round(visual.spread_deg, 2),
                "views": sorted({key[0] for key, _ in view_estimates}),
            },
            "visual_used": prediction.visual_used,
            "visual_rejected": prediction.visual_rejected,
            "travel_bearing_used": prediction.bearing_used,
            "texture_score": round(texture, 1),
            "side_views": sorted(key[0] for key in side_views),
            "predicted_pose": [round(value, 2) for value in prior_pose],
            "registration": registration.summary(),
            "pose_sigma": [round(position_sigma, 2), round(heading_sigma, 2)],
            "surfaces_before": {"walls": counts[0], "features": counts[1]},
        }
        with self._lock:
            previous_node = self._last_node
        edge = None
        if previous_node is not None:
            edge = (previous_node, relative(previous_pose, pose), self._edge_sigma(prediction, registration.accepted))
            estimate["edge"] = {
                "from": previous_node,
                "delta": [round(value, 3) for value in edge[1]],
                "sigma": [round(value, 3) for value in edge[2]],
            }
        observation_id = self._store.save(image_jpeg, scan, run_id, mode, pose, estimate)

        # Remember this place and try to recognize it from earlier stops.
        keyframe = make_keyframe(observation_id, run_id, image_jpeg, original_pan - 90)
        if keyframe is not None:
            self._store.save_keyframe(keyframe)
        with self._places_lock:
            event = self._places.add_stop(
                observation_id, run_id, pose, scan, (position_sigma, heading_sigma), keyframe, edge
            )
            if event is not None:
                pose = self._places.graph.nodes[observation_id]
                covariance = np.diag(
                    [event.sigma[0] ** 2, event.sigma[1] ** 2, math.radians(event.sigma[2]) ** 2]
                )
            place_counts = self._places.counts()
        if event is not None:
            logger.info("Recognized place %s: %s", event.keyframe_id, event.summary())
            report = self._rebuild_map(observation_id, run_id)
        else:
            with self._map_lock:
                report = self._surface_map.integrate(scan, pose, position_sigma, heading_sigma)
                self._path.append(pose)
        captured_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._pose, self._pose_covariance = pose, covariance
            self._last_node = observation_id
            self._last_odometry = odometry
            self._last_camera_views = views
            self._odometry_scale = scale
            self._place_counts = place_counts
            if event is not None:
                self._last_place = event.summary()
            self._last_localization = {
                "visual_used": prediction.visual_used,
                "visual_rejected": prediction.visual_rejected,
                "side_views": bool(side_views),
                "registration": registration.summary(),
                "place": event.summary() if event is not None else None,
                "surfaces_created": report.created if event is None and report else None,
                "surfaces_refined": report.fused if event is None and report else None,
                "surfaces_retired": report.retired if event is None and report else None,
            }
            self._last_observation = captured_at
            self._last_id = observation_id
        return scan

    @staticmethod
    def _edge_sigma(prediction: MotionPrediction, registered: bool) -> tuple[float, float, float]:
        """Uncertainty of the motion between two stops, for the pose graph."""
        covariance = prediction.covariance
        factor = 0.5 if registered else 1.0
        return (
            factor * math.sqrt(covariance[0, 0]) + 0.5,
            factor * math.sqrt(covariance[1, 1]) + 0.5,
            max(0.3, factor * math.degrees(math.sqrt(covariance[2, 2]))),
        )

    def _sweep(
        self,
        tilt: int,
        original_pan: int,
        view_pans: Sequence[int],
    ) -> tuple[list[dict[str, object]] | None, dict[ViewKey, tuple[bytes, ...]]]:
        """Sweep the ultrasonic sensor, grabbing camera views at the requested pans."""
        scan: list[dict[str, object]] = []
        views: dict[ViewKey, tuple[bytes, ...]] = {}
        sensor_enabled = bool(self._range_sensor.status()["enabled"])
        pans = sorted(set(view_pans) | (set(SCAN_ANGLES) if sensor_enabled else set()))
        if not pans:
            return scan, views
        self._pan_tilt.set_inhibited(True)
        try:
            self._pan_tilt.set_angles(SCAN_LEAD_IN_PAN, tilt, override=True)
            if self._stop_event.wait(SCAN_FIRST_SETTLE_SECONDS):
                return None, views
            for pan in pans:
                if self._stop_event.is_set():
                    return None, views
                self._pan_tilt.set_angles(pan, tilt, override=True)
                if self._stop_event.wait(SCAN_STEP_SETTLE_SECONDS):
                    return None, views
                if pan in view_pans:
                    frame = self._camera.capture_snapshot()
                    if frame is not None:
                        views[(pan, tilt)] = (frame,)
                if not sensor_enabled or pan not in SCAN_ANGLES:
                    continue
                samples: list[float] = []
                errors: list[str] = []
                for _ in range(SCAN_SAMPLES_PER_ANGLE):
                    try:
                        distance = self._range_sensor.sample_now()
                        if distance is not None:
                            samples.append(distance)
                        else:
                            errors.append("No echo received")
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
                        logger.warning("Ultrasonic scan sample failed: %s", exc)
                    if self._stop_event.wait(SCAN_SAMPLE_INTERVAL_SECONDS):
                        return None, views
                scan.append(
                    {
                        "angle_deg": pan - 90,
                        "distance_cm": round(statistics.median(samples), 1)
                        if samples
                        else None,
                        "samples": len(samples),
                        "error": errors[-1] if not samples and errors else None,
                    }
                )
        finally:
            try:
                self._pan_tilt.set_angles(original_pan, tilt, override=True)
            finally:
                self._pan_tilt.set_inhibited(False)
        return scan, views

    def _learned_odometry_scale(
        self,
        raw_odometry: Pose,
        previous_pose: Pose,
        pose: Pose,
        covariance: np.ndarray,
        prediction: MotionPrediction,
        registered: bool,
    ) -> tuple[float, float]:
        """Nudge wheel odometry scale toward what sonar and camera measured."""
        with self._lock:
            translation_scale, rotation_scale = self._odometry_scale
        actual = relative(previous_pose, pose)
        distance = math.hypot(raw_odometry[0], raw_odometry[1])
        if registered and distance > 10.0:
            unit = np.array(raw_odometry[:2]) / distance
            world_unit = pose_rotation(previous_pose[2]) @ unit
            along_sigma = math.sqrt(max(float(world_unit @ covariance[:2, :2] @ world_unit), 0.0))
            if along_sigma < 0.2 * distance:
                ratio = float(np.array(actual[:2]) @ unit) / distance
                translation_scale = min(2.0, max(0.5, 0.8 * translation_scale + 0.2 * ratio))
        if (registered or prediction.visual_used) and abs(raw_odometry[2]) > 5.0:
            heading_sigma = math.degrees(math.sqrt(max(float(covariance[2, 2]), 0.0)))
            if heading_sigma < 0.2 * abs(raw_odometry[2]):
                ratio = wrap_degrees(actual[2]) / raw_odometry[2]
                if 0.3 <= ratio <= 3.0:
                    rotation_scale = min(2.0, max(0.5, 0.8 * rotation_scale + 0.2 * ratio))
        return translation_scale, rotation_scale

    def _count_move(self) -> None:
        with self._lock:
            self._move_count += 1

    def _set_state(self, state: str) -> None:
        with self._lock:
            if self._enabled:
                self._state = state

    def _set_enabled_false(self) -> None:
        with self._lock:
            self._enabled = False
