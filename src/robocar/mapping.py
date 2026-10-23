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

from robocar.calibration import Calibration, CalibrationStore, run_calibration
from robocar.drive import (
    FORWARD_CM_PER_SECOND_AT_FULL_DUTY,
    MOTOR_DEADBAND_FRACTION,
    MOTOR_RAMP_SECONDS,
    TURN_DEGREES_PER_SECOND_AT_FULL_DUTY,
    DriveController,
)
from robocar.exploration import CoverageGrid
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
from robocar.vision_debug import render_features, render_matches

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
AUTO_MAX_SECONDS = 240
AUTO_MAX_MOVES = 60
AUTO_MIN_CLEARANCE_CM = 60.0
AUTO_FORWARD_COMMAND = 1.0
AUTO_FORWARD_SECONDS = 0.25
# With a motion calibration, each exploration step is sized to this distance.
AUTO_STEP_CM = 25.0
AUTO_TURN_COMMAND = 1.0
AUTO_MAX_BACKUP_BURSTS = 1
# Head for the next waypoint once within this angle of it.
AUTO_HEADING_TOLERANCE_DEGREES = 15.0
# A needed turn larger than this continues to the right instead of reversing,
# so the car keeps sweeping round the room the same way.
AUTO_RIGHT_HAND_LIMIT_DEGREES = 135.0
# When the sonar blocks a waypoint within this angle of ahead, pick another.
AUTO_BLOCKED_CONE_DEGREES = 25.0
# With no reachable frontier, look round this many turns before calling it explored.
AUTO_LOOK_AROUND_TURNS = 4
# Turns made just to look (or toward something far round) are bigger: a sweep
# covers about 114 degrees, and the camera strip can still check 50.
AUTO_LOOK_TURN_DEGREES = 50.0
# Targets nearer straight ahead than this are approached in fine 25-degree steps.
AUTO_FINE_TURN_LIMIT_DEGREES = 40.0
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
AUTO_STRIP_MAX_YAW_DEGREES = 25.0
AUTO_ALIGN_TOLERANCE_DEGREES = 3.0
AUTO_ALIGN_MAX_CORRECTIONS = 2
# A measured turn must be backed by two strip pictures that agree, must be
# roughly what the wheels saw, and may only be corrected by so much at once;
# otherwise the wheels are trusted.
AUTO_STRIP_AGREEMENT_DEGREES = 4.0
AUTO_ALIGN_PLAUSIBLE_RATIO = (0.5, 2.0)
AUTO_ALIGN_MAX_CORRECTION_DEGREES = 20.0
AUTO_ALIGN_FRAMES = 2
AUTO_ALIGN_SETTLE_SECONDS = 0.15
AUTO_VIEW_STEADY_TIMEOUT_SECONDS = 1.0
PAN_LIMITS = (30, 150)
AUTO_TURN_MIN_CLEARANCE_CM = 25.0
# About 400 degrees of turning without finding a way forward, then back up.
AUTO_MAX_TURNS_WITHOUT_PROGRESS = 12
CAMERA_MOTION_FRAME_COUNT = 3
# When the pose graph moves any stop by more than this, the map is rebuilt
# from every sweep at its corrected pose.
MAP_REBUILD_MOVE_CM = 4.0
MAP_REBUILD_TURN_DEG = 2.0
VIEW_HEADING_SIGMA_AFTER_MATCH = 3.0
# Exposure settings tried for each stop's picture, darkest to brightest:
# auto-exposure with compensation, then long manual exposures with gain.
EXPOSURE_LADDER = (
    {"ev": -1.0},
    {"ev": 0.0},
    {"ev": 1.0},
    {"ev": 2.0},
    {"ev": 3.0},
    {"ev": 4.0},
    # Kept short: a 90 ms exposure right after a move came out smeared.
    {"exposure_us": 33_000, "gain": 8.0},
    {"exposure_us": 50_000, "gain": 12.0},
)
EXPOSURE_START_INDEX = 1
EXPOSURE_MAX_ATTEMPTS = 5
PICTURE_BRIGHTNESS_RANGE = (70.0, 170.0)
# Some clipping is unavoidable with dark shelves and bright windows in one
# view; only chase it when it clearly dominates one end.
PICTURE_DARK_CLIPPED_LIMIT = 0.25
PICTURE_BRIGHT_CLIPPED_LIMIT = 0.08
PICTURE_GOOD_FEATURES = 300
# Variance of the Laplacian: sharp real pictures scored 30-170, a smeared one 2.
PICTURE_MIN_SHARPNESS = 20.0
PICTURE_SHARP_REFERENCE = 60.0
# Before keeping a picture, take quick frames until two in a row match, so
# the car and head have stopped moving.
CAMERA_STEADY_DIFFERENCE = 2.5
CAMERA_STEADY_TIMEOUT_SECONDS = 2.0
CAMERA_STEADY_INTERVAL_SECONDS = 0.1
# Dim real pictures produced confident-looking nonsense (136 degree "turns"
# while driving straight) from a handful of matches; demand real support.
CAMERA_MIN_MATCHES = 25
CAMERA_MIN_INLIERS = 20
CAMERA_MIN_INLIER_FRACTION = 0.4
CAMERA_MAX_PLAUSIBLE_YAW_DEGREES = 55.0
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
    if len(good_matches) < CAMERA_MIN_MATCHES:
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
    if len(inlier_previous) < CAMERA_MIN_INLIERS or len(inlier_previous) < CAMERA_MIN_INLIER_FRACTION * len(good_matches):
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
        yaw = math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))
        # Views turned further apart than this cannot share enough of the
        # scene to have matched; such a result is a false fit.
        if abs(yaw) > CAMERA_MAX_PLAUSIBLE_YAW_DEGREES:
            return None
        return CameraMotionEstimate(yaw, bearing)

    homography, homography_inliers = cv2.findHomography(
        previous_xy,
        current_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if homography is None or homography_inliers is None or homography_inliers.sum() < CAMERA_MIN_INLIERS:
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
    yaw = statistics.median(angles)
    if abs(yaw) > CAMERA_MAX_PLAUSIBLE_YAW_DEGREES:
        return None
    return CameraMotionEstimate(yaw, None)


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


@dataclass(frozen=True)
class PictureQuality:
    brightness: float
    contrast: float
    clipped_dark: float
    clipped_bright: float
    features: int
    sharpness: float = PICTURE_SHARP_REFERENCE

    @property
    def too_dark(self) -> bool:
        return self.brightness < PICTURE_BRIGHTNESS_RANGE[0] or (
            self.clipped_dark > PICTURE_DARK_CLIPPED_LIMIT and self.clipped_bright < 0.02
        )

    @property
    def too_bright(self) -> bool:
        return self.brightness > PICTURE_BRIGHTNESS_RANGE[1] or (
            self.clipped_bright > PICTURE_BRIGHT_CLIPPED_LIMIT and self.clipped_dark < 0.10
        )

    @property
    def blurred(self) -> bool:
        return self.sharpness < PICTURE_MIN_SHARPNESS

    @property
    def decent(self) -> bool:
        return (
            not self.too_dark
            and not self.too_bright
            and not self.blurred
            and self.features >= PICTURE_GOOD_FEATURES
        )

    @property
    def score(self) -> float:
        low, high = PICTURE_BRIGHTNESS_RANGE
        outside = max(0.0, low - self.brightness, self.brightness - high)
        exposure = math.exp(-((outside / 40.0) ** 2))
        sharp = min(1.0, self.sharpness / PICTURE_SHARP_REFERENCE)
        # Usable, sharp detail first; between pictures with equally little
        # detail, the better-exposed one wins.
        return self.features * (1.0 - self.clipped_dark - self.clipped_bright) * exposure * sharp + exposure

    def summary(self) -> dict[str, object]:
        return {
            "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1),
            "clipped": round(self.clipped_dark + self.clipped_bright, 3),
            "features": self.features,
            "sharpness": round(self.sharpness, 1),
            "decent": self.decent,
        }


def assess_picture(jpeg: bytes) -> PictureQuality | None:
    """Levels and usable detail of a picture, before any enhancement."""
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return PictureQuality(
        float(image.mean()),
        float(image.std()),
        float((image <= 4).mean()),
        float((image >= 251).mean()),
        len(cv2.ORB_create(nfeatures=500).detect(image, None)),
        float(cv2.Laplacian(image, cv2.CV_64F).var()),
    )


def _steadiness_frame(jpeg: bytes | None) -> np.ndarray | None:
    if jpeg is None:
        return None
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.GaussianBlur(cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA), (0, 0), 1.0).astype(np.float32)


def exposure_label(setting: dict[str, float]) -> str:
    if "exposure_us" in setting:
        return f"manual {setting['exposure_us'] / 1000:.0f} ms, gain {setting.get('gain', 1.0):.0f}"
    return f"auto {setting.get('ev', 0.0):+.0f} EV"


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

    @property
    def path(self) -> Path:
        return self._path

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
    action: str  # "forward", "turn", "reverse", "blocked" or "done"
    degrees: float = 0.0
    reason: str = ""
    target: tuple[float, float] | None = None


def plan_explore_step(
    scan: Sequence[dict[str, object]],
    pose: Sequence[float],
    grid: CoverageGrid,
    turns_without_progress: int,
    backup_bursts: int,
    just_backed_up: bool = False,
    previous_target: Sequence[float] | None = None,
) -> ExploreStep:
    """Decide the next autonomous move: one step toward the best frontier.

    The coverage grid holds everything every sweep has seen, so the car keeps
    heading for the edges of what is mapped rather than re-mapping surfaces
    it already knows. The sonar sweep has the final say on driving forward.
    """
    ahead = forward_clearance(scan)
    clearances = direction_clearances(scan)
    # Right after backing out of a dead end, never drive straight back in.
    ahead_clear = ahead is not None and ahead >= AUTO_MIN_CLEARANCE_CM and not just_backed_up
    can_turn = bool(clearances) and min(clearances.values()) >= AUTO_TURN_MIN_CLEARANCE_CM
    turning_allowed = can_turn and turns_without_progress < AUTO_MAX_TURNS_WITHOUT_PROGRESS

    waypoint = grid.next_waypoint(pose, previous_target=previous_target)
    if waypoint is None:
        if turning_allowed and turns_without_progress < AUTO_LOOK_AROUND_TURNS:
            return ExploreStep("turn", -AUTO_LOOK_TURN_DEGREES, "no unexplored edge in reach yet; turning right to look")
        if can_turn or ahead_clear:
            return ExploreStep("done", reason="no unexplored edges left within reach")
        if backup_bursts < AUTO_MAX_BACKUP_BURSTS and not just_backed_up:
            return ExploreStep("reverse", reason="too close to turn; backing up once")
        return ExploreStep("blocked", reason="no safe path after looking all around")
    if not ahead_clear and abs(waypoint.bearing_deg) <= AUTO_BLOCKED_CONE_DEGREES:
        waypoint = grid.next_waypoint(pose, avoid_within_deg=AUTO_BLOCKED_CONE_DEGREES) or waypoint
    bearing = waypoint.bearing_deg
    target = waypoint.frontier.target
    where = f"unexplored edge {waypoint.frontier.path_cm:.0f} cm away"

    if abs(bearing) > AUTO_HEADING_TOLERANCE_DEGREES and turning_allowed:
        # Far round: turn in bigger steps (still within the camera strip),
        # and past 135 degrees go the long way, keeping to the right.
        step_limit = AUTO_LOOK_TURN_DEGREES if abs(bearing) > AUTO_FINE_TURN_LIMIT_DEGREES else AUTO_MAX_TURN_DEGREES
        if bearing > AUTO_RIGHT_HAND_LIMIT_DEGREES:
            degrees = -step_limit
        else:
            degrees = max(-step_limit, min(step_limit, bearing))
        return ExploreStep("turn", degrees, f"turning toward {where} ({bearing:+.0f}°)", target)
    if ahead_clear:
        return ExploreStep("forward", reason=f"driving toward {where}", target=target)
    if turning_allowed:
        return ExploreStep("turn", -AUTO_LOOK_TURN_DEGREES, "way ahead blocked; turning right to look", target)
    if backup_bursts < AUTO_MAX_BACKUP_BURSTS and not just_backed_up:
        return ExploreStep("reverse", reason="too close to turn; backing up once", target=target)
    return ExploreStep("blocked", reason="no safe path after looking all around", target=target)


def _guess_step_seconds(distance_cm: float) -> float:
    """Burst length expected to cover a distance, before any calibration.

    A burst ramps up for its length and down for MOTOR_RAMP_SECONDS after; at
    full duty that covers roughly speed x (burst + ramp) / 2 for short bursts.
    """
    speed = FORWARD_CM_PER_SECOND_AT_FULL_DUTY * (1 - MOTOR_DEADBAND_FRACTION)
    return max(0.05, min(0.8, 2 * distance_cm / speed - MOTOR_RAMP_SECONDS))


class _CalibrationRig:
    """The car as calibration sees it: turns, bursts, pictures and ranges."""

    def __init__(self, controller: "MappingController") -> None:
        self._controller = controller
        status = controller._pan_tilt.status()
        self._centre, self._tilt = int(status["pan"]), int(status["tilt"])
        self._origin: Pose | None = None

    def heading_odometry(self) -> float:
        return self._controller._drive.odometry()[2]

    def distance_odometry(self) -> float:
        odometry = self._controller._drive.odometry()
        if self._origin is None:
            self._origin = odometry
        return relative(self._origin, odometry)[1]

    def turn(self, degrees: float, rotation_scale: float) -> bool:
        return self._controller._turn_by(degrees, rotation_scale)

    def burst(self, throttle: float, seconds: float) -> bool:
        return self._controller._move_burst(throttle, seconds)

    def photograph(self, pan_offsets: Sequence[int]) -> dict[int, tuple[bytes, ...]]:
        pans = [self._centre + offset for offset in pan_offsets if PAN_LIMITS[0] <= self._centre + offset <= PAN_LIMITS[1]]
        views = self._controller._capture_views(pans, self._tilt, self._centre, frames=1 if len(pans) > 1 else AUTO_ALIGN_FRAMES)
        return {pan - self._centre: frames for pan, frames in views.items()}

    def measure_turn(self, strip: dict[int, tuple[bytes, ...]], current: tuple[bytes, ...] | None) -> float | None:
        return self._controller._measure_turn({self._centre + offset: frames for offset, frames in strip.items()}, current, self._centre)

    def range_ahead(self) -> float | None:
        controller = self._controller
        controller._pan_tilt.set_inhibited(True)
        try:
            controller._pan_tilt.set_angles(self._centre, self._tilt, override=True)
            # Let a running-median sensor forget readings from before the move.
            controller._stop_event.wait(max(0.06, controller._range_sensor.settle_seconds))
            return controller._range_sensor.sample_now()
        finally:
            controller._pan_tilt.set_inhibited(False)

    def clearance_all_round(self) -> float | None:
        scan, _ = self._controller._sweep(self._tilt, self._centre, ())
        distances = [float(item["distance_cm"]) for item in scan or [] if item["distance_cm"] is not None]
        return min(distances) if distances else None

    def stopped(self) -> bool:
        return self._controller._stop_event.is_set()

    def report(self, message: str) -> None:
        with self._controller._lock:
            self._controller._plan = message


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
        store_path = getattr(self._store, "path", None)
        self._calibration_store = CalibrationStore(
            store_path.parent / "calibration.json" if store_path is not None and str(store_path) != ":memory:" else None
        )
        self._calibration: Calibration | None = self._calibration_store.load()
        self._reset_localization()
        self._surface_map = SurfaceMap()
        self._grid = CoverageGrid()
        self._explore_target: tuple[float, float] | None = None
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
        # The exposure that worked best last time is where the next picture starts.
        self._exposure_index = EXPOSURE_START_INDEX
        self._vision: dict[str, dict[str, object]] = {}
        self._replay_latest_run()

    def _reset_localization(self) -> None:
        self._pose: Pose = (0.0, 0.0, 0.0)
        self._pose_covariance = np.zeros((3, 3))
        self._last_odometry: Pose | None = None
        self._last_camera_views: dict[ViewKey, tuple[bytes, ...]] | None = None
        calibration = getattr(self, "_calibration", None)
        # Start from what calibration measured; each session keeps refining it.
        self._odometry_scale = (
            (calibration.forward_ratio if calibration and calibration.forward_ratio else 1.0),
            (calibration.mean_turn_ratio() if calibration and calibration.mean_turn_ratio() else 1.0),
        )
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

    def start_calibration(self, attended: bool) -> None:
        """Measure real turns and steps (car facing a flat wall about 1-1.5 m away)."""
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before calibrating")
        drive_status = self._drive.status()
        if drive_status["mode"] != "hardware":
            raise RuntimeError("Calibration requires hardware mode")
        if any(abs(float(drive_status[axis])) > 0.01 for axis in ("throttle", "strafe", "steering")):
            raise RuntimeError("Stop the car before calibrating")
        if not self._range_sensor.status()["enabled"]:
            raise RuntimeError("Calibration requires an enabled ultrasonic sensor")
        self._start("calibrate", self._run_calibration)

    def calibration(self) -> dict[str, object] | None:
        with self._lock:
            return self._calibration.to_json() if self._calibration else None

    def _run_calibration(self) -> None:
        self._hold_camera(True)
        try:
            self._set_state("calibrating")
            self._drive.set_inhibited(False)
            with self._lock:
                previous = self._calibration
            result = run_calibration(_CalibrationRig(self), previous, _guess_step_seconds)
            if self._stop_event.is_set():
                return
            if result.turn_ratios or result.step_durations:
                self._calibration_store.save(result)
                with self._lock:
                    self._calibration = result
                    self._state = "calibrated"
                    self._odometry_scale = (
                        result.forward_ratio or self._odometry_scale[0],
                        result.mean_turn_ratio() or self._odometry_scale[1],
                    )
            else:
                self._set_state("calibration_failed")
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("Calibration stopped after an error")
                with self._lock:
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._drive.stop()
            self._set_enabled_false()
            self._hold_camera(False)

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
                self._grid = CoverageGrid()
                self._explore_target = None
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
                "calibrated_at": self._calibration.created_at if self._calibration else None,
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
            if self._grid.log_odds.size:
                data["grid"] = {
                    "bounds": [round(value, 1) for value in self._grid.bounds()],
                    "explored": round(self._grid.explored_fraction(), 3),
                }
            data["frontier_target"] = (
                [round(self._explore_target[0], 1), round(self._explore_target[1], 1)] if self._explore_target else None
            )
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
                self._grid = CoverageGrid()
                self._explore_target = None
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
        grid = CoverageGrid()
        path: list[Pose] = []
        history: dict[str, list[Pose]] = {}
        report = None
        for other, stop, pose in stops:
            report = surface_map.integrate(stop.scan, pose, *stop.pose_sigma)
            grid.integrate(stop.scan, pose, surface_map.beam_width_deg)
            if stop.run_id == run_id:
                path.append(pose)
            else:
                history.setdefault(stop.run_id, []).append(pose)
        with self._map_lock:
            self._surface_map = surface_map
            self._grid = grid
            self._path = path
            self._history = list(history.values())
            self._links = links
            self._map_run_id = run_id
        return report

    def _hold_camera(self, held: bool) -> None:
        hold = getattr(self._camera, "hold", None)
        if hold is not None:
            hold(held)

    def _run_survey(self) -> None:
        capture_pending = True
        stopped_since: float | None = None
        self._hold_camera(True)
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
            self._hold_camera(False)

    def _run_autonomous(self) -> None:
        started_at = time.monotonic()
        turns_without_progress = 0
        backup_bursts = 0
        just_backed_up = False
        self._hold_camera(True)
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
                with self._map_lock:
                    step = plan_explore_step(
                        scan,
                        pose,
                        self._grid,
                        turns_without_progress,
                        backup_bursts,
                        just_backed_up,
                        self._explore_target,
                    )
                    self._explore_target = step.target
                with self._lock:
                    self._plan = step.reason
                if step.action == "done":
                    self._set_state("explored")
                    break
                if step.action == "blocked":
                    self._set_state("blocked")
                    break

                self._drive.set_inhibited(False)
                if step.action == "forward":
                    self._set_state("moving")
                    if not self._move_burst(AUTO_FORWARD_COMMAND, self._step_seconds()):
                        break
                    turns_without_progress = 0
                    just_backed_up = False
                elif step.action == "turn":
                    self._set_state("turning")
                    if not self._turn_and_align(step.degrees):
                        break
                    turns_without_progress += 1
                    # Only a change of direction earns another backing-up.
                    backup_bursts = 0
                    just_backed_up = False
                else:
                    self._set_state("backing_up")
                    if not self._move_burst(-AUTO_FORWARD_COMMAND, AUTO_FORWARD_SECONDS):
                        break
                    backup_bursts += 1
                    turns_without_progress = 0
                    just_backed_up = True
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
            self._hold_camera(False)

    def _step_seconds(self) -> float:
        """Burst length for one exploration step, from calibration when there is one."""
        with self._lock:
            calibration = self._calibration
        measured = calibration.step_duration(AUTO_STEP_CM) if calibration else None
        return measured if measured is not None else AUTO_FORWARD_SECONDS

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
        current_frames: tuple[bytes, ...] | None = None
        corrections = 0
        for attempt in range(AUTO_ALIGN_MAX_CORRECTIONS + 1):
            current = self._capture_views([centre_pan], tilt, centre_pan, frames=AUTO_ALIGN_FRAMES)
            current_frames = current.get(centre_pan)
            actual = self._measure_turn(strip, current_frames, centre_pan)
            if actual is None:
                break
            odometry_turned = self._drive.odometry()[2] - odometry_before
            expected = odometry_turned * self._rotation_ratio(degrees)
            low, high = AUTO_ALIGN_PLAUSIBLE_RATIO
            if abs(expected) > 3.0 and not low <= actual / expected <= high:
                logger.info("Camera turn %.1f disagrees with wheels %.1f; trusting the wheels", actual, expected)
                actual = None
                break
            # Positive means the car went too far counterclockwise.
            error = actual - degrees
            if abs(error) <= AUTO_ALIGN_TOLERANCE_DEGREES or attempt == AUTO_ALIGN_MAX_CORRECTIONS:
                break
            # Correct with the turn ratio this very turn showed, so a badly
            # calibrated turn rate cannot make the correction overshoot too.
            ratio = actual / odometry_turned if abs(odometry_turned) > 2.0 else None
            correction = max(-AUTO_ALIGN_MAX_CORRECTION_DEGREES, min(AUTO_ALIGN_MAX_CORRECTION_DEGREES, -error))
            self._set_state("aligning")
            if not self._turn_by(correction, ratio if ratio is not None and 0.2 <= ratio <= 5.0 else None):
                return False
            corrections += 1
        self._show_turn(strip, current_frames, centre_pan, degrees, actual, corrections)
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

    def _show_turn(
        self,
        strip: dict[int, tuple[bytes, ...]],
        current: tuple[bytes, ...] | None,
        centre_pan: int,
        planned: float,
        actual: float | None,
        corrections: int,
    ) -> None:
        if not strip or not current:
            return
        aim = actual if actual is not None else planned
        best_pan = min(strip, key=lambda pan: abs(pan - centre_pan - aim))
        lines = [
            f"planned {planned:+.0f} deg, camera measured "
            + (f"{actual:+.1f} deg" if actual is not None else "nothing usable (wheels trusted)"),
            f"left: strip picture at {best_pan - centre_pan:+d} deg; right: view after the turn"
            + (f"; {corrections} correction(s)" if corrections else ""),
        ]
        self._set_vision("turn", "Turn check", lines, lambda: render_matches(strip[best_pan][0], current[0], "Turn check", lines))

    def _set_vision(self, name: str, title: str, lines: list[str], render: Callable[[], bytes | None]) -> None:
        """Store a dashboard picture; drawing problems never stop mapping."""
        try:
            jpeg = render()
        except Exception:
            logger.warning("Could not draw the %s picture", name, exc_info=True)
            return
        if jpeg:
            with self._lock:
                self._vision[name] = {
                    "title": title,
                    "caption": " · ".join(lines),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "jpeg": jpeg,
                }

    def coverage_png(self) -> bytes | None:
        """The coverage grid as a transparent PNG, north up."""
        with self._map_lock:
            if not self._grid.log_odds.size:
                return None
            image = self._grid.to_image()
        encoded, png = cv2.imencode(".png", image)
        return png.tobytes() if encoded else None

    def vision_panels(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {"name": name, "title": panel["title"], "caption": panel["caption"], "updated_at": panel["updated_at"]}
                for name, panel in self._vision.items()
            ]

    def vision_image(self, name: str) -> bytes | None:
        with self._lock:
            panel = self._vision.get(name)
            return panel["jpeg"] if panel else None

    def _wait_until_steady(self, timeout: float = CAMERA_STEADY_TIMEOUT_SECONDS) -> bool:
        """Take quick frames until two in a row match: the car and head are still."""
        previous = _steadiness_frame(self._camera.capture_snapshot())
        if previous is None:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_event.wait(CAMERA_STEADY_INTERVAL_SECONDS):
                return False
            current = _steadiness_frame(self._camera.capture_snapshot())
            if current is None:
                return True
            if float(np.abs(current - previous).mean()) < CAMERA_STEADY_DIFFERENCE:
                return True
            previous = current
        logger.info("Camera still changing after %.1f s; taking the picture anyway", timeout)
        return False

    def _capture_best_picture(self) -> tuple[bytes | None, dict[str, object] | None]:
        """Take a picture, check its levels, and retake with adjusted exposure
        until it is decent or the settings run out; keep the best one."""
        set_exposure = getattr(self._camera, "set_exposure", None)
        if set_exposure is None:
            return self._camera.capture_snapshot(), None
        with self._lock:
            index = self._exposure_index
        set_exposure(EXPOSURE_LADDER[index])
        self._wait_until_steady()
        tried: dict[int, PictureQuality] = {}
        retook_blurred = False
        best: tuple[int, bytes, PictureQuality] | None = None
        for _ in range(EXPOSURE_MAX_ATTEMPTS):
            set_exposure(EXPOSURE_LADDER[index])
            jpeg = self._camera.capture_snapshot()
            if jpeg is None:
                break
            quality = assess_picture(jpeg)
            if quality is None:
                # Not a picture we can measure; keep it rather than nothing.
                return jpeg, None
            tried[index] = quality
            if best is None or quality.score > best[2].score:
                best = (index, jpeg, quality)
            if quality.decent:
                break
            if quality.blurred and not quality.too_dark and not quality.too_bright and not retook_blurred:
                # Still moving or vibrating: wait and retake at the same exposure.
                retook_blurred = True
                del tried[index]
                self._wait_until_steady()
                continue
            if quality.too_dark:
                step = 1
            elif quality.too_bright:
                step = -1
            else:
                # Levels are fine but detail is thin: try pulling more out of
                # the shadows, or the highlights in a bright picture.
                step = 1 if quality.brightness < 120 else -1
            following = index + step
            if not 0 <= following < len(EXPOSURE_LADDER) or following in tried:
                break
            index = following
        if best is None:
            return None, None
        best_index, jpeg, quality = best
        set_exposure(EXPOSURE_LADDER[best_index])
        with self._lock:
            self._exposure_index = best_index
        info: dict[str, object] = {
            "setting": EXPOSURE_LADDER[best_index],
            "attempts": len(tried),
            **quality.summary(),
        }
        return jpeg, info

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
        agreeing = max(
            (
                [other for other in readings if abs(other[0] - reading[0]) <= AUTO_STRIP_AGREEMENT_DEGREES]
                for reading in readings
            ),
            key=lambda group: (len(group), sum(weight for _, weight in group)),
        )
        if len(agreeing) < 2:
            return None
        total = sum(weight for _, weight in agreeing)
        return sum(turn * weight for turn, weight in agreeing) / total

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
                self._wait_until_steady(AUTO_VIEW_STEADY_TIMEOUT_SECONDS)
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

    def _rotation_ratio(self, degrees: float) -> float:
        """Real degrees per odometry degree for a turn of this size and direction:
        the calibrated ratio, nudged by what this session has learned since."""
        with self._lock:
            session = self._odometry_scale[1]
            calibration = self._calibration
        reference = calibration.mean_turn_ratio() if calibration else None
        measured = calibration.turn_ratio(degrees) if calibration else None
        if measured is not None and reference:
            return measured * session / reference
        return session

    def _turn_by(self, degrees: float, rotation_scale: float | None = None) -> bool:
        """Rotate in place by about `degrees` (counterclockwise), closed loop on wheel odometry."""
        if self._stop_event.is_set():
            return False
        if rotation_scale is None:
            rotation_scale = self._rotation_ratio(degrees)
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
        image_jpeg, exposure = self._capture_best_picture()
        if image_jpeg is None:
            raise RuntimeError("Camera snapshot failed; see camera status for details")
        if exposure is not None:
            exposure_lines = [
                f"{exposure_label(exposure['setting'])} after {exposure['attempts']} tr"
                + ("y" if exposure["attempts"] == 1 else "ies")
                + ("" if exposure["decent"] else " (best available)"),
                f"brightness {exposure['brightness']:.0f}/255, contrast {exposure['contrast']:.0f}, "
                f"clipped {100 * float(exposure['clipped']):.1f}%",
            ]
            self._set_vision(
                "stop", "Stop picture", exposure_lines, lambda: render_features(image_jpeg, "Stop picture", exposure_lines)
            )
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
            raw_odometry[2] * (self._rotation_ratio(raw_odometry[2] * rotation_scale) if raw_odometry[2] else rotation_scale),
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
        previous_centre = (previous_views or {}).get(centre_key)
        if previous_centre:
            motion_lines = [
                "left: previous stop; right: this stop",
                "camera turn "
                + (f"{visual.yaw_deg:+.1f} deg" if visual is not None else "not measurable")
                + (" (used)" if prediction.visual_used else " (rejected)" if prediction.visual_rejected else "")
                + f"; wheels {odometry_delta[2]:+.1f} deg",
            ]
            self._set_vision(
                "motion",
                "Stop to stop",
                motion_lines,
                lambda: render_matches(previous_centre[0], image_jpeg, "Stop to stop", motion_lines),
            )

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
            "exposure": exposure,
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
            update = self._places.add_stop(
                observation_id, run_id, pose, scan, (position_sigma, heading_sigma), keyframe, edge
            )
            event = update.event
            # The pose graph has the final word, using every constraint so far.
            pose = self._places.graph.nodes[observation_id]
            if event is not None:
                covariance = np.diag(
                    [event.sigma[0] ** 2, event.sigma[1] ** 2, math.radians(event.sigma[2]) ** 2]
                )
            elif update.views:
                covariance = covariance.copy()
                covariance[2, 2] = min(covariance[2, 2], math.radians(VIEW_HEADING_SIGMA_AFTER_MATCH) ** 2)
            place_counts = self._places.counts()
        rebuild = event is not None or update.moved_cm > MAP_REBUILD_MOVE_CM or update.moved_deg > MAP_REBUILD_TURN_DEG
        if event is not None:
            logger.info("Recognized place %s: %s", event.keyframe_id, event.summary())
            remembered = self._store.image(event.keyframe_id)
            if remembered:
                place_lines = [
                    f"left: remembered stop {event.keyframe_id}"
                    + (" from an earlier session" if event.kind == "relocalized" else "")
                    + "; right: now",
                    f"{event.inliers} camera inliers; pose moved "
                    f"{math.hypot(event.correction[0], event.correction[1]):.0f} cm, {event.correction[2]:+.1f} deg"
                    + ("; sonar confirmed" if event.metric else ""),
                ]
                self._set_vision(
                    "place",
                    "Recognized place",
                    place_lines,
                    lambda: render_matches(remembered, image_jpeg, "Recognized place", place_lines),
                )
        if rebuild:
            report = self._rebuild_map(observation_id, run_id)
        else:
            with self._map_lock:
                report = self._surface_map.integrate(scan, pose, position_sigma, heading_sigma)
                self._grid.integrate(scan, pose, self._surface_map.beam_width_deg)
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
                "views_matched": update.views,
                "graph_adjusted_cm": round(update.moved_cm, 1),
                "surfaces_created": report.created if not rebuild and report else None,
                "surfaces_refined": report.fused if not rebuild and report else None,
                "surfaces_retired": report.retired if not rebuild and report else None,
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
        # A sensor that reports a running median needs time to forget the last
        # direction; such a reading is already a median, so one per angle will do.
        lag = self._range_sensor.settle_seconds
        settle = max(SCAN_STEP_SETTLE_SECONDS, lag)
        samples_per_angle = 1 if lag > 0 else SCAN_SAMPLES_PER_ANGLE
        self._pan_tilt.set_inhibited(True)
        try:
            self._pan_tilt.set_angles(SCAN_LEAD_IN_PAN, tilt, override=True)
            if self._stop_event.wait(SCAN_FIRST_SETTLE_SECONDS):
                return None, views
            for pan in pans:
                if self._stop_event.is_set():
                    return None, views
                self._pan_tilt.set_angles(pan, tilt, override=True)
                if self._stop_event.wait(settle):
                    return None, views
                if pan in view_pans:
                    frame = self._camera.capture_snapshot()
                    if frame is not None:
                        views[(pan, tilt)] = (frame,)
                if not sensor_enabled or pan not in SCAN_ANGLES:
                    continue
                samples: list[float] = []
                errors: list[str] = []
                for _ in range(samples_per_angle):
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
