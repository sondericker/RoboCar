import json
import logging
import math
from contextlib import contextmanager
import os
import shutil
import sqlite3
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, replace
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
from robocar.exploration import CELL_CM, CoverageGrid, ExplorationLearner, load_learner, save_learner
from robocar.hunt import HuntEngine, Look, cat_view_from, load_memory, save_memory
from robocar.localization import (
    MotionPrediction,
    Pose,
    VisualMotion,
    compose,
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
from robocar.semantics import SemanticMap, Sighting, hazards_from, mark_dynamic, sightings_from, static_scan
from robocar.surfaces import SONAR_MAX_RANGE_CM, IntegrationReport, SurfaceMap, direction, extract_echoes, pose_rotation
from robocar.vision_debug import render_detections, render_features, render_matches, render_panorama

logger = logging.getLogger(__name__)


def pan_bearing(pan: float) -> float:
    """Direction a pan angle looks, counterclockwise from straight ahead.

    The Freenove pan servo turns the head clockwise as its angle rises: pan 30
    looks left and 150 looks right (checked on the car, and how Freenove's own
    avoidance code reads it)."""
    return 90.0 - pan

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
# Forward legs run as far toward the next waypoint as a straight line allows,
# up to AUTO_LONG_STEP_CM between sweeps, stopping AUTO_WALL_STANDOFF_CM short
# of whatever the sonar sees ahead.
AUTO_LONG_STEP_CM = 60.0
AUTO_MIN_STEP_CM = 10.0
# The car believes it knows where it is on the map while: the map has walls
# to match against; a sweep matched them (or it recognized a place) within
# the last few stops; its pose estimate is this certain; and it has not since
# moved further than this on wheel counts alone.
LOCALIZED_MIN_WALLS = 2
LOCALIZED_MAX_STOPS_SINCE_FIX = 2
LOCALIZED_MAX_SIGMA_CM = 30.0
LOCALIZED_MAX_HEADING_SIGMA_DEG = 12.0
LOCALIZED_MAX_DEAD_RECKONING_CM = 200.0
LOCALIZED_MAX_DEAD_RECKONING_DEG = 180.0
# The hunt waits this long after pointing the head before a picture or
# ping, plus a little per degree turned (the servo takes about 0.2 s for 60).
HUNT_HEAD_SETTLE_SECONDS = 0.08
HUNT_HEAD_SECONDS_PER_DEGREE = 0.0025
# A hunt picture waits until the car and head have been still this long after
# their last move, then for the camera to see two matching frames in a row
# (giving up after the timeout): pictures taken sooner came out blurred.
HUNT_STILL_SECONDS = 0.4
HUNT_STEADY_TIMEOUT_SECONDS = 0.8
# Each hunt picture's sharpness is compared with recent good ones. A soft one
# is retaken after a further wait (up to HUNT_MAX_SHOTS shots, keeping the
# sharpest). If the retake was clearly sharper, that was motion blur and the
# settle delay grows; first shots that keep coming out sharp shrink it again.
HUNT_SOFT_FRACTION = 0.7
HUNT_BLUR_CONFIRM_RATIO = 1.25
HUNT_MAX_SHOTS = 3
HUNT_STILL_LIMITS_SECONDS = (0.25, 1.5)
# While hunting, a fresh picture for the dashboard this often whenever the
# car is still and not using the camera itself.
HUNT_PREVIEW_SECONDS = 0.5
# A person or animal the camera sees within AUTO_BLOCKED_CONE_DEGREES of
# where the car would drive, nearer than this (or at an unknown distance),
# blocks the way.
AUTO_HAZARD_RANGE_CM = 150.0
# Too close to a wall to spin: slide this far sideways toward the roomier
# side (readings beyond AUTO_STRAFE_SIDE_DEGREES), if it has at least
# AUTO_STRAFE_MIN_ROOM_CM.
AUTO_STRAFE_CM = 12.0
AUTO_STRAFE_SIDE_DEGREES = 40.0
AUTO_STRAFE_MIN_ROOM_CM = 30.0
# In a corridor, a target within this angle of its direction is reached by
# following the centre line, aiming this far ahead along it.
AUTO_CORRIDOR_FOLLOW_DEGREES = 70.0
AUTO_CORRIDOR_AIM_CM = 60.0
AUTO_WALL_STANDOFF_CM = 45.0
CELL_AREA_M2 = (CELL_CM / 100.0) ** 2
AUTO_TURN_COMMAND = 1.0
AUTO_MAX_BACKUP_BURSTS = 1
# Head for the next waypoint once within this angle of it.
AUTO_HEADING_TOLERANCE_DEGREES = 10.0
# When the sonar blocks a waypoint within this angle of ahead, pick another.
AUTO_BLOCKED_CONE_DEGREES = 25.0
# With no reachable frontier, look round this many turns before calling it explored.
AUTO_LOOK_AROUND_TURNS = 3
# Turns made just to look cover about a sweep's width (114 degrees), so three
# looks see all round.
AUTO_LOOK_TURN_DEGREES = 110.0
# A turn toward a target is made in one stop, up to this much, as pieces the
# camera strip can check (at most AUTO_TURN_PIECE_DEGREES each) with no sweep
# in between: a sweep only pays once the car faces somewhere new.
AUTO_MAX_TURN_PER_STOP_DEGREES = 150.0
AUTO_TURN_PIECE_DEGREES = 50.0
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
# While exploring, the strip pictures for the first turn piece are taken
# during the stop's sonar sweep (which passes every pan anyway) rather than in
# a second pass of the head. They are only reused while the car and head are
# still where they were: within this much movement.
SWEEP_STRIP_MAX_MOVE_CM = 1.0
SWEEP_STRIP_MAX_TURN_DEG = 1.0
# Strip pictures are taken at the sweep's nearest step (it steps every 3
# degrees) rather than adding stops of their own; the turn check uses each
# picture's actual angle, so being off by up to this much costs nothing.
SWEEP_STRIP_SNAP_DEG = 1.5
PANORAMA_HFOV_DEGREES = CAMERA_HORIZONTAL_FOV_DEGREES
AUTO_ALIGN_SETTLE_SECONDS = 0.15
AUTO_VIEW_STEADY_TIMEOUT_SECONDS = 1.0
PAN_LIMITS = (30, 150)
# A side reading this short at the sweep's edge (57 degrees) puts a wall about
# 17 cm from the centre: just room for the car to spin.
AUTO_TURN_MIN_CLEARANCE_CM = 20.0
# About 400 degrees of turning without finding a way forward, then back up.
AUTO_MAX_TURNS_WITHOUT_PROGRESS = 12
CAMERA_MOTION_FRAME_COUNT = 3
# When the pose graph moves any stop by more than this, the map is rebuilt
# from every sweep at its corrected pose.
MAP_REBUILD_MOVE_CM = 4.0
# A new run carries on the map from the last stop. The car may have been
# moved by hand in between, so its first stop starts this uncertain and is
# tied to the last stop only this loosely; its first sweep against the known
# walls (or a recognized place) pins it down.
CONTINUE_POSITION_SIGMA_CM = 25.0
CONTINUE_HEADING_SIGMA_DEG = 20.0
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
CAMERA_NOISE_GROWTH = 1.15
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
        math.radians(estimate.travel_bearing_deg + pan_bearing(key[0]))
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


def picture_sharpness(frame: np.ndarray) -> float:
    """Fine detail in a picture (variance of the Laplacian, at half size):
    motion blur drops it sharply, while a plain scene is just low throughout."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _steadiness_frame(jpeg: bytes | None) -> np.ndarray | None:
    if jpeg is None:
        return None
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    # A heavy blur keeps frames correlated across small shifts (so movement
    # builds up from frame to frame) and suppresses sensor noise.
    return cv2.GaussianBlur(cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA), (0, 0), 3.0).astype(np.float32)


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


class ObjectDetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class SweepStrip:
    """Turn-measurement pictures taken during a stop's sweep, and where from."""

    pictures: dict[int, tuple[bytes, ...]]  # by pan
    centre_pan: int
    tilt: int
    odometry: Pose


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
    action: str  # "forward", "turn", "strafe", "reverse", "blocked" or "done"
    degrees: float = 0.0
    reason: str = ""
    target: tuple[float, float] | None = None
    # For "forward", the leg. For "turn", a leg to drive straight after the
    # turn if a forward ping confirms the way (0: sweep after turning). For
    # "strafe", how far to slide sideways (positive: to the right).
    distance_cm: float = 0.0


def plan_explore_step(
    scan: Sequence[dict[str, object]],
    pose: Sequence[float],
    grid: CoverageGrid,
    turns_without_progress: int,
    backup_bursts: int,
    just_backed_up: bool = False,
    previous_target: Sequence[float] | None = None,
    learner: ExplorationLearner | None = None,
    hazards: Sequence[tuple[float, float | None]] = (),
) -> ExploreStep:
    """Plan the next move; people and animals the camera sees close ahead
    (`hazards`: bearing and range) block the way like a wall would."""
    step = _plan_explore_step(
        scan, pose, grid, turns_without_progress, backup_bursts, just_backed_up, previous_target, learner, hazards
    )
    if _hazard_within(hazards, 0.0) and step.action != "forward":
        step = replace(step, reason="a person or animal is close ahead; " + step.reason)
    return step


def _hazard_within(hazards: Sequence[tuple[float, float | None]], heading_change: float) -> bool:
    """Whether a person or animal is close ahead once the car has turned by `heading_change`."""
    return any(
        abs((bearing - heading_change + 180.0) % 360.0 - 180.0) <= AUTO_BLOCKED_CONE_DEGREES
        and (distance is None or distance < AUTO_HAZARD_RANGE_CM)
        for bearing, distance in hazards
    )


def _plan_explore_step(
    scan: Sequence[dict[str, object]],
    pose: Sequence[float],
    grid: CoverageGrid,
    turns_without_progress: int,
    backup_bursts: int,
    just_backed_up: bool,
    previous_target: Sequence[float] | None,
    learner: ExplorationLearner | None,
    hazards: Sequence[tuple[float, float | None]],
) -> ExploreStep:
    """Decide the next autonomous move toward the most promising frontier.

    The coverage grid holds everything every sweep has seen. The learner
    weighs how much unknown floor each frontier could reveal (a long reading
    off to one side leaves a lot) against the driving and turning needed to
    get there, using what earlier moves actually revealed. The car turns the
    short way toward the next waypoint and drives as far toward it as a
    straight line allows; the sonar sweep has the final say on driving.
    """
    learner = learner or ExplorationLearner()
    learner.observe(pose, grid.known_count())
    if turns_without_progress == 0:
        learner.last_turn = 0.0
    ahead = forward_clearance(scan)
    clearances = direction_clearances(scan)
    # Right after backing out of a dead end, never drive straight back in.
    ahead_clear = (
        ahead is not None
        and ahead >= AUTO_MIN_CLEARANCE_CM
        and not just_backed_up
        and not _hazard_within(hazards, 0.0)
    )
    can_turn = bool(clearances) and min(clearances.values()) >= AUTO_TURN_MIN_CLEARANCE_CM
    turning_allowed = can_turn and turns_without_progress < AUTO_MAX_TURNS_WITHOUT_PROGRESS
    if not can_turn and clearances:
        # Too near a wall to spin: slide sideways (the wheels are mecanum)
        # toward whichever side has more room, rather than back up blind.
        left = [value for angle, value in clearances.items() if angle >= AUTO_STRAFE_SIDE_DEGREES]
        right = [value for angle, value in clearances.items() if angle <= -AUTO_STRAFE_SIDE_DEGREES]
        room_left, room_right = min(left, default=0.0), min(right, default=0.0)
        room = max(room_left, room_right)
        if room >= AUTO_STRAFE_MIN_ROOM_CM:
            slide = AUTO_STRAFE_CM if room_right >= room_left else -AUTO_STRAFE_CM
            side = "right" if slide > 0 else "left"
            return ExploreStep("strafe", reason=f"too close to turn; sliding {AUTO_STRAFE_CM:.0f} cm {side}", distance_cm=slide)

    waypoint = grid.next_waypoint(pose, previous_target=previous_target, learner=learner)
    if waypoint is None:
        learner.finish(grid.known_count())
        if turning_allowed and turns_without_progress < AUTO_LOOK_AROUND_TURNS:
            return ExploreStep("turn", -AUTO_LOOK_TURN_DEGREES, "nothing worth a trip in view yet; turning to look")
        if can_turn or ahead_clear:
            return ExploreStep("done", reason="no unexplored edges left worth a trip")
        if backup_bursts < AUTO_MAX_BACKUP_BURSTS and not just_backed_up:
            return ExploreStep("reverse", reason="too close to turn; backing up once")
        return ExploreStep("blocked", reason="no safe path after looking all around")
    if not ahead_clear and abs(waypoint.bearing_deg) <= AUTO_BLOCKED_CONE_DEGREES:
        waypoint = (
            grid.next_waypoint(pose, avoid_within_deg=AUTO_BLOCKED_CONE_DEGREES, learner=learner) or waypoint
        )
    learner.pursue(waypoint.frontier, grid.known_count())
    bearing = waypoint.bearing_deg
    distance_to_waypoint = waypoint.distance_cm
    frontier = waypoint.frontier
    corridor = grid.corridor(pose)
    in_corridor = False
    if corridor is not None:
        # Heading for something along the corridor: follow its centre line
        # rather than picking a way between mapped cells.
        here = np.array(pose[:2], dtype=float)
        offset = np.array(frontier.target) - here
        toward = math.degrees(math.atan2(-offset[0], offset[1]))
        for axis in (corridor.axis_deg, corridor.axis_deg + 180.0):
            if abs((toward - axis + 180.0) % 360.0 - 180.0) <= AUTO_CORRIDOR_FOLLOW_DEGREES:
                centre = corridor.centre_offset_cm if axis == corridor.axis_deg else -corridor.centre_offset_cm
                aim = here + direction(axis) * AUTO_CORRIDOR_AIM_CM + direction(axis + 90.0) * centre
                aim_offset = aim - here
                bearing = (math.degrees(math.atan2(-aim_offset[0], aim_offset[1])) - float(pose[2]) + 180.0) % 360.0 - 180.0
                distance_to_waypoint = min(float(np.hypot(*offset)), AUTO_LONG_STEP_CM)
                in_corridor = True
                break
    target = frontier.target
    what = {
        "open": "open space",
        "doorway": f"a {frontier.width_cm:.0f} cm opening" if frontier.width_cm else "an opening",
        "edge": "an unexplored edge",
        "lead": "an opening the sonar saw into",
    }[frontier.kind]
    where = f"{what} {frontier.path_cm:.0f} cm away, could show {frontier.gain * CELL_AREA_M2:.1f} m²"
    if in_corridor:
        where = f"the {corridor.width_cm:.0f} cm corridor's centre line, toward {where}"

    if abs(bearing) > AUTO_HEADING_TOLERANCE_DEGREES and turning_allowed:
        limit = AUTO_MAX_TURN_PER_STOP_DEGREES
        degrees = max(-limit, min(limit, bearing))
        # Just lining up with a waypoint the map shows clear: no need for a
        # sweep in between, only a ping ahead before driving on.
        then = 0.0
        if degrees == bearing and distance_to_waypoint >= AUTO_MIN_STEP_CM + 10.0 and not _hazard_within(hazards, degrees):
            facing = (pose[0], pose[1], float(pose[2]) + degrees)
            then = min(AUTO_LONG_STEP_CM, distance_to_waypoint, grid.clear_run(facing, AUTO_LONG_STEP_CM))
            if then < AUTO_MIN_STEP_CM:
                then = 0.0
        reason = f"turning toward {where} ({bearing:+.0f}°)" + (f", then {then:.0f} cm" if then else "")
        learner.last_turn = 0.0 if then else degrees
        return ExploreStep("turn", degrees, reason, target, distance_cm=then)
    # The map has a say too: a wall met at a glancing angle never echoes.
    mapped_run = grid.clear_run(pose, AUTO_LONG_STEP_CM)
    if ahead_clear and mapped_run >= AUTO_MIN_STEP_CM:
        leg = min(AUTO_LONG_STEP_CM, distance_to_waypoint, ahead - AUTO_WALL_STANDOFF_CM, mapped_run)
        leg = max(AUTO_MIN_STEP_CM, leg)
        return ExploreStep("forward", reason=f"driving {leg:.0f} cm toward {where}", target=target, distance_cm=leg)
    if ahead_clear and turning_allowed:
        # The sonar says clear but the map shows a wall close ahead: face the
        # waypoint squarely, or if already facing it, head somewhere else.
        if abs(bearing) >= 3.0:
            return ExploreStep("turn", bearing, f"a mapped wall is close ahead; squaring up toward {where}", target)
        other = grid.next_waypoint(pose, avoid_within_deg=AUTO_BLOCKED_CONE_DEGREES, learner=learner)
        if other is not None:
            limit = AUTO_MAX_TURN_PER_STOP_DEGREES
            degrees = max(-limit, min(limit, other.bearing_deg))
            return ExploreStep("turn", degrees, "a mapped wall blocks the way; heading elsewhere", other.frontier.target)
    if turning_allowed:
        return ExploreStep("turn", -AUTO_LOOK_TURN_DEGREES, "way ahead blocked; turning to look", target)
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


class _HuntBody:
    """The car as the cat hunt sees it (robocar.hunt.HuntBody)."""

    def __init__(self, controller: "MappingController") -> None:
        self._controller = controller
        status = controller._pan_tilt.status()
        self._tilt = int(status["tilt"])
        self._pan = int(status["pan"])
        self.status: dict[str, object] = {}
        # When the car or head last moved, and whether the hunt is mid-action
        # (moving, sweeping, taking its own picture): the dashboard preview
        # only takes pictures while neither.
        self._lock = threading.Lock()
        self._still_since = time.monotonic()
        self._busy = 0
        # Settle delay before pictures, learned from how often they blur, and
        # a running reference for how sharp a still picture here looks.
        self.still_seconds = HUNT_STILL_SECONDS
        self._sharp_reference: float | None = None
        self.retakes = 0

    def now(self) -> float:
        return time.monotonic()

    @contextmanager
    def _acting(self, moves: bool = True):
        with self._lock:
            self._busy += 1
        try:
            yield
        finally:
            with self._lock:
                self._busy -= 1
                if moves:
                    self._still_since = time.monotonic()

    def idle_and_still(self, seconds: float) -> bool:
        with self._lock:
            return self._busy == 0 and time.monotonic() - self._still_since >= seconds

    def _wait_until_still(self) -> bool:
        """Let the car and head come to rest before a picture."""
        with self._lock:
            remaining = self.still_seconds - (time.monotonic() - self._still_since)
        if remaining > 0 and self._controller._stop_event.wait(remaining):
            return False
        self._controller._wait_until_steady(HUNT_STEADY_TIMEOUT_SECONDS)
        return not self._controller._stop_event.is_set()

    def wait(self, seconds: float) -> bool:
        return not self._controller._stop_event.wait(max(0.0, seconds))

    def stopped(self) -> bool:
        return self._controller._stop_event.is_set()

    def _point_head(self, pan: int) -> bool:
        controller = self._controller
        pan = max(PAN_LIMITS[0], min(PAN_LIMITS[1], int(pan)))
        if pan == self._pan:
            return True
        with self._acting():
            controller._pan_tilt.set_inhibited(True)
            try:
                controller._pan_tilt.set_angles(pan, self._tilt, override=True)
            finally:
                controller._pan_tilt.set_inhibited(False)
            turned = abs(pan - self._pan)
            self._pan = pan
            return not controller._stop_event.wait(HUNT_HEAD_SETTLE_SECONDS + HUNT_HEAD_SECONDS_PER_DEGREE * turned)

    def look(self, pan: int) -> Look:
        controller = self._controller
        if not self._point_head(pan):
            return Look(None)
        with self._acting(moves=False):
            if not self._wait_until_still():
                return Look(None)
            return self._look_now()

    def _sharp_picture(self, check: bool) -> np.ndarray | None:
        """A picture, retaken after a further wait while it looks blurred
        compared with recent ones; the sharpest one is kept."""
        controller = self._controller
        best: tuple[float, np.ndarray] | None = None
        first: float | None = None
        for shot in range(HUNT_MAX_SHOTS if check else 1):
            jpeg = controller._camera.capture_snapshot()
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) if jpeg else None
            if frame is None:
                break
            sharpness = picture_sharpness(frame)
            first = sharpness if first is None else first
            if best is None or sharpness > best[0]:
                best = (sharpness, frame)
            reference = self._sharp_reference
            if not check or reference is None or sharpness >= HUNT_SOFT_FRACTION * reference:
                break
            # Soft: give the car longer to settle, then try again.
            self.retakes += 1
            if controller._stop_event.wait(0.5 * self.still_seconds + 0.1 * (shot + 1)):
                break
        if best is None:
            return None
        sharpness, frame = best
        if check and first is not None:
            if sharpness >= HUNT_BLUR_CONFIRM_RATIO * first:
                # The retake was clearly sharper: the first one was motion blur.
                self.still_seconds = min(HUNT_STILL_LIMITS_SECONDS[1], self.still_seconds * 1.25 + 0.05)
            elif sharpness == first:
                self.still_seconds = max(HUNT_STILL_LIMITS_SECONDS[0], self.still_seconds * 0.97)
            # Follow the scene: the reference drifts toward what still pictures here look like.
            self._sharp_reference = sharpness if self._sharp_reference is None else 0.8 * self._sharp_reference + 0.2 * sharpness
        return frame

    def _look_now(self, check: bool = True) -> Look:
        controller = self._controller
        frame = self._sharp_picture(check)
        if frame is None:
            return Look(None)
        try:
            detections = controller._detector.detect(frame) if controller._detector is not None else []
        except Exception:
            logger.exception("Detection failed during the hunt")
            detections = []
        look = cat_view_from(detections, self._pan, frame.shape[1], frame.shape[0])
        controller._publish_hunt_frame(frame, detections, self.status)
        return look

    def ping(self, pan: int) -> float | None:
        controller = self._controller
        if not self._point_head(pan):
            return None
        with self._acting(moves=False):
            return self._ping_now()

    def _ping_now(self) -> float | None:
        controller = self._controller
        if controller._stop_event.wait(max(0.0, controller._range_sensor.settle_seconds - HUNT_HEAD_SETTLE_SECONDS)):
            return None
        readings = []
        for _ in range(3):
            reading = controller._range_sensor.sample_now()
            if reading is not None:
                readings.append(reading)
            if controller._stop_event.wait(SCAN_SAMPLE_INTERVAL_SECONDS):
                return None
        return statistics.median(readings) if readings else None

    def observe(self) -> None:
        controller = self._controller
        if not controller._drive.set_inhibited(True):
            raise RuntimeError("Drive was not stopped before a range scan")
        try:
            with self._acting():
                controller._capture_observation(mode="hunt")
        finally:
            controller._drive.set_inhibited(False)
        self._pan = int(controller._pan_tilt.status()["pan"])

    def pose(self) -> Pose:
        controller = self._controller
        with controller._lock:
            pose, last = controller._pose, controller._last_odometry
        return compose(pose, relative(last, controller._drive.odometry())) if last else pose

    def odometry(self) -> Pose:
        return self._controller._drive.odometry()

    def grid(self) -> CoverageGrid:
        return self._controller._grid

    def landmarks(self) -> list[tuple[str, float, float]]:
        with self._controller._map_lock:
            return [(landmark.label, landmark.x, landmark.y) for landmark in self._controller._semantic.confirmed()]

    def turn(self, degrees: float, precise: bool) -> bool:
        controller = self._controller
        if abs(degrees) < 1.0:
            return not self.stopped()
        with self._acting():
            if precise:
                pieces = max(1, math.ceil(abs(degrees) / AUTO_TURN_PIECE_DEGREES - 1e-9))
                return all(controller._turn_and_align(degrees / pieces) for _ in range(pieces))
            return controller._turn_by(degrees)

    def move(self, forward_cm: float, sideways_cm: float, speed: float = 1.0) -> bool:
        distance = math.hypot(forward_cm, sideways_cm)
        if distance < 1.0:
            return not self.stopped()
        controller = self._controller
        speed = max(0.3, min(1.0, speed))
        if abs(sideways_cm) <= abs(forward_cm) * 0.05:
            seconds = controller._step_seconds(distance)
        else:
            # Sideways travel is slower and has no calibration of its own.
            seconds = AUTO_FORWARD_SECONDS * distance / AUTO_STEP_CM * (1.0 + 0.3 * abs(sideways_cm) / distance)
        throttle = AUTO_FORWARD_COMMAND * speed * forward_cm / distance
        strafe = AUTO_FORWARD_COMMAND * speed * sideways_cm / distance
        with self._acting():
            return controller._move_burst(throttle, seconds / speed, strafe=strafe)

    def arc(self, strafe: float, steering: float, seconds: float) -> bool:
        with self._acting():
            return self._controller._move_burst(0.0, seconds, steering=steering, strafe=strafe)

    def preview(self) -> None:
        """A picture just for the dashboard, taken only while the car is
        still and the hunt is not using the camera."""
        if not self.idle_and_still(HUNT_STILL_SECONDS):
            return
        with self._acting(moves=False):
            self._look_now(check=False)

    def battery_ok(self) -> bool:
        level = self._controller.battery_level() if self._controller.battery_level else None
        return level != "critical"

    def report(self, status: dict[str, object]) -> None:
        self.status = status
        controller = self._controller
        with controller._lock:
            controller._hunt_status = status
            controller._hunt_camera = {"settle_seconds": round(self.still_seconds, 2), "retakes": self.retakes}
            controller._plan = str(status.get("action") or "")
            if controller._enabled:
                controller._state = f"hunt_{status.get('phase')}"


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
        # Offsets are counterclockwise of where the head faces.
        pans = [self._centre - offset for offset in pan_offsets if PAN_LIMITS[0] <= self._centre - offset <= PAN_LIMITS[1]]
        views = self._controller._capture_views(pans, self._tilt, self._centre, frames=1 if len(pans) > 1 else AUTO_ALIGN_FRAMES)
        return {self._centre - pan: frames for pan, frames in views.items()}

    def measure_turn(self, strip: dict[int, tuple[bytes, ...]], current: tuple[bytes, ...] | None) -> float | None:
        return self._controller._measure_turn({self._centre - offset: frames for offset, frames in strip.items()}, current, self._centre)

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
        detector: "ObjectDetectorLike | None" = None,
    ) -> None:
        self._camera = camera
        # Recognizes objects in each stop's picture (on the AI HAT+ when present).
        self._detector = detector
        self._semantic = SemanticMap()
        # People and animals seen at the latest stop, as (bearing, range).
        self._hazards: list[tuple[float, float | None]] = []
        # Turn-strip pictures from the latest sweep, until a turn uses them.
        self._sweep_strip: SweepStrip | None = None
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
        # What exploring has learned (how much each kind of frontier really
        # shows, and what effort buys), kept across runs beside the database.
        self._learner_path = (
            store_path.parent / "exploration.json" if store_path is not None and str(store_path) != ":memory:" else None
        )
        self._learner = load_learner(self._learner_path)
        # The cat hunt: what it has learned, its latest status and camera frame.
        self._hunt_memory_path = (
            store_path.parent / "hunt.json" if store_path is not None and str(store_path) != ":memory:" else None
        )
        self._hunt_memory = load_memory(self._hunt_memory_path)
        self._hunt_status: dict[str, object] | None = None
        self._hunt_frame: bytes | None = None
        self._hunt_frame_at: float = 0.0
        # The hunt camera's learned settle delay and how many pictures it retook.
        self._hunt_camera: dict[str, float] = {}
        # Set by the web app: "ok", "getting_low", "low", "critical" or None.
        self.battery_level: Callable[[], str | None] | None = None
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
        self._joining_run = False
        # Stops since the pose was last pinned to the map (a sonar sweep
        # matched to mapped walls, or a recognized place); None if never.
        self._stops_since_fix: int | None = None

    def start(self) -> None:
        self._start("survey", self._run_survey)

    def start_autonomous(self, attended: bool) -> None:
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before exploring")
        drive_status = self._drive.status()
        if drive_status.get("fault"):
            raise RuntimeError(f"Motor board unavailable: {drive_status['fault']}")
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

    def start_hunt(self, attended: bool) -> None:
        """Hunt for the cat in the mapped room and play with it (attended)."""
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before the hunt")
        drive_status = self._drive.status()
        if drive_status.get("fault"):
            raise RuntimeError(f"Motor board unavailable: {drive_status['fault']}")
        if drive_status["mode"] != "hardware":
            raise RuntimeError("The hunt requires hardware mode")
        if any(abs(float(drive_status[axis])) > 0.01 for axis in ("throttle", "strafe", "steering")):
            raise RuntimeError("Stop the car before starting the hunt")
        if not self._range_sensor.status()["enabled"]:
            raise RuntimeError("The hunt requires an enabled ultrasonic sensor")
        if self._detector is None or not getattr(self._detector, "available", True):
            raise RuntimeError("The hunt needs object detection to spot the cat")
        if self._store.count() == 0:
            raise RuntimeError("Map the room first: the hunt plans its search on the map")
        self._start("hunt", self._run_hunt)

    def _run_hunt(self) -> None:
        self._hold_camera(True)
        body = _HuntBody(self)
        preview_done = threading.Event()

        def preview() -> None:
            # Keep the dashboard's picture fresh between the hunt's own looks.
            while not preview_done.wait(HUNT_PREVIEW_SECONDS):
                with self._lock:
                    fresh = time.time() - self._hunt_frame_at < HUNT_PREVIEW_SECONDS
                if fresh:
                    continue
                try:
                    body.preview()
                except Exception:
                    logger.debug("Hunt preview picture failed", exc_info=True)

        previewer = threading.Thread(target=preview, name="hunt-preview", daemon=True)
        previewer.start()
        try:
            self._drive.set_inhibited(False)
            engine = HuntEngine(body, self._hunt_memory)
            reason = engine.run()
            with self._lock:
                self._state = f"hunt_{reason}"
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.exception("The hunt stopped after an error")
                with self._lock:
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
        finally:
            preview_done.set()
            previewer.join(timeout=5)
            self._drive.stop()
            try:
                save_memory(self._hunt_memory_path, self._hunt_memory)
            except OSError as exc:
                logger.warning("Could not save what the hunt learned: %s", exc)
            self._set_enabled_false()
            self._hold_camera(False)

    def _publish_hunt_frame(self, frame: np.ndarray, detections: list[dict[str, object]], status: dict[str, object]) -> None:
        """The hunt's latest picture for the dashboard: what it detected, and
        what it is up to."""
        try:
            image = frame.copy()
            for detection in detections:
                x, y, width, height = (int(value) for value in detection["box"])
                cat = detection.get("label") == "cat"
                colour = (60, 170, 255) if cat else (45, 220, 90)
                cv2.rectangle(image, (x, y), (x + width, y + height), colour, 3 if cat else 2)
                cv2.putText(image, f"{detection['label']} {float(detection['confidence']):.0%}", (x, max(y - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
            phase = str(status.get("phase") or "").upper()
            action = str(status.get("action") or "")[:70]
            cv2.rectangle(image, (0, image.shape[0] - 44), (image.shape[1], image.shape[0]), (16, 24, 32), -1)
            cv2.putText(image, phase, (8, image.shape[0] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (93, 179, 242), 2, cv2.LINE_AA)
            cv2.putText(image, action, (8, image.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 243, 247), 1, cv2.LINE_AA)
            encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if encoded:
                with self._lock:
                    self._hunt_frame = jpeg.tobytes()
                    self._hunt_frame_at = time.time()
        except Exception:
            logger.debug("Could not draw the hunt frame", exc_info=True)

    def hunt_frame(self) -> bytes | None:
        with self._lock:
            return self._hunt_frame

    def hunt_status(self) -> dict[str, object]:
        with self._lock:
            running = self._enabled and self._mode == "hunt"
            status = dict(self._hunt_status or {})
            frame_at = self._hunt_frame_at
        return {
            "running": running,
            "status": status or None,
            "frame_at": frame_at or None,
            "frame_age_seconds": round(time.time() - frame_at, 1) if frame_at else None,
            "camera": dict(self._hunt_camera),
            "learned": {
                "favourites": self._hunt_memory.favourites(5),
                "hunts": self._hunt_memory.hunts,
                "finds": self._hunt_memory.finds,
                "cat_spots": [[round(v, 1) for v in spot] for spot in self._hunt_memory.cat_spots[:8]],
            },
        }

    def start_calibration(self, attended: bool) -> None:
        """Measure real turns and steps (car facing a flat wall about 1-1.5 m away)."""
        if attended is not True:
            raise ValueError("Attended supervision must be confirmed before calibrating")
        drive_status = self._drive.status()
        if drive_status.get("fault"):
            raise RuntimeError(f"Motor board unavailable: {drive_status['fault']}")
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
            self._last_place = None
            with self._map_lock:
                self._explore_target = None
            self._stop_event.clear()
            run_id = self._run_id

            def run() -> None:
                # Calibration only measures motion; mapping runs build on
                # everything mapped so far rather than starting over.
                if mode != "calibrate":
                    self._continue_map(run_id)
                target()

            self._thread = threading.Thread(
                target=run,
                name=f"mapping-{mode}",
                daemon=True,
            )
            self._thread.start()

    def _continue_map(self, run_id: str) -> None:
        """Start a run from the newest stop of any earlier run, with its map."""
        with self._places_lock:
            last = max(self._places.graph.nodes, default=None)
        if last is None:
            with self._lock:
                self._reset_localization()
            with self._map_lock:
                self._surface_map = SurfaceMap()
                self._grid = CoverageGrid()
                self._semantic = SemanticMap()
                self._path, self._history, self._links = [], [], []
                self._map_run_id = run_id
            return
        self._rebuild_map(last, run_id)
        with self._places_lock:
            pose = self._places.graph.nodes[last]
        with self._lock:
            # Wheel odometry since that stop, the previous views and the
            # learned scales carry over when the service has kept running.
            self._pose = pose
            self._pose_covariance = np.diag(
                [CONTINUE_POSITION_SIGMA_CM**2, CONTINUE_POSITION_SIGMA_CM**2, math.radians(CONTINUE_HEADING_SIGMA_DEG) ** 2]
            )
            self._last_node = last
            self._joining_run = True
            self._last_localization = None
            self._verified_turn = None

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
                "exploration_learning": self._learner.to_json(),
                "hazards": [
                    {"bearing_deg": round(bearing, 1), "range_cm": None if distance is None else round(distance, 1)}
                    for bearing, distance in self._hazards
                ],
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
            status["landmark_count"] = len(self._semantic.confirmed())
        detector_info = getattr(self._detector, "info", None)
        status["detector"] = detector_info() if callable(detector_info) else None
        status["localized"] = self.localized()
        return status

    def localized(self) -> dict[str, object]:
        """Whether the car believes it knows roughly where it is and which way
        it faces on the map, and why (not)."""
        with self._lock:
            position_sigma, heading_sigma = pose_sigma(self._pose_covariance)
            since = self._stops_since_fix
            last_odometry = self._last_odometry
        with self._map_lock:
            walls = len(self._surface_map.walls())
        moved = turned = 0.0
        if last_odometry is not None:
            change = relative(last_odometry, self._drive.odometry())
            moved, turned = math.hypot(change[0], change[1]), abs(wrap_degrees(change[2]))
        reasons = []
        if walls < LOCALIZED_MIN_WALLS:
            reasons.append("no map to match against yet")
        elif since is None:
            reasons.append("has not matched itself to the map yet")
        elif since > LOCALIZED_MAX_STOPS_SINCE_FIX:
            reasons.append(f"no map match at the last {since} stops")
        if position_sigma > LOCALIZED_MAX_SIGMA_CM:
            reasons.append(f"position uncertain (±{position_sigma:.0f} cm)")
        if heading_sigma > LOCALIZED_MAX_HEADING_SIGMA_DEG:
            reasons.append(f"heading uncertain (±{heading_sigma:.0f}°)")
        if moved > LOCALIZED_MAX_DEAD_RECKONING_CM:
            reasons.append(f"drove {moved / 100:.1f} m on wheel counts since its last check")
        if turned > LOCALIZED_MAX_DEAD_RECKONING_DEG:
            reasons.append(f"turned {turned:.0f}° on wheel counts since its last check")
        if reasons:
            detail = "; ".join(reasons)
        else:
            when = "at the last stop" if since == 0 else f"{since} stops ago"
            detail = f"matched to the map {when}: ±{position_sigma:.0f} cm, ±{heading_sigma:.0f}°"
        return {
            "known": not reasons,
            "detail": detail,
            "sigma_cm": round(position_sigma, 1),
            "heading_sigma_deg": round(heading_sigma, 1),
            "stops_since_fix": since,
        }

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
            data.update(self._semantic.to_json(self._grid.wall_between))
        with self._lock:
            last_cat = (self._hunt_status or {}).get("last_cat_position")
        data["cat"] = {
            "last_seen": last_cat,
            "spots": [[round(v, 1) for v in spot] for spot in self._hunt_memory.cat_spots[:8]],
        }
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
                self._semantic = SemanticMap()
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
        semantic = SemanticMap()
        path: list[Pose] = []
        history: dict[str, list[Pose]] = {}
        report = None
        for other, stop, pose in stops:
            report = surface_map.integrate(stop.scan, pose, *stop.pose_sigma)
            grid.integrate(stop.scan, pose, surface_map.beam_width_deg)
            semantic.add(stop.sightings, pose)
            if stop.run_id == run_id:
                path.append(pose)
            else:
                history.setdefault(stop.run_id, []).append(pose)
        with self._map_lock:
            self._surface_map = surface_map
            self._grid = grid
            self._semantic = semantic
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
        self._learner.start_run()
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
                    hazards = list(self._hazards)
                with self._map_lock:
                    step = plan_explore_step(
                        scan,
                        pose,
                        self._grid,
                        turns_without_progress,
                        backup_bursts,
                        just_backed_up,
                        self._explore_target,
                        self._learner,
                        hazards,
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
                    if not self._move_burst(AUTO_FORWARD_COMMAND, self._step_seconds(step.distance_cm)):
                        break
                    turns_without_progress = 0
                    just_backed_up = False
                elif step.action == "turn":
                    self._set_state("turning")
                    pieces = max(1, math.ceil(abs(step.degrees) / AUTO_TURN_PIECE_DEGREES - 1e-9))
                    if not all(self._turn_and_align(step.degrees / pieces) for _ in range(pieces)):
                        break
                    turns_without_progress += 1
                    # Only a change of direction earns another backing-up.
                    backup_bursts = 0
                    just_backed_up = False
                    if step.distance_cm > 0:
                        # Lined up with a waypoint the map shows clear: one
                        # ping straight ahead must agree before driving on.
                        ahead = self._ping_ahead()
                        if ahead is not None and ahead - step.distance_cm >= AUTO_WALL_STANDOFF_CM:
                            self._set_state("moving")
                            if not self._move_burst(AUTO_FORWARD_COMMAND, self._step_seconds(step.distance_cm)):
                                break
                            turns_without_progress = 0
                elif step.action == "strafe":
                    self._set_state("sliding")
                    seconds = AUTO_FORWARD_SECONDS * abs(step.distance_cm) / AUTO_STEP_CM
                    if not self._move_burst(0, seconds, strafe=math.copysign(AUTO_FORWARD_COMMAND, step.distance_cm)):
                        break
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
            with self._map_lock:
                self._learner.finish(self._grid.known_count())
            try:
                save_learner(self._learner_path, self._learner)
            except OSError as exc:
                logger.warning("Could not save exploration learning: %s", exc)

    def _ping_ahead(self) -> float | None:
        """Median of a few sonar readings straight ahead, drive locked meanwhile."""
        if not self._drive.set_inhibited(True):
            return None
        status = self._pan_tilt.status()
        centre, tilt = int(status["pan"]), int(status["tilt"])
        self._pan_tilt.set_inhibited(True)
        try:
            self._pan_tilt.set_angles(centre, tilt, override=True)
            readings = []
            for _ in range(3):
                if self._stop_event.wait(max(0.06, self._range_sensor.settle_seconds)):
                    return None
                reading = self._range_sensor.sample_now()
                if reading is not None:
                    readings.append(reading)
            return statistics.median(readings) if readings else None
        finally:
            self._pan_tilt.set_inhibited(False)
            self._drive.set_inhibited(False)

    def _step_seconds(self, distance_cm: float = AUTO_STEP_CM) -> float:
        """Burst length for an exploration step, from calibration when there is one."""
        with self._lock:
            calibration = self._calibration
        measured = calibration.step_duration(distance_cm) if calibration else None
        # Uncalibrated, scale the basic step; ramps make longer bursts go a
        # little further than this, which errs short.
        return measured if measured is not None else AUTO_FORWARD_SECONDS * distance_cm / AUTO_STEP_CM

    def _turn_and_align(self, degrees: float) -> bool:
        """Turn by `degrees`, then use the camera to measure and correct the turn."""
        pan_status = self._pan_tilt.status()
        centre_pan, tilt = int(pan_status["pan"]), int(pan_status["tilt"])
        sign = 1 if degrees >= 0 else -1
        # Photograph toward where the car is turning (lower pans look left).
        pans = [
            centre_pan - sign * step
            for step in range(0, AUTO_STRIP_REACH_DEGREES + 1, AUTO_STRIP_STEP_DEGREES)
            if PAN_LIMITS[0] <= centre_pan - sign * step <= PAN_LIMITS[1]
        ]
        strip = self._strip_from_sweep(pans, centre_pan, tilt)
        if strip is None:
            strip = self._capture_views(pans, tilt, centre_pan, frames=1)
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
        best_pan = min(strip, key=lambda pan: abs(centre_pan - pan - aim))
        lines = [
            f"planned {planned:+.0f} deg, camera measured "
            + (f"{actual:+.1f} deg" if actual is not None else "nothing usable (wheels trusted)"),
            f"left: strip picture at {centre_pan - best_pan:+d} deg; right: view after the turn"
            + (f"; {corrections} correction(s)" if corrections else ""),
        ]
        self._set_vision("turn", "Turn check", lines, lambda: render_matches(strip[best_pan][0], current[0], "Turn check", lines))

    def _recognize(self, image_jpeg: bytes | None, camera_bearing: float, scan: list[dict[str, object]]) -> list[Sighting]:
        """Objects in a stop's picture, placed by bearing and sonar range.
        Recognition is a bonus: any trouble with it leaves the stop without."""
        if self._detector is None or not image_jpeg:
            return []
        try:
            frame = cv2.imdecode(np.frombuffer(image_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return []
            detections = self._detector.detect(frame)
            sightings = sightings_from(detections, frame.shape[1], camera_bearing, scan)
        except Exception:
            logger.exception("Object recognition failed at this stop; carrying on without it")
            return []
        if detections:
            drawn = [
                {**detection, "dynamic": detection.get("label") in {s.label for s in sightings if s.dynamic}}
                for detection in detections
            ]
            lines = [
                ", ".join(
                    f"{s.label} {s.bearing_deg:+.0f} deg" + (f" {s.range_cm:.0f} cm" if s.range_cm is not None else "")
                    for s in sightings[:4]
                )
                or "nothing that matters for the map",
            ]
            if any(s.dynamic for s in sightings):
                lines.append("people/animals (orange) are kept out of the map")
            self._set_vision(
                "objects", "What the camera recognized", lines, lambda: render_detections(image_jpeg, drawn, "What the camera recognized", lines)
            )
        return sightings

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
        earlier: np.ndarray | None = None
        while time.monotonic() < deadline:
            if self._stop_event.wait(CAMERA_STEADY_INTERVAL_SECONDS):
                return False
            current = _steadiness_frame(self._camera.capture_snapshot())
            if current is None:
                return True
            change = float(np.abs(current - previous).mean())
            if change < CAMERA_STEADY_DIFFERENCE:
                return True
            # In dim light sensor noise alone exceeds the threshold. Noise does
            # not build up over two frames the way movement does, so a change
            # from two frames back no bigger than from one frame back means
            # the view is still and only noisy.
            if earlier is not None:
                two_back = float(np.abs(current - earlier).mean())
                if two_back <= CAMERA_NOISE_GROWTH * change:
                    return True
            earlier, previous = previous, current
        logger.info("Camera still changing after %.1f s; taking the picture anyway", timeout)
        return False

    def _capture_best_picture(
        self, steady_timeout: float = CAMERA_STEADY_TIMEOUT_SECONDS, *, already_steady: bool = False
    ) -> tuple[bytes | None, dict[str, object] | None]:
        """Take a picture, check its levels, and retake with adjusted exposure
        until it is decent or the settings run out; keep the best one.

        `already_steady` skips the wait for matching frames first (the caller
        has held everything still long enough); a blurred picture is still
        retaken after waiting for steadiness."""
        set_exposure = getattr(self._camera, "set_exposure", None)
        if set_exposure is None:
            return self._camera.capture_snapshot(), None
        with self._lock:
            index = self._exposure_index
        set_exposure(EXPOSURE_LADDER[index])
        if not already_steady:
            self._wait_until_steady(steady_timeout)
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
                self._wait_until_steady(steady_timeout)
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
        """Combine the turn each strip picture implies: its direction relative
        to the centre view plus the rotation still left between it and the
        current forward view."""
        if not current:
            return None
        readings: list[tuple[float, float]] = []
        for pan, frames in strip.items():
            key = (pan, 0)
            visual = combine_view_estimates(estimate_view_motion({key: frames}, {key: current}))
            if visual is not None and abs(visual.yaw_deg) <= AUTO_STRIP_MAX_YAW_DEGREES:
                # Pictures the car ended up nearly facing overlap most.
                offset = pan_bearing(pan) - pan_bearing(centre_pan)
                readings.append((offset + visual.yaw_deg, 1.0 / (1.0 + abs(visual.yaw_deg))))
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
        from below so servo backlash is the same for every picture.

        Each picture gets the same exposure check as a stop's picture: retaken
        brighter, darker or steadier until decent. It starts from the setting
        that last worked, so in even light that is usually one shot, and any
        extra frames reuse the setting chosen."""
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
                first, _ = self._capture_best_picture(AUTO_VIEW_STEADY_TIMEOUT_SECONDS)
                pictures = [first] + [self._camera.capture_snapshot() for _ in range(frames - 1)]
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
        strafe: float = 0,
    ) -> bool:
        if self._stop_event.is_set():
            return False
        try:
            self._drive.set_command(throttle, strafe, steering)
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
                    self._drive.set_command(throttle, strafe, steering)
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
        # A strip of pictures across the head's range, taken on the way past
        # during the sweep: a turn while exploring may reuse it, and the
        # dashboard shows it as a panorama.
        strip_pans = self._sweep_strip_pans(original_pan) if mode in ("explore", "survey") else []
        scan, side_views = self._sweep(original_tilt, original_pan, side_pans, strip_pans)
        if self._sweep_strip is not None:
            # The car stands still for a stop, so the odometry read for it holds.
            self._sweep_strip = replace(self._sweep_strip, odometry=odometry)
            self._show_panorama(self._sweep_strip)
        if scan is None:
            return []
        # What the camera recognizes: people and animals are kept out of the
        # map (mapping_scan), furniture becomes landmarks.
        sightings = self._recognize(image_jpeg, pan_bearing(original_pan), scan)
        scan = mark_dynamic(scan, sightings)
        mapping_scan = static_scan(scan)
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
            echoes = [echo for run in extract_echoes(mapping_scan, self._surface_map.beam_width_deg) for echo in run]
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
            "detections": [sighting.to_json() for sighting in sightings],
        }
        with self._lock:
            previous_node = self._last_node
            joining, self._joining_run = self._joining_run, False
        edge = None
        if previous_node is not None:
            sigma = self._edge_sigma(prediction, registration.accepted)
            if joining:
                sigma = (
                    max(sigma[0], CONTINUE_POSITION_SIGMA_CM),
                    max(sigma[1], CONTINUE_POSITION_SIGMA_CM),
                    max(sigma[2], CONTINUE_HEADING_SIGMA_DEG),
                )
            edge = (previous_node, relative(previous_pose, pose), sigma)
            estimate["edge"] = {
                "from": previous_node,
                "delta": [round(value, 3) for value in edge[1]],
                "sigma": [round(value, 3) for value in edge[2]],
            }
        observation_id = self._store.save(image_jpeg, scan, run_id, mode, pose, estimate)

        # Remember this place and try to recognize it from earlier stops.
        keyframe = make_keyframe(observation_id, run_id, image_jpeg, pan_bearing(original_pan))
        if keyframe is not None:
            self._store.save_keyframe(keyframe)
        with self._places_lock:
            update = self._places.add_stop(
                observation_id, run_id, pose, scan, (position_sigma, heading_sigma), keyframe, edge, sightings
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
                report = self._surface_map.integrate(mapping_scan, pose, position_sigma, heading_sigma)
                self._grid.integrate(mapping_scan, pose, self._surface_map.beam_width_deg)
                self._semantic.add(sightings, pose)
                self._path.append(pose)
        captured_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._pose, self._pose_covariance = pose, covariance
            if registration.accepted or event is not None:
                self._stops_since_fix = 0
            elif self._stops_since_fix is not None:
                self._stops_since_fix += 1
            self._hazards = hazards_from(sightings)
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
        strip_pans: Sequence[int] = (),
    ) -> tuple[list[dict[str, object]] | None, dict[ViewKey, tuple[bytes, ...]]]:
        """Sweep the ultrasonic sensor, grabbing camera views at the requested
        pans and turn-strip pictures at `strip_pans` (kept in self._sweep_strip)."""
        scan: list[dict[str, object]] = []
        views: dict[ViewKey, tuple[bytes, ...]] = {}
        strip: dict[int, tuple[bytes, ...]] = {}
        self._sweep_strip = None
        sensor_enabled = bool(self._range_sensor.status()["enabled"])
        pans = sorted(set(view_pans) | set(strip_pans) | (set(SCAN_ANGLES) if sensor_enabled else set()))
        # Side views are compared with the previous stop's, so they keep the
        # stop's exposure even if a strip picture needed another one.
        with self._lock:
            stop_exposure = self._exposure_index
        set_exposure = getattr(self._camera, "set_exposure", None)
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
                if pan in strip_pans:
                    # The same picture checks as a strip taken in a pass of
                    # its own, so turns are measured as well. The head has
                    # only stepped 3 degrees and been held still at least as
                    # long as that pass would, so the up-front wait for
                    # matching frames is skipped (blur still forces a retake).
                    if self._stop_event.wait(max(0.0, AUTO_ALIGN_SETTLE_SECONDS - settle)):
                        return None, views
                    picture, _ = self._capture_best_picture(AUTO_VIEW_STEADY_TIMEOUT_SECONDS, already_steady=True)
                    if picture is not None:
                        strip[pan] = (picture,)
                if pan in view_pans:
                    if strip and set_exposure is not None:
                        set_exposure(EXPOSURE_LADDER[stop_exposure])
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
                        "angle_deg": int(pan_bearing(pan)),
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
        if strip:
            # Where the car stood is filled in by the caller, which has read it.
            self._sweep_strip = SweepStrip(strip, original_pan, tilt, (math.nan, math.nan, math.nan))
        return scan, views

    def _sweep_strip_pans(self, centre_pan: int) -> list[int]:
        """Pans for turn-strip pictures during a sweep: the strip's every 10
        degrees both sides, moved to the sweep's nearest step."""
        stops = set(SCAN_ANGLES) | {SCAN_LEAD_IN_PAN} if self._range_sensor.status()["enabled"] else set()
        pans = set()
        for offset in range(-AUTO_STRIP_REACH_DEGREES, AUTO_STRIP_REACH_DEGREES + 1, AUTO_STRIP_STEP_DEGREES):
            wanted = centre_pan + offset
            if not PAN_LIMITS[0] <= wanted <= PAN_LIMITS[1]:
                continue
            nearest = min(stops, key=lambda pan: abs(pan - wanted), default=wanted)
            pans.add(nearest if abs(nearest - wanted) <= SWEEP_STRIP_SNAP_DEG else wanted)
        return sorted(pans)

    def _show_panorama(self, strip: SweepStrip) -> None:
        views = [(pan_bearing(pan), frames[0]) for pan, frames in strip.pictures.items() if frames]
        if len(views) < 2:
            return
        widest = max(bearing for bearing, _ in views) + PANORAMA_HFOV_DEGREES / 2
        lines = [f"{len(views)} pictures taken during the sonar sweep, joined: about {widest:.0f} deg each side of ahead"]
        self._set_vision(
            "panorama",
            "Sweep panorama",
            lines,
            lambda: render_panorama(views, PANORAMA_HFOV_DEGREES, "Sweep panorama", lines),
        )

    def _strip_from_sweep(self, pans: Sequence[int], centre_pan: int, tilt: int) -> dict[int, tuple[bytes, ...]] | None:
        """The sweep's pictures nearest each of `pans`, if every one was taken
        and the car and head have not moved since. Used once: the turn moves
        the car."""
        sweep_strip, self._sweep_strip = self._sweep_strip, None
        if sweep_strip is None or sweep_strip.centre_pan != centre_pan or sweep_strip.tilt != tilt:
            return None
        moved = relative(sweep_strip.odometry, self._drive.odometry())
        if math.hypot(moved[0], moved[1]) > SWEEP_STRIP_MAX_MOVE_CM or abs(moved[2]) > SWEEP_STRIP_MAX_TURN_DEG:
            return None
        strip: dict[int, tuple[bytes, ...]] = {}
        for wanted in pans:
            nearest = min(sweep_strip.pictures, key=lambda pan: abs(pan - wanted))
            if abs(nearest - wanted) > SWEEP_STRIP_SNAP_DEG:
                return None
            strip[nearest] = sweep_strip.pictures[nearest]
        return strip

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
