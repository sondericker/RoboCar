"""Judging and comparing the cameras' pictures.

- **Exposure:** the ladder of settings a stop's picture climbs until it is
  decent, and how a picture is judged (brightness, clipping, sharpness,
  usable detail).
- **Turns between views:** how far the car turned (and which way it went)
  between two pictures, from matched features, with the checks that keep
  dim, featureless pictures from producing confident nonsense.
"""

import logging
import math
import statistics
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from robocar.drive import pan_bearing
from robocar.localization import VisualMotion
from robocar.places import CAMERA_HORIZONTAL_FOV_DEGREES, detect_features

logger = logging.getLogger(__name__)

# Exposure settings tried for each stop's picture, darkest to brightest:
# auto-exposure with compensation, then long manual exposures with gain.
EXPOSURE_LADDER = (
    {"ev": -1.0},
    {"ev": 0.0},
    {"ev": 1.0},
    {"ev": 2.0},
    {"ev": 3.0},
    {"ev": 4.0},
    # A 90 ms exposure right after a move once came out smeared, but stop
    # pictures now wait for the view to hold still first (and a blurred one
    # is retaken). The OV5647 in the workshop at night needed all of these:
    # every stop of the 10-31 runs sat at 50 ms, gain 12, still dark.
    {"exposure_us": 33_000, "gain": 8.0},
    {"exposure_us": 50_000, "gain": 12.0},
    {"exposure_us": 66_000, "gain": 16.0},
    {"exposure_us": 95_000, "gain": 16.0},
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

# Dim real pictures produced confident-looking nonsense (136 degree "turns"
# while driving straight) from a handful of matches; demand real support.
CAMERA_MIN_MATCHES = 25
CAMERA_MIN_INLIERS = 20
CAMERA_MIN_INLIER_FRACTION = 0.4
CAMERA_MAX_PLAUSIBLE_YAW_DEGREES = 55.0


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
