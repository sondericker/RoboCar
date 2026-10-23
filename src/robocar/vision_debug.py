"""Pictures for the dashboard showing what the camera saw and how views matched."""

import cv2
import numpy as np

from robocar.places import camera_matrix, detect_features

PANEL_WIDTH = 320
CAPTION_LINE_HEIGHT = 16
MAX_KEPT_LINES = 50
MAX_REJECTED_LINES = 20


def _thumbnail(jpeg: bytes) -> tuple[np.ndarray, float] | None:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    scale = PANEL_WIDTH / image.shape[1]
    return cv2.resize(image, (PANEL_WIDTH, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA), scale


def _with_caption(image: np.ndarray, title: str, lines: list[str]) -> bytes:
    rows = [title, *lines]
    bar = np.full((CAPTION_LINE_HEIGHT * len(rows) + 8, image.shape[1], 3), 24, dtype=np.uint8)
    for index, text in enumerate(rows):
        cv2.putText(
            bar,
            text,
            (6, CAPTION_LINE_HEIGHT * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (236, 243, 247) if index == 0 else (169, 232, 215),
            1,
            cv2.LINE_AA,
        )
    encoded, jpeg = cv2.imencode(".jpg", np.vstack([image, bar]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes() if encoded else b""


def render_features(jpeg: bytes, title: str, lines: list[str]) -> bytes | None:
    """The picture with the corners feature matching will use."""
    thumbnail = _thumbnail(jpeg)
    detected = detect_features(jpeg, 1000)
    if thumbnail is None or detected is None:
        return None
    image, scale = thumbnail
    for keypoint in detected[0]:
        x, y = keypoint.pt
        cv2.circle(image, (round(x * scale), round(y * scale)), 2, (93, 179, 242), 1, cv2.LINE_AA)
    return _with_caption(image, title, [f"{len(detected[0])} features", *lines])


def render_matches(first_jpeg: bytes, second_jpeg: bytes, title: str, lines: list[str]) -> bytes | None:
    """Two pictures side by side; green lines are matches the geometry kept, red were rejected."""
    first, second = _thumbnail(first_jpeg), _thumbnail(second_jpeg)
    first_features, second_features = detect_features(first_jpeg, 1000), detect_features(second_jpeg, 1000)
    if first is None or second is None or first_features is None or second_features is None:
        return None
    (left, left_scale), (right, right_scale) = first, second
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, PANEL_WIDTH * 2, 3), dtype=np.uint8)
    canvas[: left.shape[0], :PANEL_WIDTH] = left
    canvas[: right.shape[0], PANEL_WIDTH:] = right
    kept = rejected = 0
    if first_features[1] is not None and second_features[1] is not None:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first_features[1], second_features[1], k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if good:
            start = np.float32([first_features[0][match.queryIdx].pt for match in good])
            end = np.float32([second_features[0][match.trainIdx].pt for match in good])
            inliers = np.zeros(len(good), dtype=bool)
            if len(good) >= 8:
                height_px, width_px = first_features[2]
                _, mask = cv2.findEssentialMat(start, end, camera_matrix(width_px, height_px), method=cv2.RANSAC, threshold=1.5)
                if mask is not None:
                    inliers = mask.ravel().astype(bool)
            kept, rejected = int(inliers.sum()), int((~inliers).sum())
            # Draw an even sample so the pictures stay visible under the lines.
            for wanted, limit, colour in ((False, MAX_REJECTED_LINES, (80, 80, 220)), (True, MAX_KEPT_LINES, (120, 220, 120))):
                chosen = np.flatnonzero(inliers == wanted)
                for index in chosen[:: max(1, len(chosen) // limit)][:limit]:
                    (x1, y1), (x2, y2) = start[index], end[index]
                    cv2.line(
                        canvas,
                        (round(x1 * left_scale), round(y1 * left_scale)),
                        (PANEL_WIDTH + round(x2 * right_scale), round(y2 * right_scale)),
                        colour,
                        1,
                        cv2.LINE_AA,
                    )
    return _with_caption(canvas, title, [f"{kept} matches kept (green), {rejected} rejected (red)", *lines])
