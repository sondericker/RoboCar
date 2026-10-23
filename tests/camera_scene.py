"""Render camera views of a textured 360-degree scene for camera-motion tests."""

import math

import cv2
import numpy as np

from robocar.mapping import CAMERA_HORIZONTAL_FOV_DEGREES

PIXELS_PER_DEGREE = 10
WIDTH, HEIGHT = 640, 480


class PanoramaScene:
    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        panorama = rng.integers(0, 256, (900, 360 * PIXELS_PER_DEGREE), dtype=np.uint8)
        panorama = cv2.GaussianBlur(panorama, (0, 0), 2)
        for _ in range(2500):
            centre = (int(rng.integers(0, panorama.shape[1])), int(rng.integers(0, 900)))
            cv2.circle(panorama, centre, int(rng.integers(4, 18)), int(rng.integers(0, 256)), 2)
        self.panorama = panorama
        focal = WIDTH / (2 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEGREES) / 2))
        u, v = np.meshgrid(np.arange(WIDTH) - WIDTH / 2, np.arange(HEIGHT) - HEIGHT / 2)
        self._angle_offsets = np.degrees(np.arctan2(u, focal))
        self._map_y = (v * focal / np.sqrt(u**2 + focal**2) + 450).astype(np.float32)

    def view(self, yaw_deg: float) -> bytes:
        """JPEG seen by a camera facing `yaw_deg` (counterclockwise) from the centre."""
        azimuth = yaw_deg - self._angle_offsets  # the right of the image is clockwise
        map_x = ((azimuth % 360) * PIXELS_PER_DEGREE).astype(np.float32)
        image = cv2.remap(self.panorama, map_x, self._map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return cv2.imencode(".jpg", image)[1].tobytes()


class RoomScene:
    """A room of textured vertical walls seen by the car's camera, with parallax."""

    def __init__(
        self,
        walls,
        seed: int = 11,
        wall_height_cm: float = 120.0,
        camera_height_cm: float = 15.0,
        camera_forward_cm: float = 7.0,
        width: int = 320,
        height: int = 240,
        pixels_per_cm: float = 4.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.walls = [(np.array(start, dtype=float), np.array(end, dtype=float)) for start, end in walls]
        self.wall_height = wall_height_cm
        self.camera_height = camera_height_cm
        self.camera_forward = camera_forward_cm
        self.width, self.height = width, height
        self.pixels_per_cm = pixels_per_cm
        rows = int(wall_height_cm * pixels_per_cm)
        textures, self.offsets = [], []
        column = 0
        for start, end in self.walls:
            columns = int(np.linalg.norm(end - start) * pixels_per_cm) + 1
            texture = cv2.GaussianBlur(rng.integers(40, 216, (rows, columns), dtype=np.uint8), (0, 0), 1.5)
            for _ in range(max(1, columns * rows // 1500)):
                centre = (int(rng.integers(0, columns)), int(rng.integers(0, rows)))
                colour = int(rng.integers(0, 256))
                if rng.random() < 0.5:
                    cv2.circle(texture, centre, int(rng.integers(4, 20)), colour, int(rng.integers(1, 4)))
                else:
                    size = rng.integers(6, 30, 2)
                    cv2.rectangle(texture, centre, (centre[0] + int(size[0]), centre[1] + int(size[1])), colour, -1)
            textures.append(texture)
            self.offsets.append(column)
            column += columns
        # A plain floor and ceiling, appended as two flat strips.
        self.floor_column = column
        textures.append(np.full((rows, 4), 90, dtype=np.uint8))
        textures.append(np.full((rows, 4), 170, dtype=np.uint8))
        self.atlas = np.hstack(textures)
        self.focal = width / (2 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEGREES) / 2))

    def view(self, pose, pan_offset_deg: float = 0.0) -> bytes:
        x, y, heading = pose
        yaw = math.radians(heading + pan_offset_deg)
        forward = np.array([-math.sin(math.radians(heading)), math.cos(math.radians(heading))])
        camera = np.array([x, y]) + self.camera_forward * forward
        look = np.array([-math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), math.sin(yaw)])
        normalized_x = (np.arange(self.width) - self.width / 2) / self.focal
        rays = look[None, :] + normalized_x[:, None] * right[None, :]
        best_t = np.full(self.width, np.inf)
        best_column = np.zeros(self.width)
        for (start, end), offset in zip(self.walls, self.offsets):
            edge = end - start
            denominator = rays[:, 0] * (-edge[1]) - rays[:, 1] * (-edge[0])
            relative_start = start - camera
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (relative_start[0] * (-edge[1]) - relative_start[1] * (-edge[0])) / denominator
                s = (rays[:, 0] * relative_start[1] - rays[:, 1] * relative_start[0]) / denominator
            hit = (t > 1e-3) & (s >= 0) & (s <= 1) & (t < best_t)
            best_t[hit] = t[hit]
            best_column[hit] = offset + s[hit] * np.linalg.norm(edge) * self.pixels_per_cm
        normalized_y = (self.height / 2 - np.arange(self.height)) / self.focal
        heights = self.camera_height + best_t[None, :] * normalized_y[:, None]
        map_x = np.broadcast_to(best_column[None, :], heights.shape).astype(np.float32).copy()
        map_y = ((self.wall_height - heights) * self.pixels_per_cm).astype(np.float32)
        below, above = heights < 0, (heights > self.wall_height) | ~np.isfinite(heights)
        map_x[below], map_y[below] = self.floor_column + 1, 1
        map_x[above], map_y[above] = self.floor_column + 5, 1
        image = cv2.remap(self.atlas, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
