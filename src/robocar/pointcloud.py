"""The lidar's points on the map, for the dashboard: every trusted reading of
every lidar sweep placed at its stop's optimized pose, thinned to one point
per small cell."""

import math
from typing import Mapping, Sequence

import numpy as np

from robocar.surfaces import LIDAR_MAX_RANGE_CM, LIDAR_MIN_RANGE_CM, direction, sonar_origin

# The lidar's points kept for the dashboard: one per cell this size, up to
# this many (the newest win), from the stops' optimized poses.
CLOUD_CELL_CM = 2.5
CLOUD_MAX_POINTS = 20_000


def lidar_points(scan: Sequence[Mapping[str, object]], pose: Sequence[float]) -> np.ndarray:
    """World positions of a lidar sweep's trusted readings (none for sonar)."""
    points = []
    for item in scan:
        if item.get("sensor") != "lidar" or item.get("dynamic"):
            continue
        value = item.get("distance_cm")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not LIDAR_MIN_RANGE_CM <= value <= LIDAR_MAX_RANGE_CM:
            continue
        points.append(float(value) * direction(float(pose[2]) + float(item["angle_deg"])))
    if not points:
        return np.empty((0, 2))
    return np.array(points) + sonar_origin(pose)


class PointCloud:
    """Lidar points, thinned to one per small cell."""

    def __init__(self) -> None:
        self._cells: dict[tuple[int, int], tuple[float, float]] = {}

    def add(self, points: np.ndarray) -> None:
        for x, y in points:
            key = (int(math.floor(x / CLOUD_CELL_CM)), int(math.floor(y / CLOUD_CELL_CM)))
            self._cells.pop(key, None)
            self._cells[key] = (float(x), float(y))
        while len(self._cells) > CLOUD_MAX_POINTS:
            self._cells.pop(next(iter(self._cells)))

    def __len__(self) -> int:
        return len(self._cells)

    def to_json(self) -> list[list[float]]:
        return [[round(x, 1), round(y, 1)] for x, y in self._cells.values()]
