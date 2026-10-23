"""A small simulated room for exercising sonar surface mapping in tests."""

import math

import numpy as np

from robocar.surfaces import SONAR_BEAM_WIDTH_DEGREES, direction, sonar_origin

# Run-frame walls (x right, y ahead of the first stop) and one square post.
ROOM_WALLS = (
    ((-110.0, 120.0), (130.0, 120.0)),
    ((-110.0, -100.0), (-110.0, 120.0)),
    ((130.0, -100.0), (130.0, 120.0)),
    ((-110.0, -100.0), (130.0, -100.0)),
)
POST_CORNERS = ((40.0, 60.0), (52.0, 60.0), (52.0, 72.0), (40.0, 72.0))


class SonarRoom:
    def __init__(self, walls=ROOM_WALLS, posts=(POST_CORNERS,), specular_degrees: float = 40.0) -> None:
        self.segments = [tuple(map(np.array, wall)) for wall in walls]
        for post in posts:
            self.segments.extend(
                (np.array(post[index]), np.array(post[(index + 1) % len(post)]))
                for index in range(len(post))
            )
        self.corners = [np.array(corner) for post in posts for corner in post]
        self.specular_limit = math.cos(math.radians(specular_degrees))

    def distance(self, pose, angle_deg: float) -> float | None:
        origin = sonar_origin(pose)
        half = SONAR_BEAM_WIDTH_DEGREES / 2
        best = None
        for offset in np.linspace(-half, half, 31):
            ray = direction(pose[2] + angle_deg + offset)
            for start, end in self.segments:
                edge = end - start
                matrix = np.column_stack([ray, -edge])
                if abs(np.linalg.det(matrix)) < 1e-9:
                    continue
                distance, fraction = np.linalg.solve(matrix, start - origin)
                if distance <= 0 or not 0 <= fraction <= 1:
                    continue
                normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
                if abs(float(ray @ normal)) < self.specular_limit:
                    continue
                best = distance if best is None else min(best, distance)
        # Convex corners scatter sound back from any direction inside the cone.
        for corner in self.corners:
            offset = corner - origin
            distance = float(np.linalg.norm(offset))
            centre = direction(pose[2] + angle_deg)
            if float(offset @ centre) / distance >= math.cos(math.radians(half)):
                if not self._occluded(origin, corner, distance):
                    best = distance if best is None else min(best, distance)
        return None if best is None else round(float(best), 1)

    def _occluded(self, origin, target, distance: float) -> bool:
        ray = (target - origin) / distance
        for start, end in self.segments:
            edge = end - start
            matrix = np.column_stack([ray, -edge])
            if abs(np.linalg.det(matrix)) < 1e-9:
                continue
            along, fraction = np.linalg.solve(matrix, start - origin)
            if 0 < along < distance - 0.5 and 0 < fraction < 1:
                return True
        return False

    def scan(self, pose, angles=range(-60, 61, 6)) -> list[dict[str, object]]:
        return [
            {"angle_deg": angle, "distance_cm": self.distance(pose, angle), "samples": 3, "error": None}
            for angle in angles
        ]
