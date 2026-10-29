"""What the camera recognizes, placed on the map.

Each stop's picture goes through the object detector (on the AI HAT+ when
present). A detection's direction comes from where it sits in the picture
and where the head was pointing; its distance from the sonar readings in
that direction, when they are consistent with the object.

- People and animals move. Sonar readings toward them are flagged
  "dynamic" and kept out of walls, the coverage grid, scan matching and
  place memory (they still count for safety), and the explorer will not
  drive at one close ahead.
- Furniture and appliances become landmarks. A sighting with a range is a
  point; one without is a ray, and two rays from places far enough apart
  are triangulated. Landmarks of the same kind close together are merged,
  weighting each sighting by how sure it is. A landmark counts once it has
  been seen twice, or once with high confidence.
- Rooms are named from the landmarks in them (a bed makes a bedroom, an
  oven a kitchen). Landmarks join a room when they are close and no
  confirmed wall lies between them.

Everything is rebuilt from the stored sightings and the pose graph's
optimized poses whenever the map is, so a loop closure moves the landmarks
too.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from robocar.surfaces import SONAR_MAX_RANGE_CM, direction, sonar_origin

# The OV5647 in the 640x480 mode RoboCar uses reads almost the full sensor
# width (2560 of 2592 pixels): about 53 degrees across.
CAMERA_HFOV_DEG = 53.0
DYNAMIC_LABELS = frozenset({"person", "cat", "dog", "bird", "horse", "sheep", "cow", "teddy bear"})
# Things that stay put and help say where the car is and what room it is in.
LANDMARK_LABELS = frozenset(
    {
        "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "refrigerator", "oven",
        "microwave", "sink", "toaster", "potted plant", "bench", "clock", "bicycle", "piano",
    }
)
ROOM_HINTS: dict[str, dict[str, float]] = {
    "bed": {"bedroom": 3.0},
    "toilet": {"bathroom": 3.0},
    "sink": {"bathroom": 1.0, "kitchen": 1.0},
    "refrigerator": {"kitchen": 3.0},
    "oven": {"kitchen": 3.0},
    "microwave": {"kitchen": 2.0},
    "toaster": {"kitchen": 2.0},
    "couch": {"living room": 2.0},
    "tv": {"living room": 1.5, "bedroom": 0.5},
    "dining table": {"dining room": 2.0, "kitchen": 0.5},
    "laptop": {"office": 1.5},
    "chair": {"dining room": 0.3, "office": 0.3},
    "bicycle": {"garage": 1.0},
}
MIN_CONFIDENCE = 0.4
# Sonar readings this far either side of a detection's edges belong to it.
RANGE_MARGIN_DEG = 4.0
# Readings toward a person or pet, out to this far beyond the nearest one,
# are theirs rather than the wall behind.
DYNAMIC_DEPTH_CM = 40.0
DYNAMIC_MARGIN_DEG = 6.0
# Landmark fusion.
LANDMARK_GATE_CM = 60.0
BEARING_GATE_DEG = 8.0
TRIANGULATE_MIN_BASELINE_CM = 30.0
TRIANGULATE_MIN_ANGLE_DEG = 10.0
TRIANGULATE_MAX_RANGE_CM = 400.0
TRIANGULATED_SIGMA_CM = 25.0
PENDING_RAYS_PER_LABEL = 20
CONFIRM_SIGHTINGS = 2
CONFIRM_CONFIDENCE = 0.7
# Rooms.
ROOM_LINK_CM = 250.0
ROOM_MIN_SCORE = 1.5

# Is there a wall between two points?
WallTest = Callable[[tuple[float, float], tuple[float, float]], bool]


def pixel_bearing(column: float, width: int, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Direction of a picture column, counterclockwise from the camera's axis
    (so columns right of centre are negative)."""
    focal = (width / 2) / math.tan(math.radians(hfov_deg / 2))
    return -math.degrees(math.atan((column - width / 2) / focal))


@dataclass(frozen=True)
class Sighting:
    label: str
    confidence: float
    bearing_deg: float  # counterclockwise from the car's heading
    half_width_deg: float
    range_cm: float | None  # from the sonar origin, when the sweep measured it
    box: tuple[int, int, int, int]

    @property
    def dynamic(self) -> bool:
        return self.label in DYNAMIC_LABELS

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bearing_deg": round(self.bearing_deg, 1),
            "half_width_deg": round(self.half_width_deg, 1),
            "range_cm": None if self.range_cm is None else round(self.range_cm, 1),
            "box": list(self.box),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "Sighting":
        range_cm = data.get("range_cm")
        return cls(
            str(data["label"]),
            float(data["confidence"]),
            float(data["bearing_deg"]),
            float(data["half_width_deg"]),
            None if range_cm is None else float(range_cm),
            tuple(int(value) for value in data.get("box", (0, 0, 0, 0))),
        )


def _reading(item: Mapping[str, object]) -> float | None:
    value = item.get("distance_cm")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) if value <= SONAR_MAX_RANGE_CM else None


def sightings_from(
    detections: Sequence[Mapping[str, object]],
    frame_width: int,
    camera_bearing_deg: float,
    scan: Sequence[Mapping[str, object]],
) -> list[Sighting]:
    """Detections in a picture taken facing `camera_bearing_deg` (relative to
    the car), with ranges from the sweep where it measured them."""
    sightings = []
    for detection in detections:
        label = str(detection.get("label", ""))
        confidence = float(detection.get("confidence", 0.0))
        if confidence < MIN_CONFIDENCE or (label not in DYNAMIC_LABELS and label not in LANDMARK_LABELS):
            continue
        x, y, width, height = (int(value) for value in detection["box"])
        left = pixel_bearing(x, frame_width)
        right = pixel_bearing(x + width, frame_width)
        bearing = camera_bearing_deg + (left + right) / 2
        half_width = abs(left - right) / 2
        ranges = [
            reading
            for item in scan
            if (reading := _reading(item)) is not None
            and abs(float(item["angle_deg"]) - bearing) <= half_width + RANGE_MARGIN_DEG
        ]
        sightings.append(
            Sighting(label, confidence, bearing, half_width, min(ranges) if ranges else None, (x, y, width, height))
        )
    return sightings


def mark_dynamic(scan: Sequence[Mapping[str, object]], sightings: Sequence[Sighting]) -> list[dict[str, object]]:
    """The scan with readings that hit a person or animal flagged "dynamic"."""
    marked = []
    for item in scan:
        item = dict(item)
        reading = _reading(item)
        for sighting in sightings:
            if not sighting.dynamic:
                continue
            within = abs(float(item["angle_deg"]) - sighting.bearing_deg) <= sighting.half_width_deg + DYNAMIC_MARGIN_DEG
            near = sighting.range_cm is None or (reading is not None and reading <= sighting.range_cm + DYNAMIC_DEPTH_CM)
            if within and near:
                item["dynamic"] = True
                break
        marked.append(item)
    return marked


def static_scan(scan: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """The scan for mapping: readings that hit people or animals left out."""
    return [
        {**item, "distance_cm": None, "samples": 0} if item.get("dynamic") else dict(item)
        for item in scan
    ]


def hazards_from(sightings: Sequence[Sighting]) -> list[tuple[float, float | None]]:
    """(bearing, range or None) of every person or animal in view."""
    return [(sighting.bearing_deg, sighting.range_cm) for sighting in sightings if sighting.dynamic]


@dataclass
class Landmark:
    label: str
    x: float
    y: float
    weight: float  # summed 1 / sigma^2 of the sightings that placed it
    sightings: int = 1
    confidence: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.sightings >= CONFIRM_SIGHTINGS or self.confidence >= CONFIRM_CONFIDENCE

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "sigma_cm": round(1 / math.sqrt(self.weight), 1),
            "sightings": self.sightings,
            "confidence": round(self.confidence, 2),
        }


@dataclass
class _Ray:
    origin: np.ndarray
    heading: np.ndarray
    confidence: float


@dataclass
class SemanticMap:
    landmarks: list[Landmark] = field(default_factory=list)
    _rays: dict[str, list[_Ray]] = field(default_factory=dict)

    def add(self, sightings: Sequence[Sighting], pose: Sequence[float]) -> None:
        origin = sonar_origin(pose)
        for sighting in sightings:
            if sighting.dynamic:
                continue
            ray = direction(float(pose[2]) + sighting.bearing_deg)
            if sighting.range_cm is not None:
                sigma = 8.0 + 0.08 * sighting.range_cm
                self._add_point(sighting.label, origin + ray * sighting.range_cm, sigma, sighting.confidence)
            else:
                self._add_ray(sighting.label, _Ray(origin, ray, sighting.confidence))

    def _add_point(self, label: str, point: np.ndarray, sigma: float, confidence: float) -> None:
        weight = 1.0 / sigma**2
        best = None
        for landmark in self.landmarks:
            if landmark.label != label:
                continue
            gap = math.hypot(landmark.x - point[0], landmark.y - point[1])
            gate = max(LANDMARK_GATE_CM, 3.0 * max(sigma, 1 / math.sqrt(landmark.weight)))
            if gap <= gate and (best is None or gap < best[0]):
                best = (gap, landmark)
        if best is None:
            self.landmarks.append(Landmark(label, float(point[0]), float(point[1]), weight, 1, confidence))
            return
        landmark = best[1]
        total = landmark.weight + weight
        landmark.x = (landmark.x * landmark.weight + float(point[0]) * weight) / total
        landmark.y = (landmark.y * landmark.weight + float(point[1]) * weight) / total
        landmark.weight = total
        landmark.sightings += 1
        landmark.confidence = max(landmark.confidence, confidence)

    def _add_ray(self, label: str, ray: _Ray) -> None:
        # Seen again along a ray toward a landmark already placed: it counts.
        for landmark in self.landmarks:
            if landmark.label != label:
                continue
            offset = np.array([landmark.x, landmark.y]) - ray.origin
            along = float(offset @ ray.heading)
            if along <= 0 or along > TRIANGULATE_MAX_RANGE_CM:
                continue
            off_angle = math.degrees(math.atan2(abs(float(ray.heading[0] * offset[1] - ray.heading[1] * offset[0])), along))
            if off_angle <= BEARING_GATE_DEG:
                landmark.sightings += 1
                landmark.confidence = max(landmark.confidence, ray.confidence)
                return
        # Otherwise pair it with an earlier ray from far enough away.
        pending = self._rays.setdefault(label, [])
        for index, other in enumerate(pending):
            point = _intersect(ray, other)
            if point is not None:
                del pending[index]
                self._add_point(label, point, TRIANGULATED_SIGMA_CM, max(ray.confidence, other.confidence))
                return
        pending.append(ray)
        del pending[:-PENDING_RAYS_PER_LABEL]

    def confirmed(self) -> list[Landmark]:
        return [landmark for landmark in self.landmarks if landmark.confirmed]

    def rooms(self, walls: WallTest | None = None) -> list[dict[str, object]]:
        """Named rooms: groups of nearby landmarks with no wall between them."""
        landmarks = [landmark for landmark in self.confirmed() if landmark.label in ROOM_HINTS]
        groups: list[list[Landmark]] = []
        for landmark in landmarks:
            joined = [
                group
                for group in groups
                if any(
                    math.hypot(landmark.x - other.x, landmark.y - other.y) <= ROOM_LINK_CM
                    and (walls is None or not walls((landmark.x, landmark.y), (other.x, other.y)))
                    for other in group
                )
            ]
            merged = [landmark] + [member for group in joined for member in group]
            groups = [group for group in groups if all(group is not other for other in joined)] + [merged]
        rooms = []
        for group in groups:
            scores: dict[str, float] = {}
            for landmark in group:
                certainty = min(1.0, landmark.confidence * landmark.sightings / CONFIRM_SIGHTINGS)
                for room, score in ROOM_HINTS[landmark.label].items():
                    scores[room] = scores.get(room, 0.0) + score * certainty
            if not scores:
                continue
            name, score = max(scores.items(), key=lambda item: item[1])
            if score < ROOM_MIN_SCORE:
                continue
            rooms.append(
                {
                    "name": name,
                    "x": round(sum(member.x for member in group) / len(group), 1),
                    "y": round(sum(member.y for member in group) / len(group), 1),
                    "score": round(score, 2),
                    "because": sorted({member.label for member in group}),
                }
            )
        return rooms

    def to_json(self, walls: WallTest | None = None) -> dict[str, object]:
        return {
            "landmarks": [landmark.to_json() for landmark in self.confirmed()],
            "rooms": self.rooms(walls),
        }


def _intersect(first: _Ray, second: _Ray) -> np.ndarray | None:
    """Where two rays cross, if they come from far enough apart at a wide
    enough angle for the crossing to be well defined, in front of both."""
    if np.linalg.norm(first.origin - second.origin) < TRIANGULATE_MIN_BASELINE_CM:
        return None
    sine = abs(float(first.heading[0] * second.heading[1] - first.heading[1] * second.heading[0]))
    if sine < math.sin(math.radians(TRIANGULATE_MIN_ANGLE_DEG)):
        return None
    matrix = np.column_stack([first.heading, -second.heading])
    along_first, along_second = np.linalg.solve(matrix, second.origin - first.origin)
    if not (0 < along_first <= TRIANGULATE_MAX_RANGE_CM and 0 < along_second <= TRIANGULATE_MAX_RANGE_CM):
        return None
    return first.origin + first.heading * along_first
