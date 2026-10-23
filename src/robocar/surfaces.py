"""Surface reconstruction from swept ultrasonic ranges.

Coordinates follow the robot's motor and servo conventions: a robot frame has
+y forward and +x to the right, world headings and sweep angles are degrees
counterclockwise (to the left) from +y, and a pose is (x_cm, y_cm, heading_deg).

An ultrasonic reading reports the nearest reflector anywhere inside a cone of
about 15 degrees, not a point on the beam's centre line. Adjacent sweep angles
that return the same range form a region of constant depth (RCD): the reflector
sits at the RCD's centre bearing and the surface faces the sensor there. Outside
an RCD the range rises along an oblique wall, where the echo comes from the cone
edge nearest the wall's normal; shifting those bearings half a beam toward
falling range puts the points back on the wall instead of on a bowed arc.

Corrected echoes are split into straight runs and fitted as wall segments.
Walls and point features (corners, edges, legs) accumulate evidence from every
sweep, are refitted from all of their supporting echoes, merged when they turn
out to be one surface, joined at corners into contours, and retired when later
beams pass through space they claimed to occupy.
"""

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

SONAR_BEAM_WIDTH_DEGREES = 15.0
SONAR_MIN_RANGE_CM = 3.0
SONAR_MAX_RANGE_CM = 300.0
# Distance from the car's turning centre forward to the pan axis that carries
# the transducer. Measure this on the car; it matters most for turns.
SONAR_FORWARD_OFFSET_CM = 7.0
RANGE_JUMP_CM = 6.0
RANGE_JUMP_FRACTION = 0.15
RCD_TOLERANCE_CM = 1.0
RCD_TOLERANCE_FRACTION = 0.012
SPLIT_TOLERANCE_CM = 2.5
SPLIT_TOLERANCE_FRACTION = 0.03
MIN_LINE_ECHOES = 3
MIN_LINE_LENGTH_CM = 8.0
MAX_SURFACE_POINTS = 400
MAX_RECENT_ECHOES = 1500
WALL_ANGLE_TOLERANCE_DEGREES = 15.0
WALL_MERGE_ANGLE_DEGREES = 12.0
WALL_GAP_CM = 20.0
FEATURE_RADIUS_CM = 8.0
CORNER_JOIN_CM = 25.0
CORNER_EXTEND_CM = 45.0
CARVE_INCIDENCE_DEGREES = 25.0


@dataclass(frozen=True)
class Echo:
    """One reflector in the robot frame, after beam-width correction."""

    angle_deg: float
    bearing_deg: float
    range_cm: float
    x_cm: float
    y_cm: float
    kind: str
    sigma_cm: float


def pose_rotation(heading_deg: float) -> np.ndarray:
    radians = math.radians(heading_deg)
    cosine, sine = math.cos(radians), math.sin(radians)
    return np.array([[cosine, -sine], [sine, cosine]])


def to_world(pose: Sequence[float], local_xy: np.ndarray) -> np.ndarray:
    local_xy = np.asarray(local_xy, dtype=np.float64).reshape(-1, 2)
    return local_xy @ pose_rotation(pose[2]).T + np.array([pose[0], pose[1]])


def direction(bearing_deg: float) -> np.ndarray:
    radians = math.radians(bearing_deg)
    return np.array([-math.sin(radians), math.cos(radians)])


def sonar_origin(pose: Sequence[float]) -> np.ndarray:
    return to_world(pose, np.array([0.0, SONAR_FORWARD_OFFSET_CM]))[0]


def _valid_range(distance: object) -> float | None:
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        return None
    value = float(distance)
    if not math.isfinite(value) or value < SONAR_MIN_RANGE_CM:
        return None
    return value


def _jump_tolerance(first: float, second: float) -> float:
    return RANGE_JUMP_CM + RANGE_JUMP_FRACTION * min(first, second)


def _rcd_tolerance(range_cm: float) -> float:
    return max(RCD_TOLERANCE_CM, RCD_TOLERANCE_FRACTION * range_cm)


def _sweep_step(angles: Sequence[float]) -> float:
    steps = [b - a for a, b in zip(angles, angles[1:]) if b - a > 0]
    return float(np.median(steps)) if steps else SONAR_BEAM_WIDTH_DEGREES


def _make_echo(angle: float, bearing: float, range_cm: float, kind: str) -> Echo:
    bearing_sigma = {
        "plane": 2.5,
        "point": 2.5,
        "oblique": 3.5,
    }.get(kind, SONAR_BEAM_WIDTH_DEGREES / 2)
    sigma = math.hypot(1.0 + 0.01 * range_cm, range_cm * math.radians(bearing_sigma))
    x, y = np.array([0.0, SONAR_FORWARD_OFFSET_CM]) + range_cm * direction(bearing)
    return Echo(angle, bearing, range_cm, float(x), float(y), kind, sigma)


def split_runs(scan: Sequence[Mapping[str, object]]) -> list[list[tuple[float, float]]]:
    """Group a sweep into angularly adjacent readings with continuous range."""
    angles = sorted(float(item["angle_deg"]) for item in scan)
    step = _sweep_step(angles)
    readings = sorted(
        (float(item["angle_deg"]), value)
        for item in scan
        if (value := _valid_range(item.get("distance_cm"))) is not None
        and value <= SONAR_MAX_RANGE_CM
    )
    runs: list[list[tuple[float, float]]] = []
    for angle, range_cm in readings:
        if runs:
            previous_angle, previous_range = runs[-1][-1]
            if (
                angle - previous_angle <= step * 1.5
                and abs(range_cm - previous_range) <= _jump_tolerance(range_cm, previous_range)
            ):
                runs[-1].append((angle, range_cm))
                continue
        runs.append([(angle, range_cm)])
    return runs


def extract_echoes(scan: Sequence[Mapping[str, object]]) -> list[list[Echo]]:
    """Return beam-corrected echoes, one list per continuous run of the sweep."""
    half_beam = SONAR_BEAM_WIDTH_DEGREES / 2
    echo_runs: list[list[Echo]] = []
    for run in split_runs(scan):
        # Grow each region of constant depth outward from the nearest
        # unclaimed reading, so slopes never pair up into false plateaus.
        owner = [-1] * len(run)
        seeds: list[int] = []
        for seed in sorted(range(len(run)), key=lambda index: run[index][1]):
            if owner[seed] >= 0:
                continue
            owner[seed] = seed
            seeds.append(seed)
            tolerance = _rcd_tolerance(run[seed][1])
            for step in (-1, 1):
                cursor = seed + step
                while 0 <= cursor < len(run) and owner[cursor] < 0 and abs(run[cursor][1] - run[seed][1]) <= tolerance:
                    owner[cursor] = seed
                    cursor += step
        groups = [
            [run[index] for index in range(len(run)) if owner[index] == seed]
            for seed in sorted(seeds)
        ]
        centres = [
            (float(np.mean([item[0] for item in group])), float(np.median([item[1] for item in group])))
            for group in groups
        ]

        echoes: list[Echo] = []
        for index, group in enumerate(groups):
            angle, range_cm = centres[index]
            before = centres[index - 1][1] if index > 0 else None
            after = centres[index + 1][1] if index + 1 < len(centres) else None
            neighbours = [value for value in (before, after) if value is not None]
            if len(group) > 1 or (len(neighbours) == 2 and all(value > range_cm for value in neighbours)):
                # A region of constant depth: the surface faces the sensor at
                # its centre. Without continuous neighbours it is an isolated
                # point reflector such as an edge or a chair leg.
                kind = "plane" if neighbours else "point"
                echoes.append(_make_echo(angle, angle, range_cm, kind))
                continue
            if not neighbours:
                echoes.append(_make_echo(angle, angle, range_cm, "isolated"))
                continue
            if len(neighbours) == 2 and all(value < range_cm for value in neighbours):
                # A range peak between two nearer readings is where two
                # surfaces meet inside the beam (a concave corner); the
                # neighbouring readings already describe both surfaces.
                continue
            if before is not None and before < range_cm:
                toward = -1
            elif after is not None and after < range_cm:
                toward = 1
            else:
                # At the end of a run with range falling toward the end, the
                # surface keeps approaching its normal beyond the sweep.
                toward = 1 if before is not None else -1
            # Follow the falling range to its minimum; the shift never carries
            # the bearing past a minimum inside the run.
            cursor = index
            while 0 <= cursor + toward < len(centres) and centres[cursor + toward][1] < centres[cursor][1]:
                cursor += toward
            at_run_end = cursor in (0, len(centres) - 1) and len(groups[cursor]) == 1
            limit = half_beam if at_run_end else abs(centres[cursor][0] - angle)
            shift = min(half_beam, limit)
            echoes.append(_make_echo(angle, angle + toward * shift, range_cm, "oblique"))
        echo_runs.append(echoes)
    return echo_runs


def fit_line(points: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Weighted total-least-squares line: centroid, direction, normal, rms."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    centroid = (points * weights[:, None]).sum(axis=0) / total
    centred = points - centroid
    scatter = (centred * weights[:, None]).T @ centred / total
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    line_direction = eigenvectors[:, 1]
    normal = np.array([-line_direction[1], line_direction[0]])
    rms = math.sqrt(max(float(eigenvalues[0]), 0.0))
    return centroid, line_direction, normal, rms


def _split_tolerance(points: np.ndarray) -> float:
    reach = float(np.linalg.norm(points, axis=1).mean()) if len(points) else 0.0
    return max(SPLIT_TOLERANCE_CM, SPLIT_TOLERANCE_FRACTION * reach)


def split_into_lines(points: np.ndarray, origin: np.ndarray) -> list[list[int]]:
    """Split ordered points into straight pieces (recursive split, then merge)."""
    points = np.asarray(points, dtype=np.float64)

    def split(indices: list[int]) -> list[list[int]]:
        if len(indices) < MIN_LINE_ECHOES:
            return [indices]
        start, end = points[indices[0]], points[indices[-1]]
        chord = end - start
        length = float(np.linalg.norm(chord))
        if length < 1e-6:
            return [indices]
        normal = np.array([-chord[1], chord[0]]) / length
        deviations = np.abs((points[indices] - start) @ normal)
        worst = int(np.argmax(deviations))
        if deviations[worst] <= _split_tolerance(points[indices] - origin) or worst in (0, len(indices) - 1):
            return [indices]
        return split(indices[: worst + 1]) + split(indices[worst:])

    pieces = split(list(range(len(points))))
    merged: list[list[int]] = []
    for piece in pieces:
        if merged:
            candidate = sorted(set(merged[-1]) | set(piece))
            if len(candidate) >= MIN_LINE_ECHOES:
                _, _, _, rms = fit_line(points[candidate])
                if rms <= _split_tolerance(points[candidate] - origin) / 2:
                    merged[-1] = candidate
                    continue
        merged.append(piece)
    return merged


@dataclass
class Surface:
    id: int
    kind: str
    points: np.ndarray
    weights: np.ndarray
    scans: set[int] = field(default_factory=set)
    misses: int = 0
    view_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))
    line_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    extent: tuple[float, float] = (0.0, 0.0)
    rms_cm: float = 0.0

    def add(self, points: np.ndarray, weights: np.ndarray, scan_index: int, origin: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.points = np.vstack([self.points, points])[-MAX_SURFACE_POINTS:]
        self.weights = np.concatenate([self.weights, weights])[-MAX_SURFACE_POINTS:]
        self.scans.add(scan_index)
        self.view_sum = self.view_sum + (origin - points.mean(axis=0))
        self.refit()

    def absorb(self, other: "Surface") -> None:
        self.points = np.vstack([self.points, other.points])[-MAX_SURFACE_POINTS:]
        self.weights = np.concatenate([self.weights, other.weights])[-MAX_SURFACE_POINTS:]
        self.scans |= other.scans
        self.misses += other.misses
        self.view_sum = self.view_sum + other.view_sum
        self.refit()

    def refit(self) -> None:
        if self.kind == "feature" or len(self.points) < 2:
            total = float(self.weights.sum())
            self.centroid = (self.points * self.weights[:, None]).sum(axis=0) / total
            spread = np.linalg.norm(self.points - self.centroid, axis=1)
            self.rms_cm = float(math.sqrt((self.weights * spread**2).sum() / total))
            self.extent = (0.0, 0.0)
            return
        inliers = np.ones(len(self.points), dtype=bool)
        for _ in range(2):
            centroid, line_direction, normal, rms = fit_line(self.points[inliers], self.weights[inliers])
            residuals = np.abs((self.points - centroid) @ normal)
            refined = residuals <= max(3.0, 3 * rms)
            if refined.sum() < 2 or np.array_equal(refined, inliers):
                break
            inliers = refined
        if line_direction[0] < 0 or (abs(line_direction[0]) < 1e-9 and line_direction[1] < 0):
            line_direction = -line_direction
        normal = np.array([-line_direction[1], line_direction[0]])
        if float(normal @ self.view_sum) < 0:
            normal = -normal
        along = (self.points[inliers] - centroid) @ line_direction
        self.centroid, self.line_direction, self.normal, self.rms_cm = centroid, line_direction, normal, rms
        self.extent = (float(along.min()), float(along.max()))

    @property
    def observations(self) -> int:
        return len(self.scans)

    @property
    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.centroid + self.extent[0] * self.line_direction,
            self.centroid + self.extent[1] * self.line_direction,
        )

    @property
    def length_cm(self) -> float:
        return self.extent[1] - self.extent[0]

    @property
    def confidence(self) -> float:
        return self.observations / (self.observations + 2 * self.misses + 1)

    @property
    def confirmed(self) -> bool:
        return self.observations >= 2 and self.confidence >= 0.5

    def offset(self, point: np.ndarray) -> float:
        return float((point - self.centroid) @ self.normal)

    def along(self, point: np.ndarray) -> float:
        return float((point - self.centroid) @ self.line_direction)


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(first @ second)))))


def _interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, second[0] - first[1], first[0] - second[1])


def _beyond_end(wall: Surface, end: int, point: np.ndarray) -> bool:
    along = wall.along(point)
    return along <= wall.extent[0] + 5.0 if end == 0 else along >= wall.extent[1] - 5.0


def _intersection(first: Surface, second: Surface) -> np.ndarray | None:
    matrix = np.column_stack([first.line_direction, -second.line_direction])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    scale = np.linalg.solve(matrix, second.centroid - first.centroid)
    return first.centroid + scale[0] * first.line_direction


@dataclass(frozen=True)
class IntegrationReport:
    scan_index: int
    echoes: int
    line_candidates: int
    fused: int
    created: int
    retired: int


class SurfaceMap:
    """A refinable set of wall segments and point features in one run frame."""

    def __init__(self) -> None:
        self.surfaces: dict[int, Surface] = {}
        self.scan_count = 0
        self.recent_echoes: list[tuple[float, float, str]] = []
        self._next_id = 1

    def walls(self, confirmed_only: bool = False) -> list[Surface]:
        return [
            surface
            for surface in self.surfaces.values()
            if surface.kind == "wall" and (surface.confirmed or not confirmed_only)
        ]

    def features(self, confirmed_only: bool = False) -> list[Surface]:
        return [
            surface
            for surface in self.surfaces.values()
            if surface.kind == "feature" and (surface.confirmed or not confirmed_only)
        ]

    def integrate(
        self,
        scan: Sequence[Mapping[str, object]],
        pose: Sequence[float],
        position_sigma_cm: float = 0.0,
        heading_sigma_deg: float = 0.0,
    ) -> IntegrationReport:
        scan_index = self.scan_count
        self.scan_count += 1
        origin = sonar_origin(pose)
        retired = self._carve(scan, pose, origin)

        echo_runs = extract_echoes(scan)
        heading_sigma = math.radians(heading_sigma_deg)
        candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
        singles: list[tuple[np.ndarray, float, float]] = []
        echo_count = 0
        for echoes in echo_runs:
            if not echoes:
                continue
            echo_count += len(echoes)
            local = np.array([[echo.x_cm, echo.y_cm] for echo in echoes])
            world = to_world(pose, local)
            weights = np.array(
                [
                    1.0 / (echo.sigma_cm**2 + position_sigma_cm**2 + (echo.range_cm * heading_sigma) ** 2)
                    for echo in echoes
                ]
            )
            self.recent_echoes.extend(
                (float(point[0]), float(point[1]), echo.kind) for point, echo in zip(world, echoes)
            )
            for piece in split_into_lines(world, origin):
                piece_points = world[piece]
                reach = float(np.mean([echoes[index].range_cm for index in piece]))
                if (
                    len(piece) >= MIN_LINE_ECHOES
                    and float(np.linalg.norm(piece_points[-1] - piece_points[0])) >= MIN_LINE_LENGTH_CM
                ):
                    candidates.append((piece_points, weights[piece], reach))
                else:
                    singles.extend((world[index], float(weights[index]), echoes[index].range_cm) for index in piece)
        self.recent_echoes = self.recent_echoes[-MAX_RECENT_ECHOES:]

        fused = created = 0
        for points, weights, reach in candidates:
            if self._fuse_line(points, weights, reach, position_sigma_cm, scan_index, origin):
                fused += 1
            else:
                self._create("wall", points, weights, scan_index, origin)
                created += 1
        for point, weight, reach in singles:
            if self._fuse_point(point, weight, reach, position_sigma_cm, scan_index, origin):
                fused += 1
            else:
                self._create("feature", point[None, :], np.array([weight]), scan_index, origin)
                created += 1
        self._consolidate()
        return IntegrationReport(scan_index, echo_count, len(candidates), fused, created, retired)

    def _create(self, kind: str, points: np.ndarray, weights: np.ndarray, scan_index: int, origin: np.ndarray) -> Surface:
        surface = Surface(self._next_id, kind, np.empty((0, 2)), np.empty(0))
        self._next_id += 1
        surface.add(points, weights, scan_index, origin)
        self.surfaces[surface.id] = surface
        return surface

    def _fuse_line(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        reach: float,
        position_sigma_cm: float,
        scan_index: int,
        origin: np.ndarray,
    ) -> bool:
        _, line_direction, _, _ = fit_line(points, weights)
        tolerance = 4.0 + 0.03 * reach + 2 * position_sigma_cm
        matches = []
        for wall in self.walls():
            if _angle_between(line_direction, wall.line_direction) > WALL_ANGLE_TOLERANCE_DEGREES:
                continue
            if max(abs(wall.offset(points[0])), abs(wall.offset(points[-1]))) > tolerance:
                continue
            span = sorted((wall.along(points[0]), wall.along(points[-1])))
            if _interval_gap(wall.extent, (span[0], span[1])) > WALL_GAP_CM:
                continue
            matches.append(wall)
        if not matches:
            return False
        target = max(matches, key=lambda wall: wall.observations)
        for wall in matches:
            if wall is not target:
                target.absorb(self.surfaces.pop(wall.id))
        target.add(points, weights, scan_index, origin)
        return True

    def _fuse_point(
        self,
        point: np.ndarray,
        weight: float,
        reach: float,
        position_sigma_cm: float,
        scan_index: int,
        origin: np.ndarray,
    ) -> bool:
        wall_tolerance = 4.0 + 0.03 * reach + 2 * position_sigma_cm
        feature_tolerance = FEATURE_RADIUS_CM * 0.75 + 0.03 * reach + 2 * position_sigma_cm
        best: tuple[float, Surface] | None = None
        for surface in self.surfaces.values():
            if surface.kind == "wall":
                along = surface.along(point)
                if not surface.extent[0] - WALL_GAP_CM <= along <= surface.extent[1] + WALL_GAP_CM:
                    continue
                score = abs(surface.offset(point)) / wall_tolerance
            else:
                score = float(np.linalg.norm(point - surface.centroid)) / feature_tolerance
            if score <= 1.0 and (best is None or score < best[0]):
                best = (score, surface)
        if best is None:
            return False
        best[1].add(point[None, :], np.array([weight]), scan_index, origin)
        return True

    def _carve(self, scan: Sequence[Mapping[str, object]], pose: Sequence[float], origin: np.ndarray) -> int:
        """Count misses for stored surfaces that a beam passed straight through."""
        if not self.surfaces:
            return 0
        missed: set[int] = set()
        cone = math.tan(math.radians(SONAR_BEAM_WIDTH_DEGREES / 4))
        incidence_limit = math.cos(math.radians(CARVE_INCIDENCE_DEGREES))
        for item in scan:
            range_cm = _valid_range(item.get("distance_cm"))
            if range_cm is None:
                continue
            range_cm = min(range_cm, SONAR_MAX_RANGE_CM)
            ray = direction(float(pose[2]) + float(item["angle_deg"]))
            free_until = range_cm - (10.0 + 0.05 * range_cm)
            for surface in self.surfaces.values():
                if surface.id in missed:
                    continue
                if surface.kind == "wall":
                    if abs(float(ray @ surface.normal)) < incidence_limit or surface.length_cm < 6.0:
                        continue
                    matrix = np.column_stack([ray, -surface.line_direction])
                    if abs(np.linalg.det(matrix)) < 1e-6:
                        continue
                    distance, along = np.linalg.solve(matrix, surface.centroid - origin)
                    if 0 < distance < free_until and surface.extent[0] + 3 <= along <= surface.extent[1] - 3:
                        missed.add(surface.id)
                else:
                    offset = surface.centroid - origin
                    distance = float(offset @ ray)
                    lateral = float(np.linalg.norm(offset - distance * ray))
                    if 0 < distance < free_until and lateral < distance * cone:
                        missed.add(surface.id)
        retired = 0
        for surface_id in missed:
            surface = self.surfaces[surface_id]
            surface.misses += 1
            if surface.misses >= 2 and surface.misses >= surface.observations:
                del self.surfaces[surface_id]
                retired += 1
        return retired

    def _consolidate(self) -> None:
        changed = True
        while changed:
            changed = False
            walls = sorted(self.walls(), key=lambda wall: -wall.observations)
            for index, first in enumerate(walls):
                for second in walls[index + 1 :]:
                    if self._walls_coincide(first, second):
                        first.absorb(self.surfaces.pop(second.id))
                        changed = True
                        break
                if changed:
                    break

        for feature in self.features():
            for wall in self.walls():
                along = wall.along(feature.centroid)
                if (
                    abs(wall.offset(feature.centroid)) <= 5.0
                    and wall.extent[0] - 15.0 <= along <= wall.extent[1] + 15.0
                ):
                    wall.absorb(self.surfaces.pop(feature.id))
                    break

        changed = True
        while changed:
            changed = False
            features = self.features()
            for index, first in enumerate(features):
                for second in features[index + 1 :]:
                    if np.linalg.norm(first.centroid - second.centroid) <= FEATURE_RADIUS_CM:
                        first.absorb(self.surfaces.pop(second.id))
                        changed = True
                        break
                if changed:
                    break
        self._promote_collinear_features()

    def _walls_coincide(self, first: Surface, second: Surface) -> bool:
        if _angle_between(first.line_direction, second.line_direction) > WALL_MERGE_ANGLE_DEGREES:
            return False
        offsets = [abs(first.offset(point)) for point in second.endpoints] + [
            abs(second.offset(point)) for point in first.endpoints
        ]
        if max(offsets) > 5.0 + first.rms_cm + second.rms_cm:
            return False
        span = sorted(first.along(point) for point in second.endpoints)
        return _interval_gap(first.extent, (span[0], span[1])) <= WALL_GAP_CM

    def _promote_collinear_features(self) -> None:
        """Turn three or more features that lie on one short line into a wall."""
        features = self.features()
        used: set[int] = set()
        for index, first in enumerate(features):
            for second in features[index + 1 :]:
                if first.id in used or second.id in used:
                    continue
                chord = second.centroid - first.centroid
                length = float(np.linalg.norm(chord))
                if not FEATURE_RADIUS_CM <= length <= 60.0:
                    continue
                unit = chord / length
                normal = np.array([-unit[1], unit[0]])
                members = [
                    feature
                    for feature in features
                    if feature.id not in used
                    and abs(float((feature.centroid - first.centroid) @ normal)) <= 3.0
                    and -15.0 <= float((feature.centroid - first.centroid) @ unit) <= length + 15.0
                ]
                if len(members) < 3:
                    continue
                centres = np.array([member.centroid for member in members])
                _, _, _, rms = fit_line(centres)
                if rms > 2.0:
                    continue
                wall = Surface(self._next_id, "wall", np.empty((0, 2)), np.empty(0))
                self._next_id += 1
                for member in members:
                    wall.absorb(self.surfaces.pop(member.id))
                    used.add(member.id)
                wall.kind = "wall"
                wall.refit()
                self.surfaces[wall.id] = wall

    def contours(self) -> list[dict[str, object]]:
        """Join wall ends that meet at corners or continue across small gaps."""
        walls = sorted(self.walls(), key=lambda wall: wall.id)
        options: list[tuple[float, tuple[int, int], tuple[int, int], np.ndarray]] = []
        for index, first in enumerate(walls):
            for second in walls[index + 1 :]:
                angle = _angle_between(first.line_direction, second.line_direction)
                corner = _intersection(first, second) if angle > 20.0 else None
                best = None
                for first_end, first_point in enumerate(first.endpoints):
                    for second_end, second_point in enumerate(second.endpoints):
                        if corner is not None and angle >= 45.0:
                            # Smooth walls stop echoing near the corner where
                            # they meet, so extend both to their intersection
                            # when it lies a little past both ends.
                            if (
                                np.linalg.norm(corner - first_point) > CORNER_EXTEND_CM
                                or np.linalg.norm(corner - second_point) > CORNER_EXTEND_CM
                                or not _beyond_end(first, first_end, corner)
                                or not _beyond_end(second, second_end, corner)
                            ):
                                continue
                            vertex = corner
                        else:
                            if np.linalg.norm(first_point - second_point) > CORNER_JOIN_CM:
                                continue
                            vertex = (first_point + second_point) / 2
                            if corner is not None:
                                if (
                                    np.linalg.norm(corner - first_point) > CORNER_JOIN_CM
                                    or np.linalg.norm(corner - second_point) > CORNER_JOIN_CM
                                ):
                                    continue
                                vertex = corner
                        cost = float(np.linalg.norm(vertex - first_point) + np.linalg.norm(vertex - second_point))
                        if best is None or cost < best[0]:
                            best = (cost, (first.id, first_end), (second.id, second_end), vertex)
                # Two walls meet at one corner at most.
                if best is not None:
                    options.append(best)

        joints: dict[tuple[int, int], tuple[tuple[int, int], np.ndarray]] = {}
        for _, first_end, second_end, vertex in sorted(options, key=lambda option: option[0]):
            if first_end in joints or second_end in joints:
                continue
            joints[first_end] = (second_end, vertex)
            joints[second_end] = (first_end, vertex)

        by_id = {wall.id: wall for wall in walls}

        def vertex(end: tuple[int, int]) -> np.ndarray:
            return joints[end][1] if end in joints else by_id[end[0]].endpoints[end[1]]

        visited: set[int] = set()
        contours: list[dict[str, object]] = []

        def walk(start: tuple[int, int]) -> dict[str, object]:
            wall_id, end = start
            points = [vertex(start)]
            ids: list[int] = []
            closed = False
            while True:
                visited.add(wall_id)
                ids.append(wall_id)
                far_end = (wall_id, 1 - end)
                points.append(vertex(far_end))
                if far_end not in joints:
                    break
                next_wall, next_end = joints[far_end][0]
                if next_wall in visited:
                    closed = next_wall == ids[0]
                    break
                wall_id, end = next_wall, next_end
            if closed:
                points.pop()
            return {
                "points": [[round(float(point[0]), 1), round(float(point[1]), 1)] for point in points],
                "closed": closed,
                "surface_ids": ids,
            }

        for wall in walls:
            for end in (0, 1):
                if wall.id not in visited and (wall.id, end) not in joints:
                    contours.append(walk((wall.id, end)))
        for wall in walls:
            if wall.id not in visited:
                contours.append(walk((wall.id, 0)))
        return contours

    def to_dict(self) -> dict[str, object]:
        def rounded(point: np.ndarray) -> list[float]:
            return [round(float(point[0]), 1), round(float(point[1]), 1)]

        return {
            "scan_count": self.scan_count,
            "surfaces": [
                {
                    "id": wall.id,
                    "start": rounded(wall.endpoints[0]),
                    "end": rounded(wall.endpoints[1]),
                    "normal": [round(float(wall.normal[0]), 3), round(float(wall.normal[1]), 3)],
                    "length_cm": round(wall.length_cm, 1),
                    "rms_cm": round(wall.rms_cm, 2),
                    "points": len(wall.points),
                    "observations": wall.observations,
                    "misses": wall.misses,
                    "confidence": round(wall.confidence, 3),
                    "confirmed": wall.confirmed,
                }
                for wall in sorted(self.walls(), key=lambda wall: wall.id)
            ],
            "features": [
                {
                    "id": feature.id,
                    "position": rounded(feature.centroid),
                    "radius_cm": round(feature.rms_cm, 2),
                    "observations": feature.observations,
                    "misses": feature.misses,
                    "confidence": round(feature.confidence, 3),
                    "confirmed": feature.confirmed,
                }
                for feature in sorted(self.features(), key=lambda feature: feature.id)
            ],
            "contours": self.contours(),
            "echoes": [[round(x, 1), round(y, 1), kind] for x, y, kind in self.recent_echoes[-400:]],
        }
