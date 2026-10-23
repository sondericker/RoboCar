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

# Starting beam width; each map re-measures it from flat-wall profiles
# because the effective HC-SR04 beam on real walls is often much wider.
SONAR_BEAM_WIDTH_DEGREES = 25.0
BEAM_WIDTH_LIMITS_DEGREES = (10.0, 40.0)
BEAM_WIDTH_MIN_SAMPLES = 6
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
MAX_FREE_RAYS = 4000
WALL_ANGLE_TOLERANCE_DEGREES = 15.0
WALL_MERGE_ANGLE_DEGREES = 12.0
# Collinear wall pieces up to a doorway apart share one line model, but are
# drawn as separate segments wherever their supporting points leave a gap.
WALL_MERGE_GAP_CM = 60.0
SEGMENT_GAP_CM = 10.0
POINT_JOIN_MARGIN_CM = 6.0
FEATURE_RADIUS_CM = 8.0
CORNER_JOIN_CM = 25.0
CORNER_EXTEND_CM = 45.0
CARVE_INCIDENCE_DEGREES = 30.0


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
    # Half-width of the surface this echo vouches for, along the surface.
    footprint_cm: float = 0.0


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


def _make_echo(
    angle: float,
    bearing: float,
    range_cm: float,
    kind: str,
    footprint_cm: float,
    half_beam: float,
) -> Echo:
    bearing_sigma = {
        "plane": 2.5,
        "point": 2.5,
        "edge": 2.5,
        "oblique": 3.5,
    }.get(kind, half_beam)
    sigma = math.hypot(1.0 + 0.01 * range_cm, range_cm * math.radians(bearing_sigma))
    x, y = np.array([0.0, SONAR_FORWARD_OFFSET_CM]) + range_cm * direction(bearing)
    return Echo(angle, bearing, range_cm, float(x), float(y), kind, sigma, footprint_cm)


def _occluding_edges(
    run: Sequence[tuple[float, float]],
    sweep: Sequence[tuple[float, float | None]],
    half_beam: float,
) -> tuple[float | None, float | None]:
    """Estimate the bearings of a run's free edges, where the sweep jumps farther.

    The near surface keeps answering while its edge is anywhere inside the
    cone, so the edge lies half a beam back from where the range jumped.
    """
    angles = [angle for angle, _ in sweep]
    edges: list[float | None] = []
    for end, neighbour_offset, sign in ((run[0], -1, 1), (run[-1], 1, -1)):
        position = angles.index(end[0]) + neighbour_offset
        if not 0 <= position < len(sweep):
            edges.append(None)
            continue
        neighbour_angle, neighbour_range = sweep[position]
        if neighbour_range is not None and neighbour_range <= end[1] + _jump_tolerance(end[1], neighbour_range):
            # The neighbour is nearer: something else hides this run's end.
            edges.append(None)
            continue
        edges.append((end[0] + neighbour_angle) / 2 + sign * half_beam)
    return edges[0], edges[1]


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


def _cut_ends(
    run: Sequence[tuple[float, float]],
    sweep: Sequence[tuple[float, float | None]],
) -> tuple[bool, bool]:
    """Whether each end of a run is cut short by the sweep limit or a nearer object."""
    angles = [angle for angle, _ in sweep]
    cut: list[bool] = []
    for end, neighbour_offset in ((run[0], -1), (run[-1], 1)):
        position = angles.index(end[0]) + neighbour_offset
        if not 0 <= position < len(sweep):
            cut.append(True)
            continue
        neighbour_range = sweep[position][1]
        cut.append(neighbour_range is not None and neighbour_range < end[1] - _jump_tolerance(end[1], neighbour_range))
    return cut[0], cut[1]


def extract_echoes(
    scan: Sequence[Mapping[str, object]],
    beam_width_deg: float = SONAR_BEAM_WIDTH_DEGREES,
) -> list[list[Echo]]:
    """Return beam-corrected echoes, one list per continuous run of the sweep."""
    half_beam = beam_width_deg / 2
    sweep = sorted(
        (float(item["angle_deg"]), _valid_range(item.get("distance_cm"))) for item in scan
    )
    step = math.radians(_sweep_step([angle for angle, _ in sweep]))
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
            for side in (-1, 1):
                cursor = seed + side
                while 0 <= cursor < len(run) and owner[cursor] < 0 and abs(run[cursor][1] - run[seed][1]) <= tolerance:
                    owner[cursor] = seed
                    cursor += side
        grown = [
            [run[index] for index in range(len(run)) if owner[index] == seed]
            for seed in sorted(seeds)
        ]
        # A true region of constant depth is a local minimum; readings that
        # merely sit close together on a gentle slope stay separate.
        groups: list[list[tuple[float, float]]] = []
        for index, group in enumerate(grown):
            level = float(np.median([item[1] for item in group]))
            adjacent = [grown[other] for other in (index - 1, index + 1) if 0 <= other < len(grown)]
            if len(group) > 1 and any(float(np.median([item[1] for item in other])) < level for other in adjacent):
                groups.extend([item] for item in group)
            else:
                groups.append(group)
        centres = [
            (float(np.mean([item[0] for item in group])), float(np.median([item[1] for item in group])))
            for group in groups
        ]
        # A flat region cut off by the end of the sweep, or hidden by something
        # nearer, does not show both of its edges, so its middle is not the
        # surface normal. The normal lies at least half a beam beyond the
        # edge that was seen.
        cut_low, cut_high = _cut_ends(run, sweep)
        if len(groups[0]) > 1 and cut_low and not (cut_high and len(groups) == 1):
            centres[0] = (min(centres[0][0], groups[0][-1][0] - half_beam), centres[0][1])
        if len(groups[-1]) > 1 and cut_high and not (cut_low and len(groups) == 1):
            centres[-1] = (max(centres[-1][0], groups[-1][0][0] + half_beam), centres[-1][1])

        echoes: list[Echo] = []
        for index, group in enumerate(groups):
            angle, range_cm = centres[index]
            before = centres[index - 1][1] if index > 0 else None
            after = centres[index + 1][1] if index + 1 < len(centres) else None
            neighbours = [value for value in (before, after) if value is not None]
            if len(neighbours) == 2 and all(value < range_cm for value in neighbours):
                # A range peak between two nearer readings is where two
                # surfaces meet inside the beam (a concave corner); the
                # neighbouring readings already describe both surfaces.
                continue
            if len(group) > 1 or (len(neighbours) == 2 and all(value > range_cm for value in neighbours)):
                # A region of constant depth: the surface faces the sensor at
                # its centre. Without continuous neighbours it is an isolated
                # point reflector such as an edge or a chair leg.
                if not neighbours:
                    echoes.append(_make_echo(angle, angle, range_cm, "point", range_cm * step / 2, half_beam))
                    continue
                echoes.append(_make_echo(angle, angle, range_cm, "plane", range_cm * step, half_beam))
                # Readings farther than half a beam from the centre saw the
                # plane at the cone edge nearest its normal; placing each one
                # there lays the whole region onto the surface.
                for member_angle, member_range in group:
                    offset = member_angle - angle
                    if abs(offset) > half_beam:
                        bearing = member_angle - math.copysign(half_beam, offset)
                        echoes.append(_make_echo(member_angle, bearing, member_range, "plane", member_range * step, half_beam))
                continue
            if not neighbours:
                echoes.append(_make_echo(angle, angle, range_cm, "isolated", range_cm * step / 2, half_beam))
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
            echoes.append(_make_echo(angle, angle + toward * shift, range_cm, "oblique", range_cm * step, half_beam))

        echoes.sort(key=lambda echo: echo.bearing_deg)
        left_edge, right_edge = _occluding_edges(run, sweep, half_beam)
        if left_edge is not None and right_edge is not None and left_edge > right_edge:
            # Narrower than the beam (a leg or post): both edges meet.
            left_edge = right_edge = (left_edge + right_edge) / 2
        clamped: list[Echo] = []
        for echo in echoes:
            bearing = echo.bearing_deg
            if left_edge is not None and bearing < left_edge:
                bearing = left_edge
            if right_edge is not None and bearing > right_edge:
                bearing = right_edge
            if bearing != echo.bearing_deg:
                # Beam-edge bleed past a jamb or object edge: the echo came
                # from the edge itself, not from the opening beyond it.
                echo = _make_echo(echo.angle_deg, bearing, echo.range_cm, "edge", echo.range_cm * step / 2, half_beam)
            clamped.append(echo)
        echo_runs.append(clamped)
    return echo_runs


def estimate_beam_widths(scan: Sequence[Mapping[str, object]]) -> list[float]:
    """Measure the beam width from runs that look like one flat wall.

    A plane reads its perpendicular distance d while its normal is inside the
    cone (a flat top one beam wide), then d / cos(angle beyond the cone edge).
    Fitting that profile to a run gives the beam width; runs that fit poorly
    (corners, clutter) or lack a flat top with rising flanks are skipped.
    """
    widths: list[float] = []
    halves = np.arange(3.0, 25.01, 0.5)
    for run in split_runs(scan):
        if len(run) < 6:
            continue
        angles = np.array([angle for angle, _ in run])
        ranges = np.array([range_cm for _, range_cm in run])
        centres = np.arange(angles.min() - 30.0, angles.max() + 30.01, 1.0)
        beyond = np.abs(angles[None, None, :] - centres[None, :, None]) - halves[:, None, None]
        excess = np.radians(np.clip(beyond, 0.0, None))
        cosine = np.cos(excess)
        distance = np.median(ranges * cosine, axis=2)
        residual = ranges - distance[..., None] / cosine
        cost = (residual**2).sum(axis=2)
        cost[(excess >= np.radians(70)).any(axis=2)] = np.inf
        half_index, centre_index = np.unravel_index(int(np.argmin(cost)), cost.shape)
        rms = math.sqrt(float(cost[half_index, centre_index]) / len(run))
        best_excess = excess[half_index, centre_index]
        centre = centres[centre_index]
        # Both edges of the flat top must be seen; with one flank the beam
        # width and the wall's direction trade off and cannot be told apart.
        sloped = best_excess > math.radians(4)
        if (
            rms <= max(0.8, 0.01 * float(distance[half_index, centre_index]))
            and int((best_excess == 0).sum()) >= 2
            and int((sloped & (angles < centre)).sum()) >= 2
            and int((sloped & (angles > centre)).sum()) >= 2
        ):
            widths.append(2 * float(halves[half_index]))
    return widths


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
    footprints: np.ndarray = field(default_factory=lambda: np.empty(0))
    point_misses: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    scans: set[int] = field(default_factory=set)
    misses: int = 0
    view_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))
    line_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    extent: tuple[float, float] = (0.0, 0.0)
    # Stretches of the wall line with continuous support; gaps between them
    # are openings or unexplored space.
    pieces: list[tuple[float, float]] = field(default_factory=list)
    rms_cm: float = 0.0

    def add(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        footprints: np.ndarray,
        scan_index: int,
        origin: np.ndarray,
    ) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self._append(points, weights, footprints, np.zeros(len(points), dtype=int))
        self.scans.add(scan_index)
        self.view_sum = self.view_sum + (origin - points.mean(axis=0))
        self.refit()

    def absorb(self, other: "Surface") -> None:
        self._append(other.points, other.weights, other.footprints, other.point_misses)
        self.scans |= other.scans
        self.misses += other.misses
        self.view_sum = self.view_sum + other.view_sum
        self.refit()

    def keep_points(self, mask: np.ndarray) -> None:
        self.points = self.points[mask]
        self.weights = self.weights[mask]
        self.footprints = self.footprints[mask]
        self.point_misses = self.point_misses[mask]
        if len(self.points):
            self.refit()

    def _append(self, points: np.ndarray, weights: np.ndarray, footprints: np.ndarray, misses: np.ndarray) -> None:
        self.points = np.vstack([self.points, points])[-MAX_SURFACE_POINTS:]
        self.weights = np.concatenate([self.weights, weights])[-MAX_SURFACE_POINTS:]
        self.footprints = np.concatenate([self.footprints, footprints])[-MAX_SURFACE_POINTS:]
        self.point_misses = np.concatenate([self.point_misses, misses]).astype(int)[-MAX_SURFACE_POINTS:]

    def refit(self) -> None:
        if self.kind == "feature" or len(self.points) < 2:
            total = float(self.weights.sum())
            self.centroid = (self.points * self.weights[:, None]).sum(axis=0) / total
            spread = np.linalg.norm(self.points - self.centroid, axis=1)
            self.rms_cm = float(math.sqrt((self.weights * spread**2).sum() / total))
            self.extent = (0.0, 0.0)
            self.pieces = [(0.0, 0.0)]
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
        self.centroid, self.line_direction, self.normal, self.rms_cm = centroid, line_direction, normal, rms

        along = (self.points[inliers] - centroid) @ line_direction
        order = np.argsort(along)
        along, footprints = along[order], self.footprints[inliers][order]
        runs: list[tuple[float, float, int]] = []
        start, count = float(along[0]), 1
        for index in range(1, len(along)):
            allowed = max(SEGMENT_GAP_CM, 1.5 * (footprints[index] + footprints[index - 1]))
            if along[index] - along[index - 1] > allowed:
                runs.append((start, float(along[index - 1]), count))
                start, count = float(along[index]), 0
            count += 1
        runs.append((start, float(along[-1]), count))
        # A lone point beyond a gap is not yet a surface; it still steadies the line.
        supported = [(low, high) for low, high, members in runs if members >= 2]
        self.pieces = supported or [(float(along[0]), float(along[-1]))]
        self.extent = (self.pieces[0][0], self.pieces[-1][1])

    @property
    def observations(self) -> int:
        return len(self.scans)

    @property
    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.centroid + self.extent[0] * self.line_direction,
            self.centroid + self.extent[1] * self.line_direction,
        )

    def piece_endpoints(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (self.centroid + low * self.line_direction, self.centroid + high * self.line_direction)
            for low, high in self.pieces
        ]

    def near_piece(self, along: float, margin: float) -> bool:
        return any(low - margin <= along <= high + margin for low, high in self.pieces)

    @property
    def length_cm(self) -> float:
        return self.extent[1] - self.extent[0]

    @property
    def covered_cm(self) -> float:
        return sum(high - low for low, high in self.pieces)

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


def _intersection(first: Surface, second: Surface) -> np.ndarray | None:
    matrix = np.column_stack([first.line_direction, -second.line_direction])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None
    scale = np.linalg.solve(matrix, second.centroid - first.centroid)
    return first.centroid + scale[0] * first.line_direction


@dataclass(frozen=True)
class _Segment:
    key: tuple[int, int]
    wall: Surface
    low: float
    high: float

    def end(self, which: int) -> np.ndarray:
        return self.wall.centroid + (self.low if which == 0 else self.high) * self.wall.line_direction

    def beyond_end(self, which: int, point: np.ndarray) -> bool:
        along = self.wall.along(point)
        return along <= self.low + 5.0 if which == 0 else along >= self.high - 5.0


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
        # Recent beam centre lines (origin, direction, clear distance): space
        # the sound crossed, which contours may not close over.
        self.free_rays = np.empty((0, 5))
        self.beam_width_deg = SONAR_BEAM_WIDTH_DEGREES
        self._beam_samples: list[float] = []
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
        self._beam_samples = (self._beam_samples + estimate_beam_widths(scan))[-60:]
        if len(self._beam_samples) >= BEAM_WIDTH_MIN_SAMPLES:
            low, high = BEAM_WIDTH_LIMITS_DEGREES
            self.beam_width_deg = min(high, max(low, float(np.median(self._beam_samples))))
        retired = self._carve(scan, pose, origin)
        self._remember_free_rays(scan, pose, origin)

        echo_runs = extract_echoes(scan, self.beam_width_deg)
        heading_sigma = math.radians(heading_sigma_deg)
        candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        singles: list[tuple[np.ndarray, float, float, float]] = []
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
            footprints = np.array([echo.footprint_cm for echo in echoes])
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
                    candidates.append((piece_points, weights[piece], footprints[piece], reach))
                else:
                    singles.extend(
                        (world[index], float(weights[index]), float(footprints[index]), echoes[index].range_cm)
                        for index in piece
                    )
        self.recent_echoes = self.recent_echoes[-MAX_RECENT_ECHOES:]

        fused = created = 0
        for points, weights, footprints, reach in candidates:
            if self._fuse_line(points, weights, footprints, reach, position_sigma_cm, scan_index, origin):
                fused += 1
            else:
                self._create("wall", points, weights, footprints, scan_index, origin)
                created += 1
        for point, weight, footprint, reach in singles:
            if self._fuse_point(point, weight, footprint, reach, position_sigma_cm, scan_index, origin):
                fused += 1
            else:
                self._create("feature", point[None, :], np.array([weight]), np.array([footprint]), scan_index, origin)
                created += 1
        self._consolidate()
        return IntegrationReport(scan_index, echo_count, len(candidates), fused, created, retired)

    def _create(
        self,
        kind: str,
        points: np.ndarray,
        weights: np.ndarray,
        footprints: np.ndarray,
        scan_index: int,
        origin: np.ndarray,
    ) -> Surface:
        surface = Surface(self._next_id, kind, np.empty((0, 2)), np.empty(0))
        self._next_id += 1
        surface.add(points, weights, footprints, scan_index, origin)
        self.surfaces[surface.id] = surface
        return surface

    def _fuse_line(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        footprints: np.ndarray,
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
            if _interval_gap(wall.extent, (span[0], span[1])) > WALL_MERGE_GAP_CM:
                continue
            matches.append(wall)
        if not matches:
            return False
        target = max(matches, key=lambda wall: wall.observations)
        for wall in matches:
            if wall is not target:
                target.absorb(self.surfaces.pop(wall.id))
        target.add(points, weights, footprints, scan_index, origin)
        return True

    def _fuse_point(
        self,
        point: np.ndarray,
        weight: float,
        footprint: float,
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
                # Only extend a wall right next to a stretch it already covers,
                # so surfaces cannot creep across an opening sweep by sweep.
                margin = max(POINT_JOIN_MARGIN_CM, 2 * footprint)
                if not surface.near_piece(surface.along(point), margin):
                    continue
                score = abs(surface.offset(point)) / wall_tolerance
            else:
                score = float(np.linalg.norm(point - surface.centroid)) / feature_tolerance
            if score <= 1.0 and (best is None or score < best[0]):
                best = (score, surface)
        if best is None:
            return False
        best[1].add(point[None, :], np.array([weight]), np.array([footprint]), scan_index, origin)
        return True

    def _remember_free_rays(self, scan: Sequence[Mapping[str, object]], pose: Sequence[float], origin: np.ndarray) -> None:
        rays = []
        for item in scan:
            range_cm = _valid_range(item.get("distance_cm"))
            if range_cm is None:
                continue
            ray = direction(float(pose[2]) + float(item["angle_deg"]))
            clear = min(range_cm, SONAR_MAX_RANGE_CM) - (5.0 + 0.03 * range_cm)
            if clear > 0:
                rays.append((origin[0], origin[1], ray[0], ray[1], clear))
        if rays:
            self.free_rays = np.vstack([self.free_rays, np.array(rays)])[-MAX_FREE_RAYS:]

    def crossed_by_free_ray(self, start: np.ndarray, end: np.ndarray, from_fraction: float = 0.0) -> bool:
        """True if a remembered beam passed clean through the segment start-end."""
        if not len(self.free_rays):
            return False
        edge = end - start
        origins, rays, clear = self.free_rays[:, :2], self.free_rays[:, 2:4], self.free_rays[:, 4]
        denominator = rays[:, 0] * edge[1] - rays[:, 1] * edge[0]
        usable = np.abs(denominator) > 1e-9
        offset = start - origins
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (offset[:, 0] * edge[1] - offset[:, 1] * edge[0]) / denominator
            fraction = (offset[:, 0] * rays[:, 1] - offset[:, 1] * rays[:, 0]) / denominator
        crossing = usable & (distance > 0) & (distance < clear) & (fraction > from_fraction) & (fraction <= 1.0)
        return bool(crossing.any())

    def _carve(self, scan: Sequence[Mapping[str, object]], pose: Sequence[float], origin: np.ndarray) -> int:
        """Use beams that passed through stored surfaces as evidence against them.

        Wall points inside a beam's core that the beam passed well beyond are
        dropped after two such misses, which opens gaps where there are real
        openings; whole surfaces also lose confidence and can be retired.
        """
        if not self.surfaces:
            return 0
        missed: set[int] = set()
        point_hits: dict[int, set[int]] = {}
        core = math.tan(math.radians(self.beam_width_deg / 4))
        incidence_limit = math.cos(math.radians(CARVE_INCIDENCE_DEGREES))
        for item in scan:
            range_cm = _valid_range(item.get("distance_cm"))
            if range_cm is None:
                continue
            range_cm = min(range_cm, SONAR_MAX_RANGE_CM)
            ray = direction(float(pose[2]) + float(item["angle_deg"]))
            free_until = range_cm - (10.0 + 0.05 * range_cm)
            for surface in self.surfaces.values():
                if surface.kind == "wall":
                    if abs(float(ray @ surface.normal)) < incidence_limit:
                        continue
                    matrix = np.column_stack([ray, -surface.line_direction])
                    if abs(np.linalg.det(matrix)) < 1e-6:
                        continue
                    distance, along = np.linalg.solve(matrix, surface.centroid - origin)
                    if not 0 < distance < free_until:
                        continue
                    projections = (surface.points - surface.centroid) @ surface.line_direction
                    window = distance * core + 2.0
                    crossed = np.nonzero(np.abs(projections - along) <= window)[0]
                    if len(crossed):
                        point_hits.setdefault(surface.id, set()).update(crossed.tolist())
                    if surface.near_piece(along, -3.0):
                        missed.add(surface.id)
                else:
                    offset = surface.centroid - origin
                    distance = float(offset @ ray)
                    lateral = float(np.linalg.norm(offset - distance * ray))
                    if 0 < distance < free_until and lateral < distance * core:
                        missed.add(surface.id)
        retired = 0
        for surface_id in missed:
            self.surfaces[surface_id].misses += 1
        for surface_id, indices in point_hits.items():
            surface = self.surfaces[surface_id]
            surface.point_misses[sorted(indices)] += 1
            keep = surface.point_misses < 2
            if not keep.all():
                surface.keep_points(keep)
        for surface_id in missed | set(point_hits):
            surface = self.surfaces[surface_id]
            emptied = surface.kind == "wall" and len(surface.points) < 2
            if emptied or (surface.misses >= 2 and surface.misses >= surface.observations):
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
                if abs(wall.offset(feature.centroid)) <= 5.0 and wall.near_piece(
                    wall.along(feature.centroid), POINT_JOIN_MARGIN_CM + 4.0
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
        return _interval_gap(first.extent, (span[0], span[1])) <= WALL_MERGE_GAP_CM

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
        """Join segment ends that meet at corners; openings within a wall stay open."""
        segments = [
            _Segment((wall.id, index), wall, low, high)
            for wall in sorted(self.walls(), key=lambda wall: wall.id)
            for index, (low, high) in enumerate(wall.pieces)
        ]
        options: list[tuple[float, tuple[tuple[int, int], int], tuple[tuple[int, int], int], np.ndarray]] = []
        for index, first in enumerate(segments):
            for second in segments[index + 1 :]:
                if first.wall is second.wall:
                    continue
                angle = _angle_between(first.wall.line_direction, second.wall.line_direction)
                corner = _intersection(first.wall, second.wall) if angle > 20.0 else None
                best = None
                for first_end in (0, 1):
                    first_point = first.end(first_end)
                    for second_end in (0, 1):
                        second_point = second.end(second_end)
                        if corner is not None and angle >= 45.0:
                            # Smooth walls stop echoing near the corner where
                            # they meet, so extend both to their intersection
                            # when it lies a little past both ends.
                            if (
                                np.linalg.norm(corner - first_point) > CORNER_EXTEND_CM
                                or np.linalg.norm(corner - second_point) > CORNER_EXTEND_CM
                                or not first.beyond_end(first_end, corner)
                                or not second.beyond_end(second_end, corner)
                                or self.crossed_by_free_ray(first_point, corner, 0.15)
                                or self.crossed_by_free_ray(second_point, corner, 0.15)
                            ):
                                continue
                            vertex = corner
                        else:
                            if np.linalg.norm(first_point - second_point) > CORNER_JOIN_CM:
                                continue
                            if self.crossed_by_free_ray(first_point, second_point, 0.1):
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
                        # Confirmed walls claim their corners before fragments seen once.
                        cost += 100.0 * ((not first.wall.confirmed) + (not second.wall.confirmed))
                        if best is None or cost < best[0]:
                            best = (cost, (first.key, first_end), (second.key, second_end), vertex)
                # Two segments meet at one corner at most.
                if best is not None:
                    options.append(best)

        joints: dict[tuple[tuple[int, int], int], tuple[tuple[tuple[int, int], int], np.ndarray]] = {}
        for _, first_end, second_end, vertex in sorted(options, key=lambda option: option[0]):
            if first_end in joints or second_end in joints:
                continue
            joints[first_end] = (second_end, vertex)
            joints[second_end] = (first_end, vertex)

        by_key = {segment.key: segment for segment in segments}

        def vertex(end: tuple[tuple[int, int], int]) -> np.ndarray:
            return joints[end][1] if end in joints else by_key[end[0]].end(end[1])

        visited: set[tuple[int, int]] = set()
        contours: list[dict[str, object]] = []

        def walk(start: tuple[tuple[int, int], int]) -> dict[str, object]:
            key, end = start
            points = [vertex(start)]
            keys: list[tuple[int, int]] = []
            closed = False
            while True:
                visited.add(key)
                keys.append(key)
                far_end = (key, 1 - end)
                points.append(vertex(far_end))
                if far_end not in joints:
                    break
                next_key, next_end = joints[far_end][0]
                if next_key in visited:
                    closed = next_key == keys[0]
                    break
                key, end = next_key, next_end
            if closed:
                points.pop()
            return {
                "points": [[round(float(point[0]), 1), round(float(point[1]), 1)] for point in points],
                "closed": closed,
                "surface_ids": [key[0] for key in keys],
            }

        for segment in segments:
            for end in (0, 1):
                if segment.key not in visited and (segment.key, end) not in joints:
                    contours.append(walk((segment.key, end)))
        for segment in segments:
            if segment.key not in visited:
                contours.append(walk((segment.key, 0)))
        return contours

    def to_dict(self) -> dict[str, object]:
        def rounded(point: np.ndarray) -> list[float]:
            return [round(float(point[0]), 1), round(float(point[1]), 1)]

        return {
            "scan_count": self.scan_count,
            "beam_width_deg": round(self.beam_width_deg, 1),
            "surfaces": [
                {
                    "id": wall.id,
                    "start": rounded(wall.endpoints[0]),
                    "end": rounded(wall.endpoints[1]),
                    "segments": [[rounded(start), rounded(end)] for start, end in wall.piece_endpoints()],
                    "normal": [round(float(wall.normal[0]), 3), round(float(wall.normal[1]), 3)],
                    "length_cm": round(wall.length_cm, 1),
                    "covered_cm": round(wall.covered_cm, 1),
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
