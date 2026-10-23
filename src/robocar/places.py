"""Remembering places: keyframes, place recognition, loop closure and a pose graph.

Every mapping stop becomes a keyframe: ORB features of its forward picture,
its sonar sweep and its pose. At each new stop:

1. Retrieval. The new picture's descriptors vote across every stored
   keyframe (all sessions); the best-voted places become candidates.
2. Verification. Each candidate is matched feature by feature and must fit a
   camera motion (essential matrix, or a homography for a pure turn) with
   enough RANSAC inliers. That yields the turn between the two views and the
   direction from one to the other, but not the distance.
3. Metric alignment. The new sweep is registered against a small surface map
   rebuilt from the remembered place's own sweeps, started from several
   offsets along the camera's direction, and rejected if two different
   offsets fit equally well (a long corridor looks the same everywhere).
4. Loop closure. The resulting relative pose becomes a constraint in a pose
   graph whose nodes are stops and whose other edges are the motion between
   consecutive stops. Optimizing the graph spreads accumulated drift over the
   whole path; constraints inconsistent with the rest are discarded.
5. Sessions. A run starts in its own frame. Recognizing a place from an
   earlier run joins the two graphs, so the new run continues in the old
   frame and both runs' sweeps build one persistent map.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from robocar.localization import Pose, compose, register_scan, relative, wrap_degrees
from robocar.surfaces import SurfaceMap, extract_echoes, pose_rotation

logger = logging.getLogger(__name__)

CAMERA_HORIZONTAL_FOV_DEGREES = 62.2
KEYFRAME_DETECT_FEATURES = 1000
KEYFRAME_KEEP_FEATURES = 400
KEYFRAME_MIN_FEATURES = 40
RETRIEVAL_MAX_DISTANCE = 48
RETRIEVAL_MIN_VOTES = 15
RETRIEVAL_CANDIDATES = 4
RETRIEVAL_CHUNK_ROWS = 200_000
VERIFY_RATIO = 0.8
VERIFY_MIN_INLIERS = 30
VERIFY_MIN_INLIER_FRACTION = 0.3
ROTATION_SINGULAR_RATIO = 1.03
# Places this many stops back in the same run are just the recent path.
RECENT_STOPS_EXCLUDED = 4
LOOP_SEARCH_RADIUS_CM = 150.0
LOCAL_MAP_NEIGHBOURS = 3
METRIC_MIN_MATCHES = 8
# The camera says which place this is but only roughly how the views are
# turned (about +-15 degrees with mostly one wall in view), so the sonar fit
# is started from a spread of turns and offsets.
METRIC_YAW_OFFSETS_DEGREES = (-10.0, 0.0, 10.0)
METRIC_BEARING_DISTANCES_CM = (25.0, 50.0, 80.0, 120.0)
METRIC_GRID_DISTANCES_CM = (35.0, 70.0)
METRIC_PRIOR_SIGMA = (25.0, 25.0, 6.0)
METRIC_AMBIGUITY_CM = 25.0
METRIC_ATTEMPTS_PER_STOP = 2
CAMERA_ONLY_MIN_INLIERS = 80
CAMERA_ONLY_POSITION_SIGMA_CM = 50.0
CAMERA_ONLY_YAW_SIGMA_DEGREES = 8.0
# After recognizing a place, skip a couple of stops before trying again.
CLOSURE_COOLDOWN_STOPS = 2
# 99.9% chi-square bound for three degrees of freedom.
LOOP_REJECT_CHI2 = 16.3
DEFAULT_EDGE_SIGMA = (3.0, 3.0, 2.0)


# The car's pictures are often dim and low in contrast (indoor light, a small
# sensor). When a normal detection finds few corners, a light blur, local
# contrast equalization and a more sensitive detector roughly tripled good
# matches between real stops. Well-exposed pictures are left alone, because
# the same treatment adds noisy corners to them.
FEATURE_ENHANCE_BELOW = 300
FEATURE_DENOISE_SIGMA = 1.0
FEATURE_CLAHE_CLIP = 3.0
FEATURE_SENSITIVE_FAST_THRESHOLD = 8


def detect_features(jpeg: bytes, features: int) -> tuple[list, np.ndarray | None, tuple[int, int]] | None:
    """ORB keypoints and descriptors, enhancing dim pictures first."""
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    keypoints, descriptors = cv2.ORB_create(nfeatures=features).detectAndCompute(image, None)
    if len(keypoints) < FEATURE_ENHANCE_BELOW:
        enhanced = cv2.createCLAHE(clipLimit=FEATURE_CLAHE_CLIP, tileGridSize=(8, 8)).apply(
            cv2.GaussianBlur(image, (0, 0), FEATURE_DENOISE_SIGMA)
        )
        detector = cv2.ORB_create(nfeatures=features, fastThreshold=FEATURE_SENSITIVE_FAST_THRESHOLD)
        keypoints, descriptors = detector.detectAndCompute(enhanced, None)
    return list(keypoints), descriptors, image.shape


def camera_matrix(width: int, height: int) -> np.ndarray:
    focal = width / (2 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEGREES) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)


@dataclass
class Keyframe:
    observation_id: int
    run_id: str
    pan_offset_deg: float
    width: int
    height: int
    points: np.ndarray
    descriptors: np.ndarray


def make_keyframe(observation_id: int, run_id: str, jpeg: bytes, pan_offset_deg: float) -> Keyframe | None:
    """Detect ORB features in a stop's forward picture and keep the strongest."""
    detected = detect_features(jpeg, KEYFRAME_DETECT_FEATURES)
    if detected is None:
        return None
    keypoints, descriptors, (height, width) = detected
    if descriptors is None or len(keypoints) < KEYFRAME_MIN_FEATURES:
        return None
    strongest = np.argsort([-keypoint.response for keypoint in keypoints])[:KEYFRAME_KEEP_FEATURES]
    points = np.float32([keypoints[index].pt for index in strongest])
    return Keyframe(observation_id, run_id, pan_offset_deg, width, height, points, descriptors[strongest])


@dataclass(frozen=True)
class PlaceMatch:
    """How the car at the query stop sits relative to a remembered keyframe."""

    keyframe_id: int
    yaw_deg: float
    bearing_deg: float | None
    inliers: int
    matches: int


def _decompose_wall_homography(
    homography: np.ndarray,
    intrinsics: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Rotation and camera-centre direction from a homography of a vertical plane."""
    normalized = np.linalg.inv(intrinsics) @ homography @ intrinsics
    u, singular, vt = np.linalg.svd(normalized)
    if singular[0] / max(singular[2], 1e-9) < ROTATION_SINGULAR_RATIO:
        # Equal singular values: a pure turn with no measurable translation.
        rotation = u @ vt
        return (rotation if np.linalg.det(rotation) > 0 else -rotation), None
    count, rotations, translations, normals = cv2.decomposeHomographyMat(homography, intrinsics)
    inlier = mask.ravel().astype(bool)
    visible = cv2.filterHomographyDecompByVisibleRefpoints(
        rotations, normals, first[inlier].reshape(-1, 1, 2), second[inlier].reshape(-1, 1, 2)
    )
    candidates = [int(index) for index in visible.ravel()] if visible is not None else list(range(count))
    if not candidates:
        return None
    # Walls are vertical: their normal has no component along the camera's
    # vertical (y) axis.
    best = min(candidates, key=lambda index: abs(float(normals[index][1, 0])))
    rotation, translation = rotations[best], translations[best].ravel()
    return rotation, -rotation.T @ translation


def verify_place(remembered: Keyframe, query: Keyframe) -> PlaceMatch | None:
    """Match two keyframes and fit a camera motion between them."""
    if (remembered.width, remembered.height) != (query.width, query.height):
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(remembered.descriptors, query.descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < VERIFY_RATIO * pair[1].distance]
    if len(good) < VERIFY_MIN_INLIERS:
        return None
    first = np.float32([remembered.points[match.queryIdx] for match in good])
    second = np.float32([query.points[match.trainIdx] for match in good])
    intrinsics = camera_matrix(query.width, query.height)

    homography, homography_mask = cv2.findHomography(first, second, cv2.RANSAC, 3.0)
    homography_inliers = int(homography_mask.sum()) if homography_mask is not None else 0
    essential, essential_mask = cv2.findEssentialMat(first, second, intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    essential_inliers = 0
    rotation = translation = None
    if essential is not None and essential_mask is not None and essential.shape == (3, 3):
        essential_inliers, rotation, translation, _ = cv2.recoverPose(
            essential, first, second, intrinsics, mask=essential_mask.copy()
        )

    bearing = None
    if homography is not None and homography_inliers >= max(VERIFY_MIN_INLIERS, 0.85 * essential_inliers):
        # One plane (usually a wall) dominates, where the essential matrix is
        # unstable; decompose the homography instead.
        decomposed = _decompose_wall_homography(homography, intrinsics, first, second, homography_mask)
        if decomposed is None:
            return None
        rotation, centre = decomposed
        inliers = homography_inliers
    elif rotation is not None and essential_inliers >= VERIFY_MIN_INLIERS:
        inliers = int(essential_inliers)
        centre = -rotation.T @ translation.ravel()
    else:
        return None
    if centre is not None and math.hypot(centre[0], centre[2]) > 1e-6:
        bearing = math.degrees(math.atan2(-centre[0], centre[2]))
    if inliers < VERIFY_MIN_INLIERS or inliers < VERIFY_MIN_INLIER_FRACTION * len(good):
        return None
    camera_yaw = math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))
    # Camera heading is the car heading plus the pan offset at each stop.
    yaw = wrap_degrees(camera_yaw - query.pan_offset_deg + remembered.pan_offset_deg)
    if bearing is not None:
        bearing = wrap_degrees(bearing + remembered.pan_offset_deg)
    return PlaceMatch(remembered.observation_id, yaw, bearing, inliers, len(good))


class PlaceRecognizer:
    """Descriptor voting across every remembered keyframe."""

    def __init__(self) -> None:
        self.keyframes: dict[int, Keyframe] = {}
        self._chunks: list[tuple[np.ndarray, np.ndarray]] | None = None

    def add(self, keyframe: Keyframe) -> None:
        self.keyframes[keyframe.observation_id] = keyframe
        self._chunks = None

    def candidates(
        self,
        query: Keyframe,
        allowed: Callable[[Keyframe], bool],
        limit: int = RETRIEVAL_CANDIDATES,
    ) -> list[tuple[Keyframe, int]]:
        if self._chunks is None:
            self._build()
        votes: dict[int, int] = {}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for descriptors, owners in self._chunks or []:
            for neighbours in matcher.knnMatch(query.descriptors, descriptors, k=3):
                voted = {int(owners[match.trainIdx]) for match in neighbours if match.distance <= RETRIEVAL_MAX_DISTANCE}
                for owner in voted:
                    votes[owner] = votes.get(owner, 0) + 1
        ranked = sorted(
            (
                (self.keyframes[owner], count)
                for owner, count in votes.items()
                if count >= RETRIEVAL_MIN_VOTES and owner != query.observation_id and allowed(self.keyframes[owner])
            ),
            key=lambda item: -item[1],
        )
        return ranked[:limit]

    def _build(self) -> None:
        chunks: list[tuple[np.ndarray, np.ndarray]] = []
        rows: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        count = 0
        for keyframe in self.keyframes.values():
            if count + len(keyframe.descriptors) > RETRIEVAL_CHUNK_ROWS and rows:
                chunks.append((np.vstack(rows), np.concatenate(owners)))
                rows, owners, count = [], [], 0
            rows.append(keyframe.descriptors)
            owners.append(np.full(len(keyframe.descriptors), keyframe.observation_id))
            count += len(keyframe.descriptors)
        if rows:
            chunks.append((np.vstack(rows), np.concatenate(owners)))
        self._chunks = chunks


@dataclass
class GraphEdge:
    source: int
    target: int
    delta: Pose
    information: np.ndarray
    kind: str
    link_id: int | None = None


def information_from_sigma(sigma: Sequence[float]) -> np.ndarray:
    """Information for (x_cm, y_cm, heading_deg) standard deviations."""
    sigma_x, sigma_y, sigma_heading = (max(float(value), 1e-3) for value in sigma)
    return np.diag([sigma_x**-2, sigma_y**-2, math.radians(sigma_heading) ** -2])


class PoseGraph:
    """A 2-D pose graph: stops as nodes, motions and recognized places as edges."""

    def __init__(self) -> None:
        self.nodes: dict[int, Pose] = {}
        self.runs: dict[int, str] = {}
        self.edges: list[GraphEdge] = []

    def add_node(self, node: int, pose: Sequence[float], run_id: str) -> None:
        self.nodes[node] = (float(pose[0]), float(pose[1]), float(pose[2]))
        self.runs[node] = run_id

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def remove_edge(self, edge: GraphEdge) -> None:
        self.edges.remove(edge)

    def component(self, node: int) -> set[int]:
        neighbours: dict[int, list[int]] = {}
        for edge in self.edges:
            neighbours.setdefault(edge.source, []).append(edge.target)
            neighbours.setdefault(edge.target, []).append(edge.source)
        seen = {node}
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for other in neighbours.get(current, []):
                if other not in seen and other in self.nodes:
                    seen.add(other)
                    frontier.append(other)
        return seen

    def components(self) -> list[set[int]]:
        remaining = set(self.nodes)
        found = []
        while remaining:
            group = self.component(min(remaining))
            found.append(group)
            remaining -= group
        return found

    def edge_error(self, edge: GraphEdge) -> np.ndarray:
        measured = relative(self.nodes[edge.source], self.nodes[edge.target])
        return np.array(
            [
                measured[0] - edge.delta[0],
                measured[1] - edge.delta[1],
                math.radians(wrap_degrees(measured[2] - edge.delta[2])),
            ]
        )

    def edge_chi2(self, edge: GraphEdge) -> float:
        error = self.edge_error(edge)
        return float(error @ edge.information @ error)

    def optimize(self, component: set[int], iterations: int = 15) -> None:
        """Gauss-Newton over one connected component; its oldest node stays fixed."""
        ordered = sorted(component)
        if len(ordered) < 2:
            return
        index = {node: position for position, node in enumerate(ordered)}
        edges = [edge for edge in self.edges if edge.source in index and edge.target in index]
        state = np.array([[x, y, math.radians(heading)] for x, y, heading in (self.nodes[node] for node in ordered)])
        size = 3 * len(ordered)
        for _ in range(iterations):
            hessian = np.zeros((size, size))
            gradient = np.zeros(size)
            for edge in edges:
                i, j = index[edge.source], index[edge.target]
                xi, xj = state[i], state[j]
                cosine, sine = math.cos(xi[2]), math.sin(xi[2])
                rotation_t = np.array([[cosine, sine], [-sine, cosine]])
                derivative_t = np.array([[-sine, cosine], [-cosine, -sine]])
                difference = xj[:2] - xi[:2]
                error = np.concatenate(
                    [
                        rotation_t @ difference - np.array(edge.delta[:2]),
                        [math.atan2(math.sin(xj[2] - xi[2] - math.radians(edge.delta[2])), math.cos(xj[2] - xi[2] - math.radians(edge.delta[2])))],
                    ]
                )
                source_jacobian = np.zeros((3, 3))
                source_jacobian[:2, :2] = -rotation_t
                source_jacobian[:2, 2] = derivative_t @ difference
                source_jacobian[2, 2] = -1.0
                target_jacobian = np.zeros((3, 3))
                target_jacobian[:2, :2] = rotation_t
                target_jacobian[2, 2] = 1.0
                information = edge.information
                if edge.kind == "loop":
                    # Huber weighting keeps one bad place match from tearing the graph.
                    chi = math.sqrt(max(float(error @ information @ error), 0.0))
                    if chi > 3.0:
                        information = information * (3.0 / chi)
                blocks = ((i, source_jacobian), (j, target_jacobian))
                for row, row_jacobian in blocks:
                    gradient[3 * row : 3 * row + 3] += row_jacobian.T @ information @ error
                    for column, column_jacobian in blocks:
                        hessian[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += (
                            row_jacobian.T @ information @ column_jacobian
                        )
            hessian[0:3, 0:3] += np.eye(3) * 1e9
            step = -np.linalg.solve(hessian + np.eye(size) * 1e-9, gradient).reshape(-1, 3)
            state += step
            state[:, 2] = np.arctan2(np.sin(state[:, 2]), np.cos(state[:, 2]))
            if np.abs(step[:, :2]).max() < 0.01 and np.abs(step[:, 2]).max() < 1e-5:
                break
        for node, (x, y, heading) in zip(ordered, state):
            self.nodes[node] = (float(x), float(y), math.degrees(float(heading)))

    def reject_inconsistent_loops(self, component: set[int]) -> list[GraphEdge]:
        """Drop the worst loop edge while any is statistically implausible."""
        rejected: list[GraphEdge] = []
        while True:
            loops = [
                edge for edge in self.edges if edge.kind == "loop" and edge.source in component and edge.target in component
            ]
            worst = max(loops, key=self.edge_chi2, default=None)
            if worst is None or self.edge_chi2(worst) <= LOOP_REJECT_CHI2:
                return rejected
            self.remove_edge(worst)
            rejected.append(worst)
            component = self.component(min(component))
            self.optimize(component)


@dataclass(frozen=True)
class PlaceEvent:
    kind: str  # "loop" within a session, "relocalized" against an earlier one
    observation_id: int
    keyframe_id: int
    keyframe_run_id: str
    correction: Pose
    inliers: int
    metric: bool
    link_id: int | None
    sigma: tuple[float, float, float]

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "observation_id": self.observation_id,
            "place_id": self.keyframe_id,
            "place_run_id": self.keyframe_run_id,
            "correction": [round(value, 2) for value in self.correction],
            "inliers": self.inliers,
            "sonar_aligned": self.metric,
        }


class PlaceStore:
    """Persistence the place memory needs; implemented by MappingStore."""

    def place_observations(self) -> list[dict[str, object]]: ...

    def keyframes(self) -> list[Keyframe]: ...

    def save_keyframe(self, keyframe: Keyframe) -> None: ...

    def links(self) -> list[dict[str, object]]: ...

    def save_link(self, source: int, target: int, delta: Pose, sigma: Sequence[float], kind: str, inliers: int) -> int: ...

    def delete_link(self, link_id: int) -> None: ...

    def image(self, observation_id: int) -> bytes | None: ...


@dataclass
class _Stop:
    run_id: str
    sequence: int
    scan: list[dict[str, object]]
    pose_sigma: tuple[float, float]


class PlaceMemory:
    """Keyframes, recognized places and the pose graph, shared across sessions."""

    def __init__(self, store: PlaceStore) -> None:
        self._store = store
        self.recognizer = PlaceRecognizer()
        self.graph = PoseGraph()
        self._stops: dict[int, _Stop] = {}
        self._run_lengths: dict[str, int] = {}
        self._last_closure: dict[str, int] = {}
        self.last_event: PlaceEvent | None = None

    def load(self, keyframe_limit: int = 800) -> None:
        """Rebuild memory from the database, adding keyframes older stops lack."""
        self.recognizer = PlaceRecognizer()
        self.graph = PoseGraph()
        self._stops = {}
        self._run_lengths = {}
        previous_by_run: dict[str, tuple[int, Pose]] = {}
        for observation in self._store.place_observations():
            if observation["pose_x_cm"] is None or not observation["run_id"]:
                continue
            node = int(observation["id"])
            run_id = str(observation["run_id"])
            pose = (float(observation["pose_x_cm"]), float(observation["pose_y_cm"]), float(observation["heading_deg"]))
            estimate = observation["estimate"] or {}
            sigma = estimate.get("pose_sigma", (0.0, 0.0))
            self._add_stop(node, run_id, pose, observation["scan"], (float(sigma[0]), float(sigma[1])))
            edge = estimate.get("edge")
            previous = previous_by_run.get(run_id)
            if edge and int(edge["from"]) in self.graph.nodes:
                self.graph.add_edge(
                    GraphEdge(int(edge["from"]), node, tuple(edge["delta"]), information_from_sigma(edge["sigma"]), "odometry")
                )
            elif previous is not None:
                # Older stops stored only poses; consecutive ones share a frame.
                self.graph.add_edge(
                    GraphEdge(previous[0], node, relative(previous[1], pose), information_from_sigma(DEFAULT_EDGE_SIGMA), "odometry")
                )
            previous_by_run[run_id] = (node, pose)

        stored = {keyframe.observation_id: keyframe for keyframe in self._store.keyframes()}
        for node in sorted(self._stops, reverse=True)[:keyframe_limit]:
            keyframe = stored.get(node)
            if keyframe is None:
                image = self._store.image(node)
                keyframe = make_keyframe(node, self._stops[node].run_id, image, 0.0) if image else None
                if keyframe is not None:
                    self._store.save_keyframe(keyframe)
            if keyframe is not None:
                self.recognizer.add(keyframe)

        for link in self._store.links():
            if int(link["source"]) in self.graph.nodes and int(link["target"]) in self.graph.nodes:
                self.graph.add_edge(
                    GraphEdge(
                        int(link["source"]),
                        int(link["target"]),
                        tuple(link["delta"]),
                        information_from_sigma(link["sigma"]),
                        "loop",
                        int(link["id"]),
                    )
                )
        for component in self.graph.components():
            self._settle(component)

    def _add_stop(self, node: int, run_id: str, pose: Pose, scan: list[dict[str, object]], pose_sigma: tuple[float, float]) -> None:
        sequence = self._run_lengths.get(run_id, 0)
        self._run_lengths[run_id] = sequence + 1
        self._stops[node] = _Stop(run_id, sequence, scan, pose_sigma)
        self.graph.add_node(node, pose, run_id)

    def add_stop(
        self,
        node: int,
        run_id: str,
        pose: Pose,
        scan: list[dict[str, object]],
        pose_sigma: tuple[float, float],
        keyframe: Keyframe | None,
        edge: tuple[int, Pose, Sequence[float]] | None,
    ) -> PlaceEvent | None:
        """Remember a stop, then try to recognize where it is."""
        self._add_stop(node, run_id, pose, scan, pose_sigma)
        if edge is not None and edge[0] in self.graph.nodes:
            self.graph.add_edge(GraphEdge(edge[0], node, edge[1], information_from_sigma(edge[2]), "odometry"))
        self.last_event = None
        if keyframe is None:
            return None
        sequence = self._stops[node].sequence
        cooling = sequence - self._last_closure.get(run_id, -10_000) <= CLOSURE_COOLDOWN_STOPS
        try:
            event = None if cooling else self._recognize(node, keyframe, pose_sigma[0])
        finally:
            self.recognizer.add(keyframe)
        if event is not None:
            self._last_closure[run_id] = sequence
        self.last_event = event
        return event

    def _recognize(self, node: int, keyframe: Keyframe, position_sigma: float) -> PlaceEvent | None:
        stop = self._stops[node]
        component = self.graph.component(node)
        pose = self.graph.nodes[node]

        def allowed(candidate: Keyframe) -> bool:
            other = self._stops.get(candidate.observation_id)
            if other is None or candidate.observation_id not in self.graph.nodes:
                return False
            if other.run_id == stop.run_id and stop.sequence - other.sequence <= RECENT_STOPS_EXCLUDED:
                return False
            if candidate.observation_id in component:
                # Within one frame, a remembered place must be plausibly near.
                place = self.graph.nodes[candidate.observation_id]
                return math.hypot(place[0] - pose[0], place[1] - pose[1]) <= LOOP_SEARCH_RADIUS_CM + 3 * position_sigma
            return True

        attempts = 0
        for candidate, _ in self.recognizer.candidates(keyframe, allowed):
            if attempts >= METRIC_ATTEMPTS_PER_STOP:
                break
            match = verify_place(candidate, keyframe)
            if match is None:
                continue
            attempts += 1
            same_frame = candidate.observation_id in component
            measured = self._metric_relative(candidate.observation_id, stop.scan, match)
            if measured is None:
                if not same_frame or match.inliers < CAMERA_ONLY_MIN_INLIERS:
                    continue
                # Strongly matching pictures from nearby in the same frame: trust
                # the turn, and hold the position loosely to the remembered spot.
                delta = (0.0, 0.0, match.yaw_deg)
                sigma = (CAMERA_ONLY_POSITION_SIGMA_CM, CAMERA_ONLY_POSITION_SIGMA_CM, CAMERA_ONLY_YAW_SIGMA_DEGREES)
            else:
                delta, sigma = measured
            event = self._close(node, candidate.observation_id, delta, sigma, match.inliers, measured is not None)
            if event is not None:
                return event
        return None

    def _metric_relative(
        self,
        place: int,
        scan: list[dict[str, object]],
        match: PlaceMatch,
    ) -> tuple[Pose, tuple[float, float, float]] | None:
        """Register the new sweep against the remembered place's local map."""
        anchor = self.graph.nodes[place]
        place_stop = self._stops[place]
        local = SurfaceMap()
        for neighbour, stop in sorted(self._stops.items()):
            if stop.run_id == place_stop.run_id and abs(stop.sequence - place_stop.sequence) <= LOCAL_MAP_NEIGHBOURS:
                local.integrate(stop.scan, relative(anchor, self.graph.nodes[neighbour]), *stop.pose_sigma)
        echoes = [echo for run in extract_echoes(scan, local.beam_width_deg) for echo in run]
        if not echoes or not local.walls():
            return None
        offsets = {(0.0, 0.0)}
        if match.bearing_deg is not None:
            radians = math.radians(match.bearing_deg)
            offsets |= {
                (round(-math.sin(radians) * distance, 3), round(math.cos(radians) * distance, 3))
                for distance in METRIC_BEARING_DISTANCES_CM
            }
        for step in range(8):
            radians = math.radians(45.0 * step)
            offsets |= {
                (round(-math.sin(radians) * distance, 3), round(math.cos(radians) * distance, 3))
                for distance in METRIC_GRID_DISTANCES_CM
            }
        sigma_x, sigma_y, sigma_heading = METRIC_PRIOR_SIGMA
        prior = np.diag([sigma_x**2, sigma_y**2, math.radians(sigma_heading) ** 2])
        solutions = []
        for x, y in sorted(offsets):
            for yaw_offset in METRIC_YAW_OFFSETS_DEGREES:
                result = register_scan(echoes, (x, y, match.yaw_deg + yaw_offset), prior, local)
                if result.accepted and result.matches >= METRIC_MIN_MATCHES:
                    solutions.append(result)
        if not solutions:
            return None
        best = min(solutions, key=lambda result: (result.rms_cm - 0.05 * result.matches))
        for other in solutions:
            separation = math.hypot(other.pose[0] - best.pose[0], other.pose[1] - best.pose[1])
            if separation > METRIC_AMBIGUITY_CM and other.rms_cm <= best.rms_cm * 1.15 and other.matches >= best.matches * 0.85:
                logger.info("Place %s fits two offsets %.0f cm apart; ignoring it", place, separation)
                return None
        covariance = best.covariance
        sigma = (
            math.sqrt(covariance[0, 0]) + 2.0,
            math.sqrt(covariance[1, 1]) + 2.0,
            math.degrees(math.sqrt(covariance[2, 2])) + 1.0,
        )
        # The registered pose is the new stop in the remembered stop's frame.
        return best.pose, sigma

    def _close(
        self,
        node: int,
        place: int,
        delta: Pose,
        sigma: tuple[float, float, float],
        inliers: int,
        metric: bool,
    ) -> PlaceEvent | None:
        before = self.graph.nodes[node]
        same_frame = place in self.graph.component(node)
        moved: dict[int, Pose] = {}
        if not same_frame:
            # Joining an earlier session: re-express this run in that frame.
            moved = self._move_component_to(node, place, delta)
        edge = GraphEdge(place, node, tuple(float(value) for value in delta), information_from_sigma(sigma), "loop")
        self.graph.add_edge(edge)
        rejected = self._settle(self.graph.component(node))
        if edge in rejected:
            # The match contradicted everything else; the run keeps its frame.
            self.graph.nodes.update(moved)
            return None
        edge.link_id = self._store.save_link(place, node, edge.delta, sigma, "loop" if same_frame else "relocalized", inliers)
        after = self.graph.nodes[node]
        return PlaceEvent(
            "loop" if same_frame else "relocalized",
            node,
            place,
            self._stops[place].run_id,
            (after[0] - before[0], after[1] - before[1], wrap_degrees(after[2] - before[2])),
            inliers,
            metric,
            edge.link_id,
            sigma,
        )

    def _move_component_to(self, node: int, place: int, delta: Pose) -> dict[int, Pose]:
        """Rigidly move the node's component so the node sits where the place says.

        Returns the original poses so the move can be undone."""
        target = compose(self.graph.nodes[place], delta)
        current = self.graph.nodes[node]
        rotation = wrap_degrees(target[2] - current[2])
        transform = pose_rotation(rotation)
        original: dict[int, Pose] = {}
        for other in self.graph.component(node):
            pose = self.graph.nodes[other]
            original[other] = pose
            offset = transform @ np.array([pose[0] - current[0], pose[1] - current[1]])
            self.graph.nodes[other] = (
                float(target[0] + offset[0]),
                float(target[1] + offset[1]),
                wrap_degrees(pose[2] + rotation),
            )
        return original

    def _settle(self, component: set[int]) -> list[GraphEdge]:
        self.graph.optimize(component)
        rejected = self.graph.reject_inconsistent_loops(component)
        for edge in rejected:
            if edge.link_id is not None:
                self._store.delete_link(edge.link_id)
        return rejected

    def component_poses(self, node: int) -> dict[int, Pose]:
        return {other: self.graph.nodes[other] for other in self.graph.component(node)}

    def stop(self, node: int) -> _Stop:
        return self._stops[node]

    def loop_links(self, nodes: set[int]) -> list[tuple[int, int, str]]:
        return [
            (edge.source, edge.target, "loop")
            for edge in self.graph.edges
            if edge.kind == "loop" and edge.source in nodes and edge.target in nodes
        ]

    def counts(self) -> dict[str, int]:
        return {
            "places": len(self.recognizer.keyframes),
            "links": sum(1 for edge in self.graph.edges if edge.kind == "loop"),
            "sessions": len(self._run_lengths),
        }

