# Surface mapping design

RoboCar turns its swept ultrasonic ranges into a map of **surfaces** — wall
segments joined into contours, plus point features such as corners, edges and
furniture legs — and refines that map every time it stops and sweeps again.
Wheel odometry, the camera and the sonar each estimate how the car moved;
the estimates are fused so that each one covers the others' blind spots.

Code: `src/robocar/surfaces.py` (geometry and map),
`src/robocar/localization.py` (motion fusion and scan registration),
`src/robocar/mapping.py` (the per-stop pipeline), and odometry in
`src/robocar/drive.py`.

## Conventions

- Robot frame: +y forward, +x right. The sonar sits `SONAR_FORWARD_OFFSET_CM`
  ahead of the turning centre on the pan axis.
- Headings and sweep angles are degrees **counterclockwise** (positive = left).
  This matches the Freenove motor mix (positive steering turns left), the
  servo (`angle_deg = pan − 90`, so pan 150 looks left) and the camera
  yaw estimate. A direction is `(−sin θ, cos θ)`.
- A run's world frame is the car's pose at its first stop, `(0, 0, 0)`.
- The old dashboard drew `x = sin θ`, which mirrored the map left/right. The
  new renderer uses the convention above.

## What happens at each stop

1. **Camera views.** Three frames at the current pan/tilt. Their texture is
   scored (ORB keypoints × image coverage).
2. **Odometry.** `DriveController` integrates the actual ramped wheel duties
   through the inverse mecanum mix, at 100 Hz, into `(x, y, heading)`. The
   difference since the last stop is the motion guess, scaled by a learned
   per-run correction.
3. **Decide whether side pictures are needed.** If the forward view is bland,
   failed to measure motion, or disagrees with the wheels (or the previous
   stop needed side views), the sweep also grabs frames at pan 54° and 126°.
   The head is already passing through those angles, so this costs only the
   snapshots.
4. **Sweep.** 40 pan positions, 3° apart (−57° to +60°), three pings per
   angle, median kept. The head first parks at pan 30 and then only steps
   upward, so every reading (and side picture) is reached from the same side
   and servo backlash cannot shift one reading against another. Adjacent beams
   overlap heavily, so surfaces show up as continuous range profiles.
5. **Motion fusion** (`predict_motion`):
   - Odometry gets an uncertainty that grows with distance and turn (mecanum
     wheels slip, especially when turning).
   - Camera yaw from the essential matrix is precise; the camera's travel
     direction (unscaled) corrects sideways slip.
   - Gate: if camera and wheels disagree by more than 3σ, several agreeing
     camera estimates overrule the wheels (slip, or the car was moved by hand);
     a lone disagreeing view is discarded.
   - The result is propagated into the pose and its 3×3 covariance.
6. **Sonar registration** (`register_scan`): the new echoes are fitted to the
   walls (point-to-line) and features (point-to-point) already in the map with
   a regularized, Huber-weighted Gauss–Newton solve. First the echoes are
   solved almost on their own and compared with the prediction using both
   covariances (a χ² innovation test at 99%). Only if they agree is the final
   pose fused. Directions the surfaces cannot see (along a single wall) stay
   with the prediction and keep their uncertainty.
7. **Learn the odometry scale.** When registration pins the distance travelled
   (or the camera pins the turn), the per-run translation/rotation scale is
   nudged toward the measured ratio, clamped to 0.5–2.0.
8. **Integrate the sweep into the map** (below), and save the observation with
   its pose and a JSON record of every estimate (`estimate_json`).

## From echoes to surfaces

**Beam correction.** The sensor reports the nearest reflector anywhere in its
cone. The datasheet says 15°, but on real walls it is wider; your first real
run measured about 28°. Each map starts at 25° and re-measures it (below).

- *Regions of constant depth (RCDs).* Adjacent readings within ~1% of a local
  minimum range face the sensor. They are grown outward from the nearest
  reading, and a group with a nearer neighbour is only a gentle slope, so it
  is split back up. Each reading in an RCD is laid on the surface: readings
  within half a beam of the centre mark the normal foot, and readings beyond
  that are moved half a beam toward the centre.
- *Cut-off RCDs.* When an RCD runs off the edge of the sweep, or into
  something nearer, its middle is not the normal. The normal lies at least
  half a beam beyond the edge that was seen.
- *Oblique readings.* Along a wall, the range rises away from the normal, and
  the echo comes from the cone edge nearest the normal. Shifting those
  bearings half a beam toward falling range puts the points back on the wall
  instead of on a bowed arc (≈9 cm bow → <1 cm in tests).
- *Occluding edges.* Where the range jumps farther (an opening, a doorway, the
  end of a box), the near surface keeps answering until its edge leaves the
  cone. So the edge lies half a beam back from the jump, and echoes beyond it
  are pulled back onto the edge instead of bleeding into the opening.
- *Range peaks* between nearer readings, of any width, are concave corners
  inside the beam and are dropped. *Range jumps* split the sweep into
  separate runs (different objects).
- Each echo carries an uncertainty from range noise and bearing uncertainty,
  plus the width of surface it vouches for (range × sweep step). Pose
  uncertainty is added when the echo is placed in the world.

**Beam width self-calibration.** A flat wall reads its perpendicular distance
across a flat top exactly one beam wide, then d / cos(angle past the beam
edge). Fitting that profile to every run that shows the flat top *and* rising
flanks on both sides measures the beam. With only one flank, beam width and
wall direction can't be separated, so those runs are skipped. After six good
fits, the map uses their median (clamped to 10–40°).

**Surfaces as fine segments.** Each run is split into straight pieces
(recursive split, then merge), and pieces with ≥3 echoes become wall
candidates; the rest are point echoes. A candidate fuses with an existing wall
when it is parallel (≤15°), close (offset tolerance grows with range and pose
uncertainty) and within a doorway's width (60 cm) along it. Walls keep their
supporting points and are refitted by weighted total least squares with
outlier rejection, so every sweep refines the line.

A wall is only drawn, matched and joined where its points are continuous. It
breaks into separate segments wherever neighbouring points are farther apart
than max(10 cm, 1.5 × their combined footprints). A doorway in a wall is
therefore one line model with two segments and a gap. A lone point beyond a
gap steadies the line but isn't drawn. Single new points join a wall only right
next to an existing segment, so a wall can't creep across an opening sweep by
sweep. Afterwards the map consolidates:

- collinear walls up to a doorway apart share one line (their gaps remain);
- features lying on a wall are absorbed into it;
- nearby features merge;
- three or more collinear features are promoted to a wall.

**Contours.** Segment ends of different walls are joined at corners, using the
walls' intersection when they meet at an angle. Smooth walls stop echoing near
a corner (specular reflection), so a hidden corner up to 45 cm beyond both
ends is filled in. But the map remembers the centre lines of recent beams (the
space sound actually crossed), and no join is made across a beam that passed
through. That keeps passages at corners open. Two segments share at most one
corner, and confirmed walls claim corners before fragments seen once.

**Negative evidence.** A beam that travelled well past a stored wall it hit
within 30° of head-on removes the wall points inside its core after two such
misses. This opens gaps where openings are confirmed later. It also counts a
miss against the wall, or against a feature inside the core. Surfaces with at
least two misses and as many misses as sightings are retired, so moved boxes,
opened doors and people fade out.

**Confidence.** A surface is *confirmed* after two sweeps with
confidence ≥ 0.5 (`sightings / (sightings + 2·misses + 1)`). Only confirmed
surfaces (or walls with ≥5 points) anchor registration.

## Remembering places

Code: `src/robocar/places.py`, wired into each stop by `mapping.py`.

**Exposure.** Mapping snapshots used to open the camera, take its first frame
and close it again. That frame was taken before auto-exposure had adjusted,
which is why the real pictures were so dark. The camera now stays open for
the session and discards frames after opening or changing settings. Each
stop's picture runs through an exposure ladder (see the README) until its
levels and detail are decent, keeping the best picture and remembering the
setting for the next stop. Camera motion estimates also need real support
now (25 matches, 20 inliers, 40% inlier fraction, turns under 55°). A turn
measurement needs two agreeing strip pictures, and the camera can't overrule
the wheels by more than 30° on one move.

**Dim pictures.** Your first real run's pictures averaged 32–44 out of 255
in brightness with very low contrast, and most yielded fewer than 40
features. When a normal detection finds fewer than 300 corners, the picture
is lightly blurred, contrast-equalized locally (CLAHE) and searched with a
more sensitive detector. On those real pictures this roughly tripled good
matches between stops. Well-exposed pictures are left alone, because the
same treatment adds noisy corners to them. This applies to keyframes and to
all camera motion estimates. Each stop also waits 0.3 s before its first
picture so the car and head have stopped moving (one real picture was
smeared).

**Keyframes.** Every stop keeps the 400 strongest ORB features of its forward
picture (with the pan angle it was taken at), alongside its sonar sweep and
pose. They live in the `keyframes` table and are shared by all sessions.
Older stops without keyframes get them from their saved pictures at start-up.

**Recognition at each stop:**

1. *Retrieval.* The new picture's descriptors vote across every stored
   keyframe (brute-force Hamming matching in chunks, about 0.4 s for 400
   places on the Pi 5). The best-voted places become candidates, excluding
   the last four stops of the same run. Within one frame, a candidate must
   also lie within 1.5 m (plus 3σ of pose uncertainty) of where the car
   thinks it is.
2. *Verification.* Each candidate is matched feature by feature and must fit
   a camera motion with at least 30 RANSAC inliers. A pure turn is detected
   when K⁻¹HK is a rotation (equal singular values). Views dominated by one
   wall use homography decomposition, keeping the solution with points in
   front and a vertical plane; otherwise the essential matrix is used. In
   simulation this identifies places reliably but measures the turn between
   views only to about ±15°, so the camera's turn is a starting point.
3. *Metric alignment.* The new sweep is registered against a small surface
   map rebuilt from the remembered place and its neighbouring stops (±3).
   The fit starts from a spread of offsets (along the camera's direction,
   plus a ring at 35 and 70 cm) and turns (±10° around the camera's
   estimate). If two different offsets fit equally well, as in a long
   corridor, the match is ignored.
4. *Heading links from every relevant picture.* Every stop is compared not
   only with the best-voted places but also with nearby earlier stops (within
   1.2 m) that faced a similar way (within 50°), up to eight verified per
   stop. Only the immediately previous stop is skipped, since stop-to-stop
   matching already covers it. A verified match without a sonar fit becomes
   a heading-only link: tight (2.5°) for a pure turn, which the camera
   measures well, loose (8°) otherwise. So the heading is always pinned to
   everything the camera has seen before.

**Pose graph.** Nodes are stops. Edges are the motion between consecutive
stops (from the fused estimate, with its uncertainty; recorded as
`estimate_json.edge`) and recognized places (`place_links` table).
Gauss–Newton over SE(2) runs on the connected component, with its oldest stop
fixed and Huber weighting on place links. After optimizing, any place link
with χ² > 16.3 (99.9% for 3 dof) is dropped, worst first, and the graph is
re-solved, so a false match cannot tear the map apart. The graph's pose for
the current stop is used every time, and the map is rebuilt whenever any stop
moves by more than 4 cm or 2°. At most two candidates per stop get the
costly sonar alignment, with a two-stop pause after a confirmed closure.

**Sessions.** A new run starts in its own frame. When it recognizes a place
from an earlier run, the run is moved rigidly into that run's frame and the
graphs join. From then on its poses, the map (rebuilt from every joined
session's sweeps at their optimized poses) and exploration's preference for
unvisited ground all use the shared frame. If the joining link is later
rejected, the move is undone. On start-up, the whole graph is rebuilt from the
database and re-optimized, so the latest session reappears joined to
everything it recognized.

**Results in simulation** (3D-rendered textured room with parallax, plus the
sonar room):

- A second session starting 38 cm and 25° away from the first one's start
  recognized it at its first stop. It placed itself within 4.4 cm and 1.7°,
  then within 1.6 cm and 1.1° after its next sweep.
- With per-stop sonar fits and camera motion disabled, and odometry 12% long
  on distance and 10% long on turns, a rectangle round the room drifted to
  27 cm and 27°. Returning to the start closed the loop, leaving the pose
  within 2.8 cm and 0.6°.
- A room with different wall textures was not mistaken for the remembered
  one.

## Exploring

Code: `src/robocar/exploration.py`; the planner is `plan_explore_step` in
`mapping.py`; the benchmark floor plan is `tests/explore_sim.py`.

- **Coverage grid.** A 5 cm log-odds grid built from every sweep (joined
  sessions too, at their optimized poses). Each reading clears space along
  three rays inside the beam's core, and only the centre ray marks an echo.
  A missing echo (past 250 cm; the sensor reports 300) clears only 50 cm.
- **Walls the way the beam sees them.** A run of echoes at similar ranges is
  fitted as a flat wall at distance d with normal n, which reads
  d / cos(max(0, |a − n| − 12.5°)) at sweep angle a. Fitting the echoes'
  centre lines instead tilted corridor walls by about 13°. Each fitted wall
  is continued 150 cm past its ends.
- **Glancing versus head-on.** Sound meeting a wall more than 40° from
  head-on bounces away silently. A ray that would cross a fitted wall, or a
  mapped wall whose direction is clear from its neighbouring cells, at such
  an angle stops there: the corridor side walls. One crossing nearly head-on
  would have echoed, so that wall cell is wrong, for example an echo from a
  doorway's edge drawn across the gap. It loses an extra 1.2 in log-odds.
  An earlier rule that treated a wide stretch of equal readings as a doorway
  was dropped: near a flat wall the readings creep up so slowly that the
  stretch spans 75°, and the rule erased the wall. A doorway narrower than
  the beam is indistinguishable from a wall in one sweep.
- **Leads.** A reading at least 25 cm deeper than readings on both sides
  within 18° is part of an opening (the beam smears its edges, so they deepen
  gradually). Runs of 6° or more become leads: the mouth, at the sides'
  distance, and the direction into it. They come from the raw readings, so
  ghost walls on the grid cannot hide them.
- **Frontiers.** Free cells next to unknown ones, excluding blind spots
  within 35 cm of anywhere the car has stood, judged in 50 cm pieces. A
  frontier along a wall can include a doorway that only shows from beside it.
  The car stands at the drivable spot nearest each piece's middle, within
  30 cm of it. Cells just behind echoes that stay unknown after three looks
  are behind walls and stop counting.
- **Information gain.** Rays every 3° across the 144° a sweep sees, out to
  250 cm, from the standing spot facing the frontier. Each unknown sample
  counts for the arc of floor it stands for. Confirmed walls (log-odds above
  1.5, with gaps under 30 cm closed, since corners rarely echo) stop a ray;
  each cell of wall seen once lets 60% through.
- **Choosing.** Value = learned weight for the kind × gain ÷ effort. Effort
  is path cost plus 1.2 cm per degree of turning, 60 cm overhead per trip,
  and 120 cm extra for turning straight back. Paths cost up to three times
  as much within 45 cm of a wall. The current target is kept unless another
  is worth 1.5× as much. Weights move 30% toward the floor a pursuit really
  showed ÷ its prediction, only for pursuits that arrived. The rate (cells
  per cm of effort) tracks every move; exploring ends when the best value is
  under 5% of it.
- **Corridors.** Mapped walls within 100 cm forming two straight sides 40–130
  cm apart. Both principal directions are tried, since short stretches of wall
  can spread as much across as along. Targets within 70° of the axis are
  approached by aiming 60 cm ahead on the centre line.
- **Safety.** Forward legs stop before the car comes within its 16 cm radius
  of any mapped wall, confirmed or not. Already inside that margin, it may
  only move where it gets no nearer. Missing echoes from walls met at a
  glancing angle make the sonar unreliable for this. A side reading under
  20 cm means no room to spin: it slides 12 cm toward the roomier side.
- **What did not work** (all in simulation): following one frontier's
  bearing exactly made the car dither when the target was within 30 cm; it
  now faces the frontier itself. A fixed "keep turning the same way" rule
  turned that dithering into full spins. Scoring gain minus a learned
  rate × effort fell apart late in a run, when the rate collapsed and effort
  became free. Letting the car leave the wall margin by any route had it
  stepping deeper into it.

## Recognized objects (`semantics.py`)

- **Detection.** `ObjectDetector` (`detector.py`) runs YOLOv8s on the AI
  HAT+ through HailoRT's `InferVStreams`. The model's output is "NMS by
  class": for each class, rows of [y_min, x_min, y_max, x_max, score] as
  fractions of the 640×640 letterboxed input, mapped back to frame pixels.
  Without the HAT it falls back to YOLO11n on the CPU. A failure on the HAT
  hands over to the CPU and the HAT is retried after 30 s; if both fail,
  there are simply no detections.
- **Bearings.** The OV5647 in the 640×480 mode reads 2560 of 2592 sensor
  columns: about 53° across. Column u lies atan((u − 320)/f) to the right of
  the axis, with f = 320/tan(26.5°); the head's pan bearing is added on top.
- **Ranges.** The nearest sonar reading within the box's angular width plus
  4° each side. None when the sweep saw nothing there, for example an object
  below or above the beam.
- **Moving things.** Readings toward a person or animal, out to 40 cm past
  the nearest one and within its width plus 6°, are flagged `dynamic` in the
  stored scan. `static_scan` blanks them for every mapping consumer; the
  planner's clearances still use the raw scan.
- **Landmarks.** A ranged sighting is a point with σ = 8 cm + 8% of range.
  It joins the nearest landmark of the same label within max(60 cm, 3σ), as
  a weighted mean. A bearing-only sighting confirms a landmark within 8° of
  its ray. Otherwise it is paired with an earlier ray from at least 30 cm
  away, at least 10° apart, crossing in front of both within 4 m, and the
  crossing becomes a point with σ = 25 cm.
- **Rooms.** Single-linkage groups of confirmed landmarks with room hints,
  links up to 250 cm that cross no confirmed grid wall. The room hint
  scores, weighted by each landmark's certainty, name the group when the
  best reaches 1.5.

## Motion calibration

Code: `src/robocar/calibration.py`. Turns: a strip of pictures every 5°
across the whole pan range, then each turn size and direction is repeated,
scaling the command by the measured ratio, until it lands within 2°. Then
the camera brings the car back to its starting heading, so the step tests
face the wall squarely. Steps: sonar median of 7 pings before and after each
burst, and the burst length is updated by fitting a line through the last
two tries (travel has an offset from dead zone and ramps). Each new step size
starts from a line through every step measured so far, and creep is driven
back out toward where the steps began. On the real car a 0.1 s burst moved
about 1 cm, so the first version's scale-from-zero guess jumped to 0.77 s and
60 cm; changes are now capped at 1.6× per try. In simulation, with small turns
overshooting 30% and a forward dead zone, every turn landed within 2° on its
second try. A step simulation using the real car's dead zone hit 5, 10 and
20 cm, the latter two on the first try, and ended within 3 cm of its start.

**Pan direction.** The first real calibration could not measure a single
turn. The code had pan 150° looking left, but on the car it looks right. Every
strip picture was therefore mirrored, and the pictures disagreed with each
other by twice their angle. Sweeps were mirrored relative to the car's turns
as well. `pan_bearing()` in `mapping.py` now does the conversion in one place.

## Sensor findings from the real car

- **Ultrasonic smoothing.** The kit's gpiozero `DistanceSensor` reports the
  median of its last 9 background pings (about 0.54 s). The sweep read each
  angle after about 0.2 s, so readings mixed in earlier directions. That
  smeared every profile in the sweep direction and inflated the apparent beam
  width. The sensor is now rebuilt with a 3-ping median, and the sweep waits
  for it to refresh at each angle.
- **Colour order.** Picamera2's `"BGR888"` delivers R, G, B, so every frame had
  red and blue swapped (the wood walls looked blue). The camera now uses
  `"RGB888"`, which is B, G, R as OpenCV expects. This also matters for the
  person/cat detector.
- **Exposure and blur.** Snapshots used to grab the first frame of a freshly
  opened camera (badly underexposed). Long exposures right after a move
  smeared. See the README for the exposure ladder, steadiness check and
  sharpness scoring.

## Persistence and API

- Each observation stores the raw sweep, the refined pose and the estimate
  record. On start-up the newest run's map is rebuilt by replaying its sweeps,
  so nothing derived needs migrating when the algorithms improve.
- `GET /api/mapping/surfaces` returns surfaces, features, contours, recent
  echoes, the path, earlier joined sessions' paths (`history`), recognized
  place links (`links`) and the pose uncertainty. `/api/status` reports which
  sources fixed the latest pose, the place-memory counts (`places`) and the
  latest recognition (`last_place`).
- **Reset mapping database** also clears keyframes and place links.

## Calibrate on the car

Simulation tests validate the geometry. These constants decide how well it
transfers to the real car:

| Constant | File | How to measure |
| --- | --- | --- |
| `SONAR_FORWARD_OFFSET_CM` | surfaces.py | Turning centre to the pan axis. |
| `SONAR_BEAM_WIDTH_DEGREES` | surfaces.py | Starting value only; each map re-measures it from flat walls (`beam_width_deg` in `/api/mapping/surfaces`). |
| `FORWARD_CM_PER_SECOND_AT_FULL_DUTY` | drive.py | Run **Calibrate turns & steps**; it measures the real ratio. |
| `TURN_DEGREES_PER_SECOND_AT_FULL_DUTY` | drive.py | Run **Calibrate turns & steps**; it measures per size and direction. |
| `STRAFE_CM_PER_SECOND_AT_FULL_DUTY`, `MOTOR_DEADBAND_FRACTION` | drive.py | Strafe 1 s; lowest duty that moves the car. |
| `CAMERA_HORIZONTAL_FOV_DEGREES` | mapping.py | IMX477 lens field of view. |

Also check the sign convention once: put a box on the car's left and confirm
the dashboard draws it on the left.

## Known limits and next steps

- A sweep takes about 11–12 s, so even the four-minute explore limit fits
  roughly 15–20 stops.
- An opening narrower than the beam's footprint at that range (about 0.44 ×
  range for a 25° beam) can't be seen through from that spot. Every reading
  hits a jamb, and it looks like a wall until sweeps from elsewhere carve it
  out.
- Close to a wall (≲30 cm) the whole sweep collapses into one RCD. Registration
  declines and the pose relies on wheels and camera for that stop.
- Soft furnishings absorb sound, and very oblique smooth surfaces never echo.
  Expect gaps rather than false walls.
- Place recognition relies on visual texture. Bare walls or very dim light give
  too few features, and the car then falls back to wheels, camera turns and
  sonar alone. Large lighting changes between sessions weaken matching too.
- A session that never sees a remembered place stays in its own frame until
  it does.
- The pose graph is solved densely. That's fine for a few hundred stops (about
  0.3 s for 600), but a very long-lived house map would want a sparse solver or
  pruning of redundant keyframes.
- Next steps: let exploration aim for unexplained contour ends (frontiers),
  and validate recognition on real pictures from the car.
