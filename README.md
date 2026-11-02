# RoboCar

RoboCar is a Raspberry Pi 4WD car application scaffold for a VNC-accessible
camera dashboard, object detection, manual driving, and later autonomous
navigation.

## Quick start

Start RoboCar with:

```bash
./start-robocar.sh
```

The script activates `/home/sondericker/robotics-venv` and launches the app.
Press Ctrl+C in the terminal to stop it.

If the virtual environment does not exist yet, create it and install RoboCar:

```bash
python3 -m venv --system-site-packages /home/sondericker/robotics-venv
/home/sondericker/robotics-venv/bin/pip install -e /home/sondericker/Documents/Python/RoboCarv1
cp .env.example .env
```

Open `http://127.0.0.1:5000` in the Raspberry Pi's desktop browser (including
through VNC). The app binds to loopback by default. Without hardware enabled,
drive commands are simulated and cannot move the car. To stop it, press
Ctrl+C in the terminal. Camera processing starts paused; press **Start camera
processing** when you want the live feed and detections.

The dashboard detects the 80 everyday COCO object classes (people, pets,
furniture, appliances…) and draws labelled boxes on the live feed.

- **On the Raspberry Pi AI HAT+** (Hailo-8 or 8L) it runs YOLOv8s from the
  `hailo-models` package, with non-maximum suppression done on the chip.
  That takes about 12 ms a frame (the chip alone manages 300 frames a second)
  and leaves the CPU free. Install it with `sudo apt install hailo-all`.
- **Otherwise** it runs YOLO11n on the CPU through ONNX Runtime. On first
  launch it downloads the official `yolo11n.onnx` to `~/.cache/robocar/`;
  set `ROBOCAR_MODEL_PATH` to use another YOLO ONNX model.
- `ROBOCAR_DETECTOR` chooses: `auto` (the default) uses the HAT when it is
  there and the CPU model otherwise. If the HAT fails mid-run, the CPU model
  takes over and the HAT is retried every 30 seconds. `hailo` and `cpu` use
  only that one; `off` turns detection off. `ROBOCAR_HAILO_MODEL` picks a
  different `.hef` file.
- HailoRT lets one program use the HAT at a time. While RoboCar runs, other
  Hailo tools (and RoboCar's hardware test) report it busy; a second RoboCar
  started meanwhile uses the CPU.

## Raspberry Pi setup

The confirmed hardware profile is the Freenove FNK0043 Rev. 2/B kit, mecanum
wheels, and Raspberry Pi IMX477 camera. Picamera2 is the selected camera
backend; verify the sensor is detected with `rpicam-hello --list-cameras`.
Install the Pi-specific
system packages there; `requirements-hardware.txt` documents the Python-side
extras. The `picamera2` package is normally supplied by Raspberry Pi OS and
should be installed with its OS package manager, not from pip. If using a
virtual environment with Picamera2, create it with `--system-site-packages`.

Copy `.env.example` to `.env` and set:

```dotenv
ROBOCAR_CAMERA_BACKEND=picamera2
ROBOCAR_HARDWARE_ENABLED=true
FREENOVE_SERVER_PATH=/path/to/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/Code/Server
ROBOCAR_MODEL_PATH=~/.cache/robocar/yolo11n.onnx
ROBOCAR_LABELS_PATH=
```

**Cameras.** The car has two, and RoboCar finds each by its sensor name:
- **Forward (hunting):** a Raspberry Pi HQ camera (IMX477) with a 6 mm
  lens, fixed facing forward on the chassis, in camera port 0. It is the
  dashboard's live view and the hunt's eyes. RoboCar reads the whole sensor
  in its fast binned mode (2028x1520, about 54 frames a second), scaled to
  640x480 for detection: about 55° across and 43° high (`CAMERA_HFOV_DEG` in
  `semantics.py`; change it with the lens). Open the lens's aperture fully
  and focus at about 2 m: at detection's resolution that keeps roughly 1 m
  to infinity sharp.
- **Head (mapping):** the OV5647 on the pan/tilt head, in camera port 1.
  Mapping pans it to measure turns, take side views and make the sweep
  panorama, and turn calibration uses it.

Both are set explicitly in `/boot/firmware/config.txt` (the OV5647 module is
not found by auto-detection):

```
camera_auto_detect=0
dtoverlay=imx477,cam0
dtoverlay=ov5647,cam1
```

`ROBOCAR_FORWARD_CAMERA` and `ROBOCAR_HEAD_CAMERA` name the sensors
(defaults `imx477` and `ov5647`). If only one is connected it does both
jobs. With only the forward camera, the hunt still works, but mapping takes
no picture strips or side views, turns go by the wheel counts, and turn
calibration is unavailable. With neither found by name, `ROBOCAR_CAMERA_INDEX`
picks the camera and `ROBOCAR_CAMERA_ON_HEAD=true` says it is on the head.

With a fixed forward camera, the hunt looks around by spinning the car on the
spot, watching all the way round, instead of panning the head.

With both cameras, mapping uses the forward one too:
- **Places for the hunt:** each stop also keeps a place picture from the
  forward camera, apart from the head camera's (pictures from different
  cameras do not match). While hunting, whenever the car stands still a
  moment, its live picture is matched against them. When two or more places
  agree on its heading and the directions from them toward the car cross
  steeply, that fixes its position and heading. The fix is combined with
  the wheel-based estimate, and dropped if the two disagree too much. It
  counts as matching the map, so "knows where it is" can stay green without
  stopping to sweep.
  - **Accuracy:** in a rendered test room, fixes from three or more places
    were within 10 cm and 3°. About one in eleven from just two places was
    off by around 25 cm and 6°, so those count as less sure.
  - **What is refused:** a single place, or two looking at the same wall
    from about the same way, can agree on a wrong heading. They give no fix.
- **Furniture:** the forward camera's stop picture is the one checked for
  couches, beds and chairs. It is sharper and brighter than the head
  camera's.
- **The cat:** the forward camera's feed is watched while mapping too. Where
  it sees the cat goes into the hunt's memory of its favourite spots.

### Lidar

A Benewake TF-Luna on the pan/tilt head, just above the ultrasonic, gives
the car precise ranges: a 2° beam, readings 20 cm to 8 m, 100 a second. It
sits on the Pi's I²C bus 1 at address `0x10`, shared with the motor board,
with its pin 5 tied to ground (I²C mode). Plug it in or out only with
everything powered off: a sensor connected live once jammed the whole bus
(motors and battery readings with it) until a reboot. `ROBOCAR_LIDAR` is
`auto` (used when it answers, the default), `on` (shown as a fault when
missing) or `off`. `i2cdetect -y 1` should list `10`, `40` and `48`.

What it changes:
- **Sweeps:** a reading every degree from 60° left to 60° right (121 in
  all, each the median of three frames, about 7 s a sweep), against 40
  smeared sonar readings before. The sonar's reading is kept alongside each.
- **The map:** each lidar reading is a point on a surface, so walls, chair
  legs and doorways come out sharp. In the simulated two-room house, lidar
  sweeps found 95% of the floor in 20 moves (the sonar about half), and a
  sweep fitted to the map placed the car within about 2 cm and 1° (the
  sonar about 12 cm and 5°). The dashboard draws the lidar's points, one per
  2.5 cm.
- **Safety:** before driving, the nearer of the lidar and the sonar counts.
  The sonar's wide beam sees glass and thin things beside the line the
  lidar passes, and the lidar sees dark fabric that soaks up sound. The way
  ahead must be clear across the car's width, not just on one line.
- **The cameras:** an object's distance comes from the lidar readings across
  the middle of its box, not the wall behind it. A "Lidar over the picture"
  panel draws each stop's readings over the camera's view, so you can see at
  a glance that they line up.
- **The hunt:** see below.

Keep the car's wheels off the ground while first testing physical controls.
The motor adapter is disabled unless `ROBOCAR_HARDWARE_ENABLED=true`. Dashboard
controls are a convenience, not a safety-rated emergency stop. Do not expose
the unauthenticated control server to an untrusted network; its default
loopback binding is intentional.

If the car's board cannot be reached at startup (switched off, flat
batteries, unplugged), the dashboard and camera still start. The motors,
pan/tilt head or range sensor that could not be reached show as
**unavailable**, with the reason, and refuse commands (stop always works).
Mapping and calibration say why they cannot start. Fix the board and restart
the service.

If the motor board is reported unavailable because the PCA9685 at I²C
address `0x40` did not respond, make sure the car battery is connected so the kit board is powered,
then check its I²C connection. On this kit the motor controller communicates
over I²C (not SPI). With the car powered, run `i2cdetect -y 1`; the board
should appear at address `40`. The YOLO/ONNX schema messages printed during
model loading are separate from this I²C error; the I²C error is what prevents
hardware mode from starting.

The drive pad sends separate forward/reverse, lateral strafe, and rotation
commands for mecanum motion. Start at low duty and verify the direction of each
axis with the wheels raised before driving on the floor. Motor duty ramps up
and down over 0.25 seconds for normal drive commands and stops. The watchdog
and explicit mapping-stop action still cut motor output immediately.

Use **Stop camera processing** on the dashboard to pause frame capture and
object detection. The camera is released while paused, and the video stream
waits without sending frames. Press **Start camera processing** to resume.

Pan/tilt sliders are also on the dashboard. Pan 30° looks to the car's left
and 150° to its right (the Freenove servo turns clockwise as the angle
rises). The head starts at pan 90° and
tilt 90° (level, so the lidar shoots flat), and **Center head** returns it there. The initial software travel
is limited to 30°–150° to avoid the servo end stops; calibrate the safe range
on the actual camera head before widening it.

When physical hardware is enabled, the dashboard also displays the live
lidar range (with its signal strength and temperature) and the ultrasonic
range in centimeters. The sensor samples in the background and
reports missing echoes or read errors instead of presenting an old value as
current. With hardware disabled, the UI identifies the range finder as
disabled; it does not simulate a distance. The reading is informational and
does not currently interlock or stop drive commands.

## Cat hunt

Once the room is mapped, **Start the hunt** in the dashboard's Cat hunt
section sends the car to find the cat and play with it. It is attended,
like auto explore. The section shows:
- the car's camera view, with what it recognizes boxed and what it is doing
  written underneath;
- its phase (seeking, approaching, playing, resting), with the current move
  in plain words;
- the cat's distance, and recent moves;
- the moves the cat has reacted to most.

The map marks where the cat was last seen and its favourite spots.

- **Eyes always open:** the camera feed is watched continuously, every frame
  checked for the cat on the AI HAT+ (about 12 ms a frame). While the car is
  looking for the cat, the moment one appears whatever it is doing stops (a
  turn, a drive, even a sonar sweep half way through) and it goes after the
  cat. In simulation it switched to the chase within a quarter of a second
  of the cat coming into view.
- **Sharp frames while moving:** the hunt runs the camera's exposure
  itself. It keeps the brightness up with exposure time and gain, but caps
  the exposure while the car or head is moving (8 ms to start with, allowing
  up to 33 ms when still), using gain instead: detection copes with grain
  far better than with blur. The moving cap adapts. If frames on the move
  come out clearly softer than still ones it shortens (down to 2 ms), and if
  they are about as sharp it lengthens again (up to 16 ms) for less noise.
  Sharpness is judged relative to contrast, and the cap never shortens while
  the light runs short (gain at its limit): a darker frame also looks
  softer, and chasing that once spiralled down to black frames. In dim light
  the cap grows instead. When the hunt ends the camera goes back to
  auto-exposure.
- **Seeking:** a belief map says where the cat might be. It covers the floor
  the map knows, weighted toward couches, beds and chairs the camera
  recognized and toward spots where the cat was found before.
  - **Looking:** everything the camera sees without a cat lowers it over
    that view, with walls blocking the view.
  - **Cats move:** it drifts back toward the prior as time passes.
  - **Where next:** the car goes where it could see the most of it per
    centimetre of travel.
  - **Getting there:** it drives without stopping, about 35 cm/s, steering
    along the map's route with the sonar watching ahead (it slows as things
    get close and stops 45 cm short). It turns in place first when the way
    is more than 35° off.
  - **On arrival:** the head glances across the room, a fraction of a second
    per direction, and the car moves on.
  - **Sonar sweeps:** only when it no longer knows where it is on the map,
    and at most every 2 minutes. The camera keeps watching through the
    sweep, and its pictures make the panorama.
- **Live picture:** the dashboard shows the feed about six times a second,
  with the frame rate and exposure underneath.
- **Approaching:** it steers straight at the cat while watching it, turning
  in place when the cat is more than 20° off. It slows as it gets close
  (creeping when its mood is mellow), measures the distance, and stops in
  play range (about 1 m). If it
  loses sight of the cat for a moment it holds still and looks; after 2.5 s
  it searches nearby.
- **Measuring the cat:** with the lidar, and the camera fixed facing ahead,
  the head turns the lidar's beam onto the cat wherever it is in view (up to
  55° either side) and measures it, trying a couple of degrees either side
  in case the thin beam slipped past its edge. Between measurements each
  camera frame scales the last one by how much bigger or smaller the cat
  looks, so the distance stays right frame by frame. The dashboard says
  whether the distance shown is measured ("lidar") or guessed from the
  cat's size ("camera"). In simulation the distance was within about 1 cm
  (median). Without a lidar, the sonar measures the cat once it is straight
  ahead.
- **Playing:** nothing is scripted.
  - **Moves:** each beat picks one of 16 moves, each with fresh random
    sizes, speeds and directions: pounce, charge and retreat, zig-zag in or
    out, sidestep, shimmy, wiggle, spin, orbit (circling while facing the
    cat), stalk, freeze, peekaboo, head-bob, back-off tease, combos of
    those, and two that react to the cat as they go:
    - **mouse:** playing prey from the live feed. It darts sideways across
      the cat's view while the cat holds still, freezes the moment the cat
      moves, and scoots off (backwards, or sideways if the map shows no room
      behind) when the cat comes at it.
    - **lure:** edging away in little pulses while the cat watches (prey
      that leaves invites a chase), then waiting.
  - **Choosing:** the choice weighs what suits the distance, the car's
    drifting mood (frisky or mellow), what it hasn't done lately, and what
    has made the cat react.
  - **Reactions:** how far the cat itself moved afterwards, with the car's
    own movement taken out.
  - **Distance:** if the cat drifts off, the car closes in, going round
    obstacles with the map; if the cat comes very close, it eases back.
  - **Rests:** every minute or two it backs off and waits about a minute,
    with an occasional tiny wiggle, inch or glance, before going back to
    bothering the cat.
  - **Lost cat:** if the cat slips away, it looks around, then hunts for it
    again starting near where it was last seen.
- **Learning:** saved in `hunt.json` beside the map database: how much the
  cat reacts to each move, the camera's distance-from-size scale (learned
  from the lidar, or the sonar), and where the cat likes to be.
- **Safety:**
  - **Distance:** it never closes to less than 30 cm from the cat. Backing
    away from a cat just seen in front is always allowed, however close it
    is.
  - **Seeing the cat:** it moves near the cat only after seeing it just now,
    in front, because the camera and sonar can't see beside or behind the
    car.
  - **Each direction checked:**
    - forward moves by the sonar and the map;
    - sideways moves by a glance that way and floor the map *knows* is
      clear;
    - reversing (never more than 25 cm) only through known-clear floor.
  - **People:** nothing drives at a person close ahead (within 30° of
    straight ahead). That covers the forward part of every play move too.
    After the car has turned, or a few seconds have passed, it takes a
    fresh look ahead before driving forward.
  - **Stopping:** the hunt stops after 20 minutes, on a critically low
    battery, or on **STOP**.

In a simulated two-room house (`tests/hunt_sim.py`, `tests/test_hunt.py`):
- **Finding:** the car found the cat in 10 of 10 runs, half of them within
  4 seconds and all within about a minute and a half.
- **Playing:** it played for about five minutes per run, resting two or
  three times and using all 14 moves.
- **Safety:** it never bumped a wall or touched the cat.
- **Learning:** with a cat that only reacts to sideways motion, sidestep,
  shimmy, orbit and zig-zag became its favourites.

**Limit:** a doorway the sonar has drawn partly closed (echoes from its edges)
can stop the car closing in through it. It then plays from where it is
until the cat comes nearer or the map clears.

## Power

The dashboard shows the car's battery as a meter, both in the **Power**
panel and as a small one beside the title, plus the Raspberry Pi's own supply.

- **Car battery:** read from the Freenove board's ADS7830 converter
  (I²C 0x48, channel 2, behind a voltage divider). The scaling follows the
  board version in the kit's `params.json`: version 1 is ×3 with a 3.3 V
  reference, version 2 ×2 with 5.2 V. It assumes two lithium cells
  (`ROBOCAR_BATTERY_CELLS`). Charge is estimated from a resting discharge
  curve, using readings taken with the motors idle for 3 seconds; while
  driving, the sagging voltage is shown separately.
- **Levels** (resting volts per cell):

  | Level | When | Meter |
  |---|---|---|
  | Getting low | below 3.65 V (about 27%) | amber, "charge soon" |
  | Low | below 3.5 V (about 10%) | red, pulsing, "charge now" |
  | Critical | below 3.3 V | auto explore and calibration refuse to start |

- **Raspberry Pi:** from the Pi 5's power chip (`vcgencmd pmic_read_adc`):
  - the 5 V input, the 3.3 V, 1.8 V and core rails;
  - the board's own draw (not USB devices or HATs);
  - the CPU temperature;
  - whether the supply has dipped too low or the CPU was throttled, now or
    since boot (`vcgencmd get_throttled`).
- **Readings that fail** are shown as unavailable, with the reason. The
  battery reading needs the car's board powered and hardware enabled.

## Mapping survey and attended auto explore

**Start mapping survey** records checkpoints while you drive manually.
**Start attended auto explore** enables bounded autonomous motion only when
physical hardware and a range sensor (lidar or ultrasonic) are enabled, the car is stopped,
and the operator confirms the room is clear and attended. The car scans while
stopped, moves in short bursts, then stops and scans again. Autonomous motion
uses full normalized motor output and the default 4095/4095 motor duty cap.

**Spin & roam explore** is the other way to explore, and suits the lidar:
- **Spin:** where it stands, it sweeps, turns a quarter, sweeps again, and so
  on round the full circle (four sweeps, each 60° either side, so each
  overlaps the last by 30° and can be fitted to it). A quarter the map
  already knows is skipped, and it never turns where it is too tight to.
- **Roam:** it then picks the next place to spin: reachable floor that is
  open enough to turn on the spot (32 cm clear of walls, and of unknown floor
  by a car's width), not within 80 cm of where it has spun already, and
  worth the most new floor a full turn there would show, for the driving it
  takes. It drives there along the map's route in legs of up to 60 cm, a
  ping ahead before each one (and a sweep ahead every 1.2 m to stay placed).
  A place it cannot reach is given up, and it never spins twice in the same
  spot.
- **Done:** it stops by itself when nowhere left would show much new, or
  after 15 minutes or 200 moves.

In the simulated two-room house it mapped 97–98% of both rooms and the
corridor from each of three starting points with no bumps, in 22–26 sweeps at
7–9 places. The map marks each place it spun with blue circling arrows.

After each sweep, auto explore decides where to go by **what it could learn there**:

- **Coverage map:** every sweep marks the space the sound crossed as free and
  the places it echoed as walls, in a 5 cm grid shared with earlier runs. The
  edges between known floor and unknown are *frontiers*.
- **Treating missing echoes with care:** the sonar sees walls out to about
  250 cm. A longer reading, or the 300 cm "no echo" value, is open space *or*
  sound that glanced off a wall at a steep angle and bounced away. That is
  the norm in corridors. So a missing echo clears only 50 cm. Runs of echoes
  are fitted as walls the way the 25° beam sees them, and continued. A missing
  echo that would cross one at a glancing angle stops there. One that crosses
  nearly head-on would have echoed, so that is a real gap. A ray passing
  head-on through a mapped wall cell is evidence the wall is not there, which
  erases stray echoes across doorways.
- **Leads:** a stretch of readings much deeper than on both sides (25 cm or
  more, or no echo) is an opening the sonar can see into: a doorway,
  corridor or open area. "A long distance over there on the left" becomes a
  lead the car goes to investigate. A lead stays a target until the floor
  past it is mapped or the car has stood at it.
- **Judging targets:** for each lead and each 50 cm piece of frontier, rays
  are cast from where the car would stand, facing it, to count the unknown
  floor a sweep from there could show. Confirmed walls stop the rays. Walls
  seen only once let part of the view through, since they may be stray echoes.
  Targets are classed as open space, doorway or corridor (walls close on both
  sides), ordinary edge, or lead.
- **Learning:** each target's value is the floor it is expected to show per
  centimetre of driving or turning to get there. The car learns, run after
  run, what share of the promised floor each kind of target really shows
  (scored only when it got there). It also learns how much new floor a
  centimetre of effort has been buying, and stops when even the best target
  promises only a small share of that. What it learned is saved in
  `exploration.json` beside the map database, and shown in the status.
- **Corridors:** between two roughly parallel walls up to 130 cm apart, a
  target along the corridor is reached by following its centre line.
- **Moving:** turns go the short way, up to 150° in one stop, made as
  camera-checked pieces of up to 50° with no sweep in between. A turn that
  only lines up with a waypoint the map shows clear is followed straight away
  by a drive, after a single forward ping, again with no sweep in between.
  Forward legs run up to 60 cm, as far as a straight line toward the
  waypoint allows. They stop 45 cm short of whatever the sonar sees ahead,
  and before the car would come within its radius of any mapped wall.
  Walls met at a glancing angle never echo, so the sonar alone cannot be
  trusted for this. Paths keep away from walls where there is room.
- **Sweep panorama:** at each survey, explore or hunt stop, pictures taken
  every 10° during the sonar sweep, from 60° left to 60° right of straight
  ahead, are joined into a panorama about 170° wide. It is shown in the
  dashboard's "What the camera sees" panels. Each picture contributes the
  slice between its neighbours, so the joins line up.
- **Knows where it is:** the green/red light in the mapping and hunt
  sections. Green needs four things:
  - a map;
  - a match of a sweep to the map (or a recognized place) at some point
    since the service started;
  - a pose estimate within ±30 cm and ±12°;
  - no more than 4 m of driving on wheel counts since that match.

  The estimate grows less certain with every move on wheel counts, so it
  goes red on its own after too much driving without a match. A new run
  (mapping or hunt) started while the service has kept counting the wheels
  keeps the estimate it had, allowing ±10 cm and ±5° in case the car was
  lifted. After a restart it must match the map again first.
  - **Matching:** a sweep is matched to the mapped walls. In clutter, where
    walls don't line up, it is fitted to the coverage map instead. Every
    echo should end on a confirmed obstacle somewhere across the 25° beam,
    and none on floor the map knows is clear. A fit is trusted only when it
    is a clear winner that explains at least 70% of the sweep. In three
    simulated layouts, trusted fits were typically within 4–7 cm and 2–5°,
    though about one in fifty came out badly twisted. So a fit is combined
    with the wheel-based estimate (counted as ±12 cm, ±5°) and dropped when
    the two disagree beyond what both allow.
  - **Limit:** in the workshop, sweeps from almost the same spot agree on
    only about 40% of their ranges, so fits to a thin map are rarely
    trusted. More mapping runs over the same floor firm the map up.
- **Camera-measured turns:** before each turn piece the car needs a strip of
  pictures every 10° across the pan on the turn side (0–60°). While exploring,
  the first piece's strip is taken during the stop's sonar sweep, which
  passes those angles anyway. Pictures go at the sweep's nearest 3° step, with
  the same exposure and blur checks, so the head makes one pass instead of
  two. That saves 0.4–1.2 s a stop with realistic camera timing. The strip is
  only reused if the car and head have not moved since the sweep; later
  pieces of a big turn, and anything else, take a strip of their own. The car then turns
  closed-loop on wheel odometry. Afterwards its new forward view is matched
  against every strip picture, and at least two must agree on how far it
  really turned. If the turn is off by more than 3° and the measurement
  roughly agrees with the wheels, the car makes up to two corrective turns
  (at most 20° each). The measured turn also recalibrates odometry and fixes
  the pose for the next sweep. Without trustworthy images it relies on
  odometry.
- **Tight spots:** too close to a wall to spin (a side reading under 20 cm),
  it slides 12 cm sideways toward the roomier side (the wheels are mecanum).
  Only if neither side has room does it back up once; the rear is not
  sensed, so keep the area behind the car clear.
- **Finished:** with nothing left worth a trip, it looks round (three 110°
  turns) and then stops with "explored".

In a simulated two-room floor plan joined by an 80 cm corridor, with missing
echoes reading 300 cm, the previous explorer mapped 49% of the floor on
average over eight starting points. It crossed into the second room once,
ended "blocked" four times and bumped into walls 183 times. This one mapped
77% on average, explored both rooms from six of the eight starts (81–94%)
and never bumped or got blocked. From the best starts it reached 85% of the
floor within 30–40 sweeps. `tests/explore_sim.py` is the floor plan and
`tests/test_exploration.py` checks it.

If the sensor returns no readings at all, it stops with a sensor fault. The
dashboard shows the reason for the current move. Each explore session is
limited to four minutes and 60 moves. **STOP** cancels the session and stops
the drive output. A one-second watchdog stops stale commands, while
autonomous moves refresh the command until they end. Exploration never
starts at boot.

**Pictures:** during a mapping session the camera stays open, so auto-exposure
settles between stops. Before each picture it waits until two quick frames in
a row match (the car and head have stopped moving), up to 2 seconds. Each
stop's picture is checked for brightness, clipping, sharpness and usable
detail. If it isn't decent, it is retaken: a blurred one at the same exposure
after a pause, otherwise with adjusted exposure (compensation from −1 to
+4 EV, then manual exposures with gain up to 95 ms at gain 16), up to five
tries. The best picture is kept, and the setting that worked is where the
next stop starts. (The head camera's pictures on the 10-31 runs were all at
the old longest setting, 50 ms at gain 12, and most were still dark: the
workshop at night needs the longer ones. The car stands still for them.)
The **What the camera sees** panel shows:

- the latest stop picture with its detected features and exposure;
- stop-to-stop matching (green lines are matches kept, red rejected) with the
  measured camera turn;
- the latest turn check against its strip picture;
- the latest recognized place against the stored picture.

**Calibrate turns & steps** (attended; needs the head camera) measures how far the car really turns
and moves. Face it toward a flat wall about 1–1.5 m away with room to turn.
It photographs a strip across the pan, then turns 10°, 20° and 30° left and
right, adjusting each command until the camera says it landed within 2°. Each
picture gets the same exposure check as a stop's picture. It then uses the
camera to return to its starting heading and makes 5, 10 and 20 cm steps
toward the wall, each followed by the same burst in reverse and, if the car
crept, a short burst back to where the steps began. It adjusts the burst
length until the sonar says each step is right (within 7%), changing it by
at most 1.6× per try and ignoring readings that jump impossibly far. It refuses to start with anything within 25 cm, and stops stepping
before the wall is closer than 45 cm. Results are saved beside the map
database (not cleared by reset), shown on the dashboard, and used for every
turn and step afterwards; each session keeps refining them.

Use **Reset mapping database** to permanently delete all saved observations.
The dashboard asks for confirmation, and reset is unavailable while mapping is
active. The motor duty cap and watchdog can be changed with
`ROBOCAR_MAX_MOTOR_DUTY` (maximum 4095) and `ROBOCAR_COMMAND_TIMEOUT`.
Existing systemd installations must update `/etc/robocar/robocar.env` and
restart the service for changed values to take effect.

Stay beside the car and be ready to cut battery power. Use only a clear,
single-level room without people, pets, stairs, drop-offs, or fragile objects.
The software stop and forward-facing ultrasonic scanner are not safety-rated
protective devices; the sensor cannot see behind the car or guarantee detection
of every surface. Test with wheels raised first, then at low speed in a cleared
area. Keep unauthenticated dashboard access restricted to a trusted network.

At each stop RoboCar sweeps the head across the room, always stepping the
same way so servo backlash cancels. With the lidar it reads every degree
(121 directions); without it, the ultrasonic every 3° (40 directions, three
pings each). From a lidar sweep each reading is a point on a surface; the
corrections below are for the sonar's wide beam. It turns the sweep into **surfaces**: fine wall segments joined at
corners into contours, plus point features such as edges and legs. Openings
and passages stay open: echoes are pulled back to the edges they came from,
walls break wherever their points leave a gap, and contours never close
across space a beam passed through. The beam-width correction, measured from
the walls themselves, straightens walls that a raw plot would bow. Each new
sweep refines, extends, merges or retires the surfaces it sees again.

To place each sweep, RoboCar estimates how the car moved:

- **Wheel odometry** integrates the actual ramped motor output.
- **The camera** compares views with the previous stop to measure the turn
  precisely and the direction of travel. When the forward view is bland or
  disagrees with the wheels, it also takes side pictures during the sweep.
- **The sonar** fits the new sweep against surfaces already mapped, which
  supplies the metric correction the camera cannot.

Each source is weighted by its uncertainty, and disagreements are tested
before they are trusted. Odometry scale is relearned during each run. The
dashboard draws confirmed surfaces solid, surfaces seen once dashed, raw
echoes as dots, and the car's path with its pose uncertainty. See
[docs/SURFACE_MAPPING.md](docs/SURFACE_MAPPING.md) for the design and the
constants to calibrate on the car.

This is still an approximate sketch, not a validated or safety-rated map.
Ultrasonic sensors miss soft and very oblique surfaces, monocular images have
no absolute scale, and wheel slip varies. Do not use it for autonomous
navigation or collision avoidance; exploration still steers from the raw
sweep. A sonar sweep takes about 11–12 seconds, a lidar sweep about 7.

**Remembering places:** every stop's picture features, sweep and pose are kept
in the database across sessions. At each stop RoboCar checks whether it has
seen the place before. It matches picture features, verifies the camera
geometry, then confirms with a sonar fit against that place's remembered
surfaces. Returning to a place within a session closes a loop: a pose graph
spreads the accumulated drift back along the path, and matches that
contradict everything else are thrown out. Recognizing a place from an
earlier session moves the new session into the old one's frame, so the map
becomes one persistent house map built from every joined session, and
exploration prefers ground no session has covered yet. The dashboard draws
earlier sessions faintly and recognized places as purple dashed links. The
whole graph is rebuilt from the database when RoboCar restarts.

**What the camera recognizes (AI HAT+ or CPU):** each stop's picture goes
through the object detector. A detection's direction comes from where it sits
in the picture and where the head was pointing; its distance comes from the
sonar readings in that direction. Several things follow from that:

- **People and pets** move, so sonar readings that hit them are flagged and
  kept out of the walls, the coverage map, scan matching and place memory.
  They still count for safety: the explorer will not drive at a person or
  animal within 25° of ahead and closer than 150 cm (or at an unknown
  distance). A stop's sighting of one is shown in orange in the dashboard's
  "What the camera recognized" picture.
- **Furniture and appliances** become landmarks on the map. These are chairs,
  couches, beds, tables, the TV, fridge, oven, microwave, sink, toilet, plants
  and so on. A sighting with a sonar range is a point. Without one, two
  sightings from places at least 30 cm apart are triangulated. Repeated
  sightings are merged, and a landmark appears once it has been seen twice
  or once with high confidence.
- **Rooms** are named from their landmarks: a bed means a bedroom, an oven or
  fridge a kitchen, a couch or TV a living room, a toilet a bathroom.
  Landmarks belong to one room when they are within 2.5 m of each other with
  no confirmed wall between them.
- **Corrections carry over:** the semantic map is rebuilt from the stored
  sightings at the pose graph's optimized poses, so loop closures move it too.

**Each run builds on the last:** starting a mapping run never clears the map.
The run starts from the newest stop of any earlier run, with every earlier
sweep in the map, and its first stop is tied to that stop in the pose graph.
That tie is loose (±25 cm, ±20°) in case the car was moved by hand in between.
Wheel odometry since that stop is used when the service kept running. The
first sweep's fit against the known walls, or a recognized place, then pins
the pose down. Each run's sweeps refine the same surfaces and coverage, and
exploration heads for whatever is still unmapped. Only **Reset** clears the
database.

By default, survey data is stored at
`~/.local/share/robocar/mapping.sqlite3` (or under `$XDG_DATA_HOME` when set).
Set `ROBOCAR_MAP_PATH` to use another database location.
Mapping stops and reports an error rather than starting a new checkpoint when
free space drops below 256 MB. The database file is restricted to the Pi
account; survey images are available through the same unauthenticated dashboard
API, so keep the service on a trusted network.

For a useful persistent room-scale occupancy map, add a calibrated 2D lidar
and use a localization/SLAM stack (with wheel odometry or an IMU if available).
With the current sensors, use the display as an approximate survey record only.

## Start at boot and connect from your Mac

After installing RoboCar into the Raspberry Pi user's Python environment, run
the installer once from a terminal on the Pi:

```bash
cd /home/sondericker/Documents/Python/RoboCarv1
./scripts/install-service.sh
```

It asks whether to enable physical motor and pan/tilt outputs (default: off),
then uses `sudo` to install and enable a systemd service. Enter your normal
account's sudo password if prompted. Camera processing is disabled at boot.
After setup, the dashboard starts automatically whenever the Pi boots and
listens on the LAN without authentication. Find the Pi address using
`hostname -I`, then visit `http://<car-ip>:5000` from the MacBook. Anyone on a
network that can reach this port can view the camera and control the car. Keep
it only on a trusted home network; do not forward the HTTP port through your
router or use it on an untrusted Wi-Fi network.

Check service logs with `sudo journalctl -u robocar -f`. Stop the running
application with:

```bash
./scripts/stop-service.sh
```

This stops RoboCar now but leaves automatic startup enabled. To stop it and
prevent it from starting at the next boot, run `sudo systemctl disable --now
robocar`.

## Code layout

All under `src/robocar/`:
- `web.py`, `__main__.py`, `config.py`: the dashboard server and its settings.
- `drive.py`, `range_sensor.py`, `camera.py`, `power.py`, `detector.py`: the
  hardware (motors and pan/tilt head, sonar and lidar, cameras, battery, the
  AI HAT+).
- `mapping.py`: the controller that runs mapping, exploring, the hunt and
  calibration, and keeps the map. It builds on:
  - `ranging.py` (sonar and lidar together, sweep pacing),
  - `pictures.py` (judging pictures, turns between views),
  - `explore_plan.py` (the frontier explorer's next move),
  - `exploration.py` (coverage grid, routes, spin stations, scan matching),
  - `surfaces.py`, `localization.py`, `places.py`, `semantics.py` (walls,
    poses, recognized places, recognized objects),
  - `store.py` (the database) and `pointcloud.py` (lidar points to draw).
- `hunt.py` (the hunt and play engine), `watch.py` (the hunt camera's
  watcher), `calibration.py`, and `bodies.py` (the real car as those engines
  see it; the tests use simulations instead).

## Kit review

The supplied Freenove tree contains 43 Python files: a Raspberry Pi server
with motor/PCA9685, servo, ultrasonic, infrared, ADC, LEDs, and Picamera2
modules; a PyQt5 desktop client/server UI; a TCP command/video protocol; and
the bundled `rpi-ws281x` library. Its setup scripts install legacy/board
specific dependencies and edit Raspberry Pi boot configuration. RoboCar does
not run those scripts or modify the kit source. It integrates the kit's
`Ordinary_Car` driver only when physical hardware is explicitly enabled.

The kit README identifies its source as Creative Commons
Attribution-NonCommercial-ShareAlike 3.0. RoboCar's adapter imports the kit
from the provided local source path rather than copying its implementation.
Review the upstream license and attribution requirements before redistributing
any kit code or derivatives.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for architecture, phased
milestones, safety limits, and decisions to verify on the actual car.
