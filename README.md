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
tilt 100°, and **Center head** returns it there. The initial software travel
is limited to 30°–150° to avoid the servo end stops; calibrate the safe range
on the actual camera head before widening it.

When physical hardware is enabled, the dashboard also displays the live
ultrasonic range in centimeters. The sensor samples in the background and
reports missing echoes or read errors instead of presenting an old value as
current. With hardware disabled, the UI identifies the range finder as
disabled; it does not simulate a distance. The reading is informational and
does not currently interlock or stop drive commands.

## Mapping survey and attended auto explore

**Start mapping survey** records checkpoints while you drive manually.
**Start attended auto explore** enables bounded autonomous motion only when
physical hardware and the ultrasonic sensor are enabled, the car is stopped,
and the operator confirms the room is clear and attended. The car scans while
stopped, moves in short bursts, then stops and scans again. Autonomous motion
uses full normalized motor output and the default 4095/4095 motor duty cap.

After each sweep it decides where to go by **what it could learn there**:

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
+4 EV, then manual exposures up to 50 ms with gain), up to five tries. The best
picture is kept, and the setting that worked is where the next stop starts.
The **What the camera sees** panel shows:

- the latest stop picture with its detected features and exposure;
- stop-to-stop matching (green lines are matches kept, red rejected) with the
  measured camera turn;
- the latest turn check against its strip picture;
- the latest recognized place against the stored picture.

**Calibrate turns & steps** (attended) measures how far the car really turns
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

At each stop RoboCar sweeps the ultrasonic sensor across 40 directions, 3°
apart (three pings each, always stepping the same way so servo backlash
cancels). It turns the sweep into **surfaces**: fine wall segments joined at
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
sweep. A sweep takes about 11–12 seconds, so a two-minute explore session fits
fewer bursts than before.

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
