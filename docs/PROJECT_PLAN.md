# RoboCar project plan

## Product goal

Build a Raspberry Pi-based 4WD rover that can be operated safely from a VNC
desktop, observe its surroundings with a camera, detect and label objects, and
eventually map and patrol the house while locating people or the cat.

## What the supplied kit provides

- 43 Python files: 21 server-side files, 8 client Python files, 11 bundled
  `rpi-ws281x` library Python files, and 3 setup scripts.
- Server-side motor PWM uses a PCA9685 at I²C address `0x40`; the ordinary
  wheel driver exposes four signed motor duty values.
- The kit includes ultrasonic distance, infrared line sensors, ADC/photoresistor,
  servo, buzzer, LED, and Picamera2 support.
- Its desktop interface is PyQt5, with separate TCP command and video streams.
  The client-side face detector is Haar-cascade based; it is not a general
  object detector.
- The kit setup scripts install a mix of OS/Python packages and one script
  edits `/boot/firmware/config.txt`. RoboCar does not run those scripts
  automatically.

## Confirmed hardware profile

- Kit: Freenove FNK0043, Rev. 2/B connection board.
- Chassis: mecanum wheels.
- Camera: Raspberry Pi IMX477, accessed through Picamera2/libcamera.
- Range sensor: Freenove ultrasonic sensor on the kit's servo scanner.
- Hardware and sensors, including the ultrasonic detector, have been bench
  tested by the owner.

## Current starting point

The code in `src/robocar` separates configuration, camera capture, YOLO ONNX
detection through ONNX Runtime, manual drive/watchdog, and the Flask dashboard.
The web UI runs in the Pi's VNC browser, streams a camera with overlays, reports
object labels, and has hold-to-drive buttons plus keyboard overrides. A
simulation driver is used by default. Physical output imports the Freenove
`Ordinary_Car` module only after explicit hardware opt-in.

The drive controller uses the Freenove `set_motor_model(FL, BL, FR, BR)` order.
Its normalized mecanum mix follows the kit's own formula:
`FL = forward + strafe - rotation`, `BL = forward - strafe - rotation`,
`FR = forward - strafe + rotation`, and
`BR = forward + strafe + rotation`. Combined commands are scaled together to
preserve their direction while staying within the configured duty limit.

The pan/tilt head uses Freenove PCA9685 servo channels 8 and 9 (kit servo
indices 0 and 1). Dashboard travel is initially limited to 30°–150° around the
kit's 90° center; calibrate the head's mechanical limits before changing that.

The default `yolo11n.onnx` COCO model includes `person` and `cat` classes and
downloads from the Ultralytics asset release on first run if absent. Inference
uses ONNX Runtime CPU providers with XNNPACK/oneDNN selected when available.
Validate model output compatibility and latency on the exact Raspberry Pi OS
image.

## Milestones

1. **Hardware profile verified:** hardware and sensors have been bench-tested.
   Record Pi OS/Python version, camera detection, battery condition, motor
   polarity and safe duty limits. Verify each mecanum movement axis with the
   wheels raised.
2. **Reliable manual drive:** test dashboard connectivity in VNC, IMX477 capture,
   stop behavior, watchdog, forward/reverse, strafe, rotation, and duty
   calibration. Keep unauthenticated remote access restricted to the trusted
   home LAN.
3. **Perception:** validate the included lightweight COCO detector's
   person/cat precision and latency on recorded frames, and make confidence,
   false positives and model health visible. Use object detection for awareness,
   not as a safety sensor.
4. **Survey and attended auto explore (implemented, experimental):** persist
   stationary camera frames and 40-direction ultrasonic sweeps, then permit
   bounded movement bursts only after an explicit attended start. Stop on a
   missing forward echo, insufficient forward/side clearance, session timeout,
   or movement limit. Sweeps are reconstructed into refinable wall/feature
   surfaces and contours. Poses fuse wheel odometry, camera yaw/travel
   direction (with side views when needed) and sonar registration against the
   surface map; see `docs/SURFACE_MAPPING.md`. Treat the display as an
   approximate sketch and do not use it for navigation or collision
   avoidance. A dependable room map still needs calibrated sensor geometry and
   better ranging/localization hardware, preferably 2D lidar with wheel
   encoders or an IMU.
5. **Safety and teleoperation:** add a physical power/enable strategy,
   obstacle-distance interlock, command audit, better stop behavior on browser
   disconnect, and tests with stalled camera/network/sensor failure.
6. **Supervised patrol:** define geofenced routes, safe speeds, return-to-home,
   low-battery handling, human supervision, privacy rules, and manual takeover.
   Do not enable autonomous patrol until safety and localization are validated.

## Early acceptance criteria

- UI accessible from a VNC session at `http://127.0.0.1:5000`.
- A systemd service starts the unauthenticated dashboard automatically at boot
  for trusted-LAN access; network peers can view video and control the car.
- Dashboard pan/tilt controls move both axes within calibrated safe travel.
- Dashboard can pause camera capture and object detection, releasing the camera
  and stopping frame delivery until resumed.
- Camera processing is paused by default at app launch and system boot.
- Camera failures and model configuration errors are surfaced, not silently
  treated as successful perception.
- Motor commands remain simulated until explicitly enabled.
- While hardware is enabled, releasing controls or losing command refresh stops
  output within the configured watchdog timeout.
- Mecanum controls independently command forward/reverse, lateral strafe, and
  rotation, with wheel outputs normalized to the configured duty limit.
- Mapping records synchronized camera frames, ultrasonic range and scan angle;
  show the map's relative/uncertain status rather than presenting it as
  validated global localization.
- For a recorded test set, report per-class precision/recall and inference
  latency before relying on cat/person detections.

## Dependency notes

- Core Python dependencies: Flask, NumPy, OpenCV, ONNX Runtime, and
  python-dotenv.
- Pi-only camera: Raspberry Pi OS `python3-picamera2` and libcamera packages;
  these are not pip dependencies.
- Freenove motor and ultrasonic adapters: the kit's own `motor.py`,
  `pca9685.py`, and `ultrasonic.py`, OS `python3-smbus`, and `gpiozero`; I²C
  must be enabled.
- Pan/tilt uses PCA9685 channels 8/9 on the same board; test head movement with
  the chassis secured before extending travel limits.
- The tested IMX477 is accessed through Raspberry Pi OS Picamera2/libcamera;
  verify sensor detection and configure camera resolution/frame rate before
  mapping sessions.
- Ultrasonic occupancy mapping needs a timestamped scan angle and robot pose for
  every range. The kit's rotating servo provides scan angle commands but not
  absolute robot pose; poses are estimated from wheel odometry, camera feature
  matching and sonar registration against previously mapped surfaces.
- RoboCar downloads default YOLO11n ONNX weights from the Ultralytics asset
  release on first use; model files remain out of source control. Verify the
  model's license before distribution.

## Open decisions to resolve on the car

- Exact Raspberry Pi model/OS version, camera orientation and the IMX477 camera
  port in use.
- Ultrasonic scan servo zero-angle calibration and measured range reliability.
- Whether relative visual mapping is sufficient or metric localization hardware
  (such as an IMU, encoders, depth sensor or lidar) will be needed.
- Required camera frame rate and detection latency on the target Pi.
- Whether a future deployment should add authentication or TLS if the dashboard
  needs access beyond the trusted home LAN.
- Desired mapping area, map persistence, and acceptable localization drift.
