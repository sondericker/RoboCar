import logging
import math
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from robocar.camera import CameraService, connected_models
from robocar.config import Config
from robocar.detector import ObjectDetector
from robocar.drive import (
    DriveController,
    DriveInhibitedError,
    FreenoveDriver,
    FreenovePanTilt,
    HARDWARE_PAN_SCALE,
    PAN_LIMITS,
    TILT_LIMITS,
    PanTiltInhibitedError,
    set_pan_scale,
    PanTiltController,
    SimulatedDriver,
    SimulatedPanTilt,
)
from robocar.imu import IMU_SEARCH_BUSES, MPU6500, ImuController
from robocar.mapping import MappingController
from robocar.store import MappingStore
from robocar.power import BoardADC, PiPower, PowerMonitor, board_version
from robocar.range_sensor import FreenoveRangeSensor, RangeSensorController, TFLunaRangeSensor


logger = logging.getLogger(__name__)
# The hunt's video: a frame this often, ending a while after the hunt stops.
HUNT_STREAM_INTERVAL_SECONDS = 0.2
HUNT_STREAM_IDLE_FRAMES = 25


def _open_hardware(name: str, opener):
    """Open a piece of the car's hardware; on failure log why and return the reason.

    A car board that is switched off or unplugged must not take the whole
    dashboard (and the camera with it) down: that part is reported as
    unavailable instead."""
    try:
        return opener(), None
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("%s unavailable: %s", name, reason)
        return None, reason


def _open_cameras(config: Config, detector) -> tuple[CameraService, CameraService, bool]:
    """(forward camera, head camera, whether the mapping camera is on the head).

    The forward camera is the dashboard's live view and the hunt's eyes; the
    head camera is mapping's. In hardware mode cameras are found by sensor
    name; when only one is connected it does both jobs, and with neither
    found by name (or outside hardware mode) the configured camera number is
    used as before."""

    def service(model: str | None, index: int = config.camera_index) -> CameraService:
        return CameraService(
            config.camera_backend, index, config.frame_width, config.frame_height, detector, config.camera_enabled, model=model
        )

    # Real cameras are looked up only in hardware mode (never from tests).
    present = connected_models(config.camera_backend) if config.hardware_enabled else []
    has = lambda model: bool(model) and any(model in name for name in present)
    head, forward = config.head_camera_model, config.forward_camera_model
    if has(head) and has(forward):
        return service(forward), service(head), True
    if has(forward):
        camera = service(forward)
        return camera, camera, False
    if has(head):
        camera = service(head)
        return camera, camera, True
    camera = service(None)
    return camera, camera, config.camera_on_head


def _open_lidar(config: Config) -> RangeSensorController | None:
    """The head's lidar, if there is one: None when it is turned off, or in
    auto mode when it does not answer (the sonar then does all the ranging)."""
    if not config.hardware_enabled or config.lidar == "off":
        return None
    try:
        driver = TFLunaRangeSensor(config.lidar_bus)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if config.lidar == "auto":
            logger.info("No lidar found on I2C bus %d (%s); ranging with the sonar only", config.lidar_bus, reason)
            return None
        logger.error("Lidar unavailable: %s", reason)
        return RangeSensorController(None, True, fault=reason, name="Lidar")
    logger.info("TF-Luna lidar found on I2C bus %d", config.lidar_bus)
    return RangeSensorController(driver, True, sample_interval=0.1, name="Lidar")


def _open_imu(config: Config) -> ImuController | None:
    """The gyro, if there is one: None when it is turned off, or in auto mode
    when none answers (turns are then judged from the wheels)."""
    if not config.hardware_enabled or config.imu == "off":
        return None
    buses = (config.imu_bus,) if config.imu_bus is not None else IMU_SEARCH_BUSES
    reasons = []
    for bus in buses:
        if not Path(f"/dev/i2c-{bus}").exists():
            reasons.append(f"no I2C bus {bus}")
            continue
        try:
            driver = MPU6500(bus)
        except Exception as exc:
            reasons.append(f"bus {bus}: {type(exc).__name__}: {exc}")
            continue
        logger.info("%s gyro found on I2C bus %d at 0x%02x", driver.model, bus, driver.address)
        return ImuController(driver)
    reason = "; ".join(reasons) or "no I2C bus to search"
    if config.imu == "auto":
        logger.info("No IMU found (%s); turns are judged from the wheels", reason)
        return None
    logger.error("IMU unavailable: %s", reason)
    return ImuController(None, fault=reason)


def create_app(config: Config | None = None) -> Flask:
    config = config or Config.from_env()
    drive_fault = pan_tilt_fault = range_fault = None
    lidar = _open_lidar(config)
    # The sonar is left off (never opened, so silent) when the lidar does its
    # job: its 40 kHz pings, sent all the time it is open, are within a
    # cat's hearing. ROBOCAR_SONAR=on brings it back (glass, mirrors).
    sonar_wanted = config.sonar == "on" or (config.sonar == "auto" and (lidar is None or not lidar.status()["enabled"]))
    if config.hardware_enabled:
        driver, drive_fault = _open_hardware("Motor board", lambda: FreenoveDriver(config.freenove_server_path))
        pan_tilt_driver, pan_tilt_fault = _open_hardware(
            "Pan/tilt servos", lambda: FreenovePanTilt(config.freenove_server_path)
        )
        range_sensor_driver = None
        if sonar_wanted:
            range_sensor_driver, range_fault = _open_hardware(
                "Ultrasonic sensor", lambda: FreenoveRangeSensor(config.freenove_server_path)
            )
        else:
            logger.info("Ultrasonic sensor left off: the lidar does the ranging (ROBOCAR_SONAR=on to use it)")
    else:
        driver, pan_tilt_driver, range_sensor_driver = SimulatedDriver(), SimulatedPanTilt(), None
    imu = _open_imu(config)
    try:
        drive = DriveController(
            driver or SimulatedDriver(),
            config.max_motor_duty,
            config.command_timeout,
            config.hardware_enabled,
            fault=drive_fault,
            imu=imu,
        )
    except OSError as exc:
        # The board answered when opened but not when first told to stop
        # (still powering up): run without motors rather than not at all.
        drive_fault = f"{type(exc).__name__}: {exc}"
        logger.error("Motor board unavailable: %s", drive_fault)
        drive = DriveController(
            SimulatedDriver(), config.max_motor_duty, config.command_timeout, config.hardware_enabled, fault=drive_fault, imu=imu
        )
    pan_tilt = PanTiltController(pan_tilt_driver or SimulatedPanTilt(), config.hardware_enabled, fault=pan_tilt_fault)
    if pan_tilt_driver is not None and config.hardware_enabled:
        # The car's pan servo turns less than told; mapping refines this.
        set_pan_scale(HARDWARE_PAN_SCALE)
    range_sensor = RangeSensorController(range_sensor_driver, config.hardware_enabled and sonar_wanted, fault=range_fault)
    adc, adc_fault = (
        _open_hardware("Battery monitor", lambda: BoardADC(board_version(config.freenove_server_path)))
        if config.hardware_enabled
        else (None, None)
    )
    power = PowerMonitor(
        adc,
        PiPower(),
        lambda: any(abs(float(drive.status()[axis])) > 0.01 for axis in ("throttle", "strafe", "steering")),
        cells=config.battery_cells,
        adc_fault=adc_fault,
        adc_opener=(lambda: BoardADC(board_version(config.freenove_server_path))) if config.hardware_enabled else None,
    )
    detector = ObjectDetector(
        config.model_path,
        config.labels_path,
        config.confidence,
        config.nms_threshold,
        hailo_model=config.hailo_model_path,
    )
    camera, head_camera, camera_on_head = _open_cameras(config, detector)
    mapping = MappingController(
        camera=head_camera,
        drive=drive,
        pan_tilt=pan_tilt,
        range_sensor=range_sensor,
        store=MappingStore(config.mapping_database),
        detector=detector,
        camera_on_head=camera_on_head,
        hunt_camera=camera,
        lidar=lidar,
    )

    app = Flask(__name__)
    # The dashboard's live view (and the hunt's eyes) is the forward camera;
    # mapping uses the head camera. With one camera they are the same.
    app.extensions["robocar_camera"] = camera
    app.extensions["robocar_head_camera"] = head_camera
    app.extensions["robocar_detector"] = detector
    app.extensions["robocar_power"] = power
    mapping.battery_level = power.battery_level
    # The drive keeps the motors' torque up as the battery runs down.
    drive.set_supply(power.volts_now, config.battery_cells)

    def battery_too_low() -> str | None:
        if power.battery_level() == "critical":
            volts = power.status()["battery"]["resting_volts"]
            return f"Battery critically low ({volts:.2f} V); charge it before driving on its own"
        return None
    app.extensions["robocar_drive"] = drive
    app.extensions["robocar_pan_tilt"] = pan_tilt
    app.extensions["robocar_range_sensor"] = range_sensor
    app.extensions["robocar_lidar"] = lidar
    app.extensions["robocar_imu"] = imu
    app.extensions["robocar_mapping"] = mapping

    @app.get("/")
    def dashboard():
        return render_template("index.html")

    @app.get("/video.mjpg")
    def video():
        return Response(
            camera.stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/status")
    def status():
        return jsonify(
            {
                "camera": camera.status(),
                "head_camera": head_camera.status() if head_camera is not camera else None,
                "drive": drive.status(),
                "range": range_sensor.status(),
                "lidar": lidar.status() if lidar is not None else None,
                "imu": imu.status() if imu is not None else None,
                "mapping": mapping.status(),
                "power": power.status(),
            }
        )

    @app.post("/api/mapping")
    def mapping_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        try:
            if payload["enabled"]:
                mapping.start()
            else:
                mapping.stop()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.post("/api/mapping/explore")
    def autonomous_mapping_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        if payload["enabled"] and not isinstance(payload.get("attended"), bool):
            return jsonify({"error": "attended must be a boolean"}), 400
        strategy = payload.get("strategy", "room")
        if not isinstance(strategy, str):
            return jsonify({"error": "strategy must be a string"}), 400
        if payload["enabled"] and (reason := battery_too_low()):
            return jsonify({"error": reason}), 409
        try:
            if payload["enabled"]:
                mapping.start_autonomous(payload["attended"], strategy)
            else:
                mapping.stop()
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.post("/api/hunt")
    def hunt_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        if payload["enabled"] and not isinstance(payload.get("attended"), bool):
            return jsonify({"error": "attended must be a boolean"}), 400
        if payload["enabled"] and (reason := battery_too_low()):
            return jsonify({"error": reason}), 409
        try:
            if payload["enabled"]:
                mapping.start_hunt(payload["attended"])
            else:
                mapping.stop()
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.hunt_status())

    @app.get("/api/hunt")
    def hunt_status():
        return jsonify(mapping.hunt_status())

    @app.get("/api/hunt/frame.jpg")
    def hunt_frame():
        frame = mapping.hunt_frame()
        if frame is None:
            return jsonify({"error": "No hunt picture yet"}), 404
        return Response(frame, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/hunt.mjpg")
    def hunt_video():
        def frames():
            last = None
            idle = 0
            while idle < HUNT_STREAM_IDLE_FRAMES:
                frame = mapping.hunt_frame()
                running = mapping.hunt_status()["running"]
                if frame is not None and (frame is not last or running):
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    last = frame
                idle = 0 if running else idle + 1
                time.sleep(HUNT_STREAM_INTERVAL_SECONDS)

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/calibration")
    def calibration_status():
        return jsonify({"calibration": mapping.calibration()})

    @app.post("/api/calibration")
    def calibration_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        if payload["enabled"] and not isinstance(payload.get("attended"), bool):
            return jsonify({"error": "attended must be a boolean"}), 400
        if payload["enabled"] and (reason := battery_too_low()):
            return jsonify({"error": reason}), 409
        try:
            if payload["enabled"]:
                mapping.start_calibration(payload["attended"])
            else:
                mapping.stop()
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.get("/api/mapping/observations")
    def mapping_observations():
        return jsonify({"observations": mapping.observations()})

    @app.get("/api/mapping/coverage.png")
    def mapping_coverage():
        image = mapping.coverage_png()
        if image is None:
            return jsonify({"error": "Nothing swept yet"}), 404
        return Response(image, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/mapping/vision")
    def mapping_vision():
        return jsonify({"panels": mapping.vision_panels()})

    @app.get("/api/mapping/vision/<name>.jpg")
    def mapping_vision_image(name: str):
        image = mapping.vision_image(name)
        if image is None:
            return jsonify({"error": "No such picture yet"}), 404
        return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/mapping/cloud")
    def mapping_cloud():
        return jsonify(mapping.lidar_cloud())

    @app.get("/api/mapping/surfaces")
    def mapping_surfaces():
        return jsonify(mapping.surface_map())

    @app.post("/api/mapping/reset")
    def mapping_reset():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("confirm") is not True:
            return jsonify({"error": "confirm must be true"}), 400
        try:
            mapping.reset()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(mapping.status())

    @app.get("/api/mapping/observations/<int:observation_id>/image")
    def mapping_observation_image(observation_id: int):
        image = mapping.image(observation_id)
        if image is None:
            return jsonify({"error": "Observation not found"}), 404
        return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/camera")
    def camera_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
            return jsonify({"error": "enabled must be a boolean"}), 400
        enabled = camera.set_enabled(payload["enabled"])
        return jsonify({"enabled": enabled})

    @app.post("/api/drive")
    def drive_command():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        throttle = payload.get("throttle")
        strafe = payload.get("strafe")
        steering = payload.get("steering")
        if (
            isinstance(throttle, bool)
            or isinstance(strafe, bool)
            or isinstance(steering, bool)
            or not isinstance(throttle, (int, float))
            or not isinstance(strafe, (int, float))
            or not isinstance(steering, (int, float))
        ):
            return jsonify({"error": "throttle, strafe, and steering must be numbers"}), 400
        try:
            throttle_value, strafe_value, steering_value = (
                float(throttle),
                float(strafe),
                float(steering),
            )
        except (OverflowError, ValueError):
            return jsonify({"error": "throttle, strafe, and steering must be finite numbers"}), 400
        if not all(math.isfinite(value) for value in (throttle_value, strafe_value, steering_value)):
            return jsonify({"error": "throttle, strafe, and steering must be finite numbers"}), 400
        try:
            command = drive.set_command(throttle_value, strafe_value, steering_value)
        except DriveInhibitedError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"throttle": command[0], "strafe": command[1], "steering": command[2]})

    @app.post("/api/stop")
    def stop():
        if mapping.status()["enabled"]:
            mapping.stop()
        drive.stop()
        return jsonify({"stopped": True})

    @app.get("/api/pan-tilt")
    def pan_tilt_status():
        return jsonify(pan_tilt.status())

    @app.post("/api/pan-tilt")
    def pan_tilt_command():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        pan, tilt = payload.get("pan"), payload.get("tilt")
        if (
            isinstance(pan, bool)
            or not isinstance(pan, int)
            or isinstance(tilt, bool)
            or not isinstance(tilt, int)
            or not PAN_LIMITS[0] <= pan <= PAN_LIMITS[1]
            or not TILT_LIMITS[0] <= tilt <= TILT_LIMITS[1]
        ):
            return jsonify({"error": "pan and tilt must be integers from 30 to 150"}), 400
        try:
            angles = pan_tilt.set_angles(pan, tilt)
        except PanTiltInhibitedError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"pan": angles[0], "tilt": angles[1]})

    return app
