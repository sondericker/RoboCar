import atexit
import logging
import os
import signal
import threading

from dotenv import load_dotenv

from robocar.config import Config
from robocar.web import create_app

logger = logging.getLogger("robocar")
# Shutting down may take this long before the process is ended anyway (the
# drive's own watchdog has stopped the motors by then).
SHUTDOWN_LIMIT_SECONDS = 15.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    config = Config.from_env()
    app = create_app(config)

    def close_resources() -> None:
        def give_up() -> None:
            logger.error("Shutdown still not finished after %.0f s; ending the process", SHUTDOWN_LIMIT_SECONDS)
            logging.shutdown()
            os._exit(1)

        watchdog = threading.Timer(SHUTDOWN_LIMIT_SECONDS, give_up)
        watchdog.daemon = True
        watchdog.start()
        extensions = app.extensions
        steps = [
            ("mapping", extensions["robocar_mapping"]),
            ("power monitor", extensions["robocar_power"]),
            ("drive", extensions["robocar_drive"]),
            ("pan/tilt", extensions["robocar_pan_tilt"]),
            ("ultrasonic", extensions["robocar_range_sensor"]),
            ("lidar", extensions["robocar_lidar"]),
            # After the drive, which reads its heading.
            ("IMU", extensions["robocar_imu"]),
            ("forward camera", extensions["robocar_camera"]),
            (
                "head camera",
                extensions["robocar_head_camera"] if extensions["robocar_head_camera"] is not extensions["robocar_camera"] else None,
            ),
            # Last, once nothing uses it: HailoRT crashes the process at exit if
            # its session is left for the interpreter to tear down.
            ("detector", extensions["robocar_detector"]),
        ]
        for name, part in steps:
            if part is None:
                continue
            logger.info("Closing the %s", name)
            try:
                part.close()
            except Exception:
                logger.exception("Closing the %s failed", name)
        watchdog.cancel()
        logger.info("Shut down")

    def stop_on_signal(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    atexit.register(close_resources)
    signal.signal(signal.SIGTERM, stop_on_signal)
    app.run(host=config.host, port=config.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
