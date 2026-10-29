import atexit
import logging
import signal

from dotenv import load_dotenv

from robocar.config import Config
from robocar.web import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    config = Config.from_env()
    app = create_app(config)

    def close_resources() -> None:
        app.extensions["robocar_mapping"].close()
        app.extensions["robocar_drive"].close()
        app.extensions["robocar_pan_tilt"].close()
        app.extensions["robocar_range_sensor"].close()
        app.extensions["robocar_camera"].close()
        # Last, once nothing uses it: HailoRT crashes the process at exit if
        # its session is left for the interpreter to tear down.
        app.extensions["robocar_detector"].close()

    def stop_on_signal(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    atexit.register(close_resources)
    signal.signal(signal.SIGTERM, stop_on_signal)
    app.run(host=config.host, port=config.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
