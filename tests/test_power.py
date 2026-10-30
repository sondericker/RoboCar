import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robocar.config import Config
from robocar.power import PiPower, PowerMonitor, board_version, charge_percent, parse_pmic, parse_throttled
from robocar.web import create_app

PMIC_SAMPLE = """\
 3V3_SYS_A current(1)=0.07124290A
  VDD_CORE_A current(7)=1.24127000A
   3V3_SYS_V volt(9)=3.29797000V
  VDD_CORE_V volt(15)=0.87113460V
     EXT5V_V volt(24)=5.16436000V
      BATT_V volt(25)=0.00000000V
"""


class FakeADC:
    def __init__(self, volts=8.0):
        self.volts = volts
        self.fail = False
        self.closed = False

    def battery_volts(self):
        if self.fail:
            raise OSError(121, "Remote I/O error")
        return self.volts

    def close(self):
        self.closed = True


class ReadingTests(unittest.TestCase):
    def test_charge_follows_the_lithium_discharge_curve(self):
        self.assertEqual(charge_percent(8.4, 2), 100.0)
        self.assertEqual(charge_percent(6.0, 2), 0.0)
        self.assertAlmostEqual(charge_percent(7.4, 2), 35.0)
        self.assertAlmostEqual(charge_percent(7.0, 2), 10.0)
        self.assertAlmostEqual(charge_percent(7.5, 2), 42.5)

    def test_pi_power_chip_output_is_parsed(self):
        rails = parse_pmic(PMIC_SAMPLE)
        self.assertAlmostEqual(rails["EXT5V"]["volts"], 5.16436)
        self.assertAlmostEqual(rails["VDD_CORE"]["amps"], 1.24127)
        flags = parse_throttled("throttled=0x50005\n")
        self.assertTrue(flags["under_voltage_now"] and flags["throttled_now"] and flags["under_voltage_since_boot"])
        self.assertFalse(flags["temperature_limit_now"])
        self.assertFalse(any(parse_throttled("throttled=0x0").values()))

    def test_pi_readings_survive_a_missing_vcgencmd(self):
        def missing(*args):
            raise FileNotFoundError("vcgencmd")

        reading = PiPower(run=missing, thermal=Path("/no/such/file")).read()
        self.assertIn("rails_error", reading)
        self.assertIn("flags_error", reading)
        good = PiPower(run=lambda what: PMIC_SAMPLE if what == "pmic_read_adc" else "throttled=0x0").read()
        self.assertEqual(good["rails"]["EXT5V"], 5.164)
        self.assertAlmostEqual(good["board_watts"], 3.297970 * 0.0712429 + 0.8711346 * 1.24127, places=2)

    def test_board_version_comes_from_the_kit_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(board_version(Path(directory)), 2)
            (Path(directory) / "params.json").write_text(json.dumps({"Pcb_Version": 1}))
            self.assertEqual(board_version(Path(directory)), 1)
        self.assertEqual(board_version(None), 2)


class MonitorTests(unittest.TestCase):
    def monitor(self, adc, moving):
        monitor = PowerMonitor(adc, None, lambda: moving[0], interval=3600)
        self.addCleanup(monitor.close)
        return monitor

    def test_levels_warn_as_the_battery_runs_down(self):
        adc, moving = FakeADC(), [False]
        monitor = self.monitor(adc, moving)
        with patch("robocar.power.RESTING_AFTER_SECONDS", 0.0):
            for volts, level in ((8.0, "ok"), (7.2, "getting_low"), (6.9, "low"), (6.5, "critical")):
                adc.volts = volts
                monitor.sample()
                self.assertEqual(monitor.battery_level(), level, volts)
        status = monitor.status()["battery"]
        self.assertEqual(status["resting_volts"], 6.5)
        self.assertLess(status["low_percent"], status["getting_low_percent"])

    def test_readings_while_driving_do_not_count_as_resting(self):
        adc, moving = FakeADC(8.0), [False]
        monitor = self.monitor(adc, moving)
        with patch("robocar.power.RESTING_AFTER_SECONDS", 0.0):
            monitor.sample()
            moving[0] = True
            adc.volts = 6.4  # the motors pull it down
            monitor.sample()
        status = monitor.status()["battery"]
        self.assertEqual(status["resting_volts"], 8.0)
        self.assertEqual(status["volts"], 6.4)
        self.assertTrue(status["under_load"])
        self.assertEqual(monitor.battery_level(), "ok")

    def test_a_failing_converter_is_reported_not_fatal(self):
        adc, moving = FakeADC(), [False]
        adc.fail = True
        monitor = self.monitor(adc, moving)
        monitor.sample()
        self.assertIn("Remote I/O error", monitor.status()["battery"]["error"])
        self.assertIsNone(monitor.battery_level())


class WebPowerTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Config(mapping_database=Path(":memory:")))
        self.client = self.app.test_client()

    def tearDown(self):
        for name in ("robocar_mapping", "robocar_drive", "robocar_pan_tilt", "robocar_range_sensor", "robocar_camera", "robocar_power"):
            self.app.extensions[name].close()

    def test_power_is_in_the_status_and_on_the_dashboard(self):
        status = self.client.get("/api/status").get_json()
        self.assertIn("battery", status["power"])
        self.assertFalse(status["power"]["battery"]["available"])  # hardware not enabled
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="battery-meter"', page)
        self.assertIn('id="battery-meter-small"', page)

    def test_a_critical_battery_stops_autonomous_runs_starting(self):
        power = self.app.extensions["robocar_power"]
        with (
            patch.object(power, "battery_level", return_value="critical"),
            patch.object(power, "status", return_value={"battery": {"resting_volts": 6.4}}),
        ):
            for url in ("/api/mapping/explore", "/api/calibration"):
                response = self.client.post(url, json={"enabled": True, "attended": True})
                self.assertEqual(response.status_code, 409)
                self.assertIn("Battery critically low", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
