from __future__ import annotations

import glob
import json
import time
from pathlib import Path

try:
    from flask import request, jsonify, send_file
except Exception as exc:
    request = None
    jsonify = None
    send_file = None
    FLASK_IMPORT_ERROR = exc
else:
    FLASK_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "data" / "config.json"
UI_DIR = PROJECT_ROOT / "data" / "ui"


def _clamp(value, low, high):
    try:
        value = int(value)
    except Exception:
        value = low
    return max(low, min(high, value))


def _read_config():
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _find_serial_port(cfg):
    hw = cfg.get("hardware", {})
    configured = hw.get("picoPort") or hw.get("serialPort")

    if configured and configured != "auto" and Path(configured).exists():
        return configured

    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        raise RuntimeError("No Pico serial port found")

    return ports[0]


def _send_pico(command, wait_seconds=1.0):
    import serial

    cfg = _read_config()
    hw = cfg.get("hardware", {})
    port = _find_serial_port(cfg)
    baud = int(hw.get("picoBaud", 115200))

    lines = []

    with serial.Serial(port, baud, timeout=0.25, write_timeout=1) as ser:
        ser.write((command.strip() + "\n").encode("utf-8"))
        ser.flush()

        end = time.time() + wait_seconds
        while time.time() < end:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                lines.append(line)

    return {
        "ok": True,
        "port": port,
        "command": command,
        "lines": lines,
    }


def _json(data, status=200):
    if jsonify is None:
        return data, status
    response = jsonify(data)
    response.status_code = status
    return response


def _payload():
    if request is None:
        return {}
    return request.get_json(silent=True) or {}


def install_calibration_api(app):
    if FLASK_IMPORT_ERROR is not None:
        raise RuntimeError(f"Flask import failed: {FLASK_IMPORT_ERROR}")

    if getattr(app, "_roulette_calibration_api_installed", False):
        return

    setattr(app, "_roulette_calibration_api_installed", True)

    @app.route("/calibrate", methods=["GET"])
    @app.route("/calibrate.html", methods=["GET"])
    def calibration_page():
        page = UI_DIR / "calibrate.html"
        if not page.exists():
            return _json({"ok": False, "error": "calibrate.html not found"}, 404)
        return send_file(str(page))

    @app.route("/api/calibration/status", methods=["GET"])
    def calibration_status():
        cfg = _read_config()
        hw = cfg.setdefault("hardware", {})

        up_us = int(hw.get("servoUpUs", 1000))
        tap_us = int(hw.get("servoTapUs", 1200))
        hold_ms = int(hw.get("servoHoldMs", 200))

        try:
            pico_status = _send_pico("GETCAL", 1.0)
        except Exception as exc:
            pico_status = {
                "ok": False,
                "error": str(exc),
            }

        return _json({
            "ok": True,
            "config": {
                "upUs": up_us,
                "tapUs": tap_us,
                "holdMs": hold_ms,
            },
            "hardware": hw,
            "pico": pico_status,
        })

    @app.route("/api/calibration/servo", methods=["POST"])
    def calibration_servo():
        data = _payload()

        command = str(data.get("command", "")).lower().strip()

        up_us = _clamp(data.get("upUs", 1000), 800, 1400)
        tap_us = _clamp(data.get("tapUs", data.get("positionUs", 1200)), 900, 1600)
        position_us = _clamp(data.get("positionUs", tap_us), 800, 1600)
        hold_ms = _clamp(data.get("holdMs", 200), 50, 700)
        buzz_ms = _clamp(data.get("buzzMs", 300), 50, 1500)

        if command == "up":
            pico_cmd = f"SERVO {up_us}"
            wait = 0.8

        elif command == "move":
            pico_cmd = f"SERVO {position_us}"
            wait = 0.8

        elif command == "tap":
            pico_cmd = f"TAP {tap_us} {hold_ms}"
            wait = 1.5

        elif command == "off":
            pico_cmd = "OFF"
            wait = 0.8

        elif command == "buzz":
            pico_cmd = f"BUZZ {buzz_ms}"
            wait = 0.8

        elif command == "getcal":
            pico_cmd = "GETCAL"
            wait = 1.0

        elif command == "setcal":
            pico_cmd = f"SETCAL {up_us} {tap_us} {hold_ms}"
            wait = 1.0

        else:
            return _json({"ok": False, "error": f"Unknown command: {command}"}, 400)

        try:
            result = _send_pico(pico_cmd, wait)
            return _json(result)
        except Exception as exc:
            return _json({
                "ok": False,
                "command": pico_cmd,
                "error": str(exc),
            }, 500)

    @app.route("/api/calibration/save", methods=["POST"])
    def calibration_save():
        data = _payload()

        up_us = _clamp(data.get("upUs", 1000), 800, 1400)
        tap_us = _clamp(data.get("tapUs", 1200), 900, 1600)
        hold_ms = _clamp(data.get("holdMs", 200), 50, 700)

        cfg = _read_config()
        hw = cfg.setdefault("hardware", {})

        hw["dryRun"] = False
        hw["hardwareMode"] = "pico_serial"
        hw["picoPort"] = hw.get("picoPort", "auto")
        hw["picoBaud"] = int(hw.get("picoBaud", 115200))

        hw["buzzerEnabled"] = True
        hw["buzzerPin"] = 0

        hw["actuatorEnabled"] = True
        hw["actuatorPin"] = 2

        hw["servoUpUs"] = up_us
        hw["servoTapUs"] = tap_us
        hw["servoHoldMs"] = hold_ms

        _write_config(cfg)

        try:
            pico_result = _send_pico(f"SETCAL {up_us} {tap_us} {hold_ms}", 1.2)
        except Exception as exc:
            pico_result = {
                "ok": False,
                "error": str(exc),
            }

        return _json({
            "ok": True,
            "saved": {
                "upUs": up_us,
                "tapUs": tap_us,
                "holdMs": hold_ms,
            },
            "pico": pico_result,
        })
