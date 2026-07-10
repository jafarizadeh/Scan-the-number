from __future__ import annotations

import glob
import threading
import time
from typing import Any


class HardwareActions:
    """
    Hardware action driver.

    Preferred mode for this project:
      hardwareMode = "pico_serial"

    Raspberry Pi sends serial commands to a Pico.
    Pico controls:
      GP14 -> active buzzer KY-012
      GP15 -> PWM signal for Tapper / Actuonix
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.lock = threading.RLock()
        self.config: dict[str, Any] = {}
        self.running = False
        self.lastActionTime = 0.0
        self.lastAction: dict[str, Any] | None = None
        self.lastError: str | None = None
        self.thread: threading.Thread | None = None

        self.serial_obj = None
        self.connectedPort: str | None = None
        self.lastSerialLines: list[str] = []

        self.configure(config or {})

    def configure(self, config: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            merged = dict(self.config)
            merged.update(config or {})

            self.config = {
                "dryRun": bool(merged.get("dryRun", True)),

                "hardwareMode": str(merged.get("hardwareMode", "pico_serial")),
                "picoPort": str(merged.get("picoPort", "auto")),
                "picoBaud": int(merged.get("picoBaud", 115200)),

                "buzzerEnabled": bool(merged.get("buzzerEnabled", False)),
                "buzzerPin": int(merged.get("buzzerPin", 14)),
                "buzzerDurationMs": int(merged.get("buzzerDurationMs", 350)),
                "buzzerDelayMs": int(merged.get("buzzerDelayMs", 0)),

                "actuatorEnabled": bool(merged.get("actuatorEnabled", False)),
                "actuatorPin": int(merged.get("actuatorPin", 15)),
                "actuatorDelayMs": int(merged.get("actuatorDelayMs", 0)),
                "actuatorOutPercent": int(merged.get("actuatorOutPercent", 35)),
                "actuatorHoldMs": int(merged.get("actuatorHoldMs", 1000)),
                "actuatorReturnMs": int(merged.get("actuatorReturnMs", 250)),

                "servoMinPulseMs": float(merged.get("servoMinPulseMs", 1.0)),
                "servoMaxPulseMs": float(merged.get("servoMaxPulseMs", 2.4)),

                "actionCooldownMs": int(merged.get("actionCooldownMs", 1200)),
                "allowOverlap": bool(merged.get("allowOverlap", False)),
            }

            return dict(self.config)

    set_config = configure

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "lastActionTime": self.lastActionTime,
                "lastAction": self.lastAction,
                "lastError": self.lastError,
                "dryRun": self.config.get("dryRun", True),
                "hardwareMode": self.config.get("hardwareMode", "pico_serial"),
                "picoPort": self.config.get("picoPort", "auto"),
                "connectedPort": self.connectedPort,
                "serialOpen": bool(self.serial_obj and getattr(self.serial_obj, "is_open", False)),
                "lastSerialLines": list(self.lastSerialLines[-8:]),
            }

    def trigger(self, hardware: dict[str, Any] | None = None, reason: str = "trigger", number: int | None = None) -> dict[str, Any]:
        if hardware:
            self.configure(hardware)

        with self.lock:
            now = time.time()
            cooldown_s = max(0.0, self.config.get("actionCooldownMs", 1200) / 1000.0)

            if self.running and not self.config.get("allowOverlap", False):
                return {
                    "accepted": False,
                    "reason": "busy",
                    "dryRun": self.config.get("dryRun", True),
                    "lastError": self.lastError,
                }

            if self.lastActionTime and (now - self.lastActionTime) < cooldown_s:
                return {
                    "accepted": False,
                    "reason": "cooldown",
                    "dryRun": self.config.get("dryRun", True),
                    "lastError": self.lastError,
                }

            self.running = True
            self.lastError = None
            self.lastActionTime = now
            self.lastAction = {
                "reason": reason,
                "number": number,
                "startedAt": now,
                "config": dict(self.config),
            }

            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

            return {
                "accepted": True,
                "reason": "started",
                "dryRun": self.config.get("dryRun", True),
            }

    # Compatibility aliases for different backend call styles
    run = trigger
    fire = trigger
    start = trigger
    request = trigger
    test_action = trigger

    def _worker(self):
        try:
            cfg = dict(self.config)

            if cfg.get("dryRun", True):
                time.sleep(0.25)
                self._set_last_result(None, ["DRY RUN"])
                return

            mode = cfg.get("hardwareMode", "pico_serial")

            if mode == "pico_serial":
                lines = self._send_pico_trigger(cfg)
                self._set_last_result(None, lines)
                return

            self._set_last_result(f"Unsupported hardwareMode: {mode}", [])

        except Exception as exc:
            self._set_last_result(str(exc), [])

        finally:
            with self.lock:
                self.running = False

    def _set_last_result(self, error: str | None, serial_lines: list[str]):
        with self.lock:
            self.lastError = error
            self.lastSerialLines = serial_lines[-20:]

            if self.lastAction is not None:
                self.lastAction["finishedAt"] = time.time()
                self.lastAction["ok"] = error is None
                self.lastAction["serialLines"] = serial_lines[-8:]
                self.lastAction["error"] = error

    def _find_serial_port(self, port_setting: str) -> str | None:
        if port_setting and port_setting != "auto":
            return port_setting

        try:
            import serial.tools.list_ports

            ports = list(serial.tools.list_ports.comports())

            preferred = []
            fallback = []

            for p in ports:
                dev = p.device
                label = " ".join([
                    str(getattr(p, "description", "")),
                    str(getattr(p, "manufacturer", "")),
                    str(getattr(p, "product", "")),
                ]).lower()

                if any(k in label for k in ["pico", "rp2", "micropython", "raspberry pi"]):
                    preferred.append(dev)
                elif dev.startswith("/dev/ttyACM") or dev.startswith("/dev/ttyUSB"):
                    fallback.append(dev)

            if preferred:
                return preferred[0]
            if fallback:
                return fallback[0]

        except Exception:
            pass

        candidates = []
        candidates.extend(glob.glob("/dev/ttyACM*"))
        candidates.extend(glob.glob("/dev/ttyUSB*"))

        return candidates[0] if candidates else None

    def _connect_serial(self, cfg: dict[str, Any]):
        try:
            import serial
        except Exception as exc:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial") from exc

        if self.serial_obj and getattr(self.serial_obj, "is_open", False):
            return self.serial_obj

        port = self._find_serial_port(str(cfg.get("picoPort", "auto")))
        if not port:
            raise RuntimeError("Pico serial port not found. Connect Pico via USB or set picoPort.")

        baud = int(cfg.get("picoBaud", 115200))

        self.serial_obj = serial.Serial(port, baud, timeout=0.25, write_timeout=1)
        self.connectedPort = port

        # Pico may reset when serial opens.
        time.sleep(1.6)

        return self.serial_obj

    def _send_pico_trigger(self, cfg: dict[str, Any]) -> list[str]:
        ser = self._connect_serial(cfg)

        sound_delay = max(0.0, int(cfg.get("buzzerDelayMs", 0)) / 1000.0)
        tapper_delay = max(0.0, int(cfg.get("actuatorDelayMs", 0)) / 1000.0)

        sound_enabled = 1 if cfg.get("buzzerEnabled", False) else 0
        tapper_enabled = 1 if cfg.get("actuatorEnabled", False) else 0

        tapper_percent = max(0, min(100, int(cfg.get("actuatorOutPercent", 35))))
        buzzer_duration = max(50, int(cfg.get("buzzerDurationMs", 350)))
        hold_ms = max(100, int(cfg.get("actuatorHoldMs", 1000)))
        return_ms = max(100, int(cfg.get("actuatorReturnMs", 250)))

        # Compatible with old Pico code:
        # TRIGGER sound_delay arm_delay sound_enabled arm_enabled
        # New Pico code also accepts extra params:
        # tapper_percent buzzer_duration_ms hold_ms return_ms
        cmd = (
            "TRIGGER "
            f"{sound_delay:.3f} "
            f"{tapper_delay:.3f} "
            f"{sound_enabled} "
            f"{tapper_enabled} "
            f"{tapper_percent} "
            f"{buzzer_duration} "
            f"{hold_ms} "
            f"{return_ms}"
        )

        lines: list[str] = []

        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        ser.write((cmd + "\n").encode("utf-8"))
        ser.flush()

        deadline = time.time() + 5.0

        while time.time() < deadline:
            try:
                raw = ser.readline()
            except Exception as exc:
                raise RuntimeError(f"Serial read failed: {exc}") from exc

            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            lines.append(line)

            if "TRIGGER DONE" in line:
                return lines

            if "Command error" in line:
                raise RuntimeError(line)

        if not lines:
            raise RuntimeError(
                f"No response from Pico after command: {cmd}. "
                "Check Pico firmware/main.py and USB serial connection."
            )

        # If Pico printed something but did not finish, keep it as an error.
        raise RuntimeError("Pico did not report TRIGGER DONE. Last lines: " + " | ".join(lines[-5:]))


# ---------------------------------------------------------------------
# Compatibility layer for web_final_app.py
# web_final_app imports:
#   HardwareActionController, DEFAULT_HARDWARE_CONFIG, clean_hardware_config
# ---------------------------------------------------------------------

DEFAULT_HARDWARE_CONFIG = {
    "dryRun": True,

    "hardwareMode": "pico_serial",
    "picoPort": "auto",
    "picoBaud": 115200,

    "buzzerEnabled": False,
    "buzzerPin": 14,
    "buzzerDurationMs": 350,
    "buzzerDelayMs": 0,

    "actuatorEnabled": False,
    "actuatorPin": 15,
    "actuatorDelayMs": 0,
    "actuatorOutPercent": 35,
    "actuatorHoldMs": 1000,
    "actuatorReturnMs": 250,

    "servoMinPulseMs": 1.0,
    "servoMaxPulseMs": 2.4,

    "actionCooldownMs": 1200,
    "allowOverlap": False,
}


def clean_hardware_config(config=None):
    config = config or {}
    cleaned = dict(DEFAULT_HARDWARE_CONFIG)
    cleaned.update(config)

    bool_keys = [
        "dryRun",
        "buzzerEnabled",
        "actuatorEnabled",
        "allowOverlap",
    ]

    int_keys = [
        "picoBaud",
        "buzzerPin",
        "buzzerDurationMs",
        "buzzerDelayMs",
        "actuatorPin",
        "actuatorDelayMs",
        "actuatorOutPercent",
        "actuatorHoldMs",
        "actuatorReturnMs",
        "actionCooldownMs",
    ]

    float_keys = [
        "servoMinPulseMs",
        "servoMaxPulseMs",
    ]

    for key in bool_keys:
        cleaned[key] = bool(cleaned.get(key, DEFAULT_HARDWARE_CONFIG[key]))

    for key in int_keys:
        try:
            cleaned[key] = int(cleaned.get(key, DEFAULT_HARDWARE_CONFIG[key]))
        except Exception:
            cleaned[key] = int(DEFAULT_HARDWARE_CONFIG[key])

    for key in float_keys:
        try:
            cleaned[key] = float(cleaned.get(key, DEFAULT_HARDWARE_CONFIG[key]))
        except Exception:
            cleaned[key] = float(DEFAULT_HARDWARE_CONFIG[key])

    cleaned["hardwareMode"] = str(cleaned.get("hardwareMode", "pico_serial"))
    cleaned["picoPort"] = str(cleaned.get("picoPort", "auto"))

    cleaned["actuatorOutPercent"] = max(0, min(100, cleaned["actuatorOutPercent"]))
    cleaned["buzzerDurationMs"] = max(50, cleaned["buzzerDurationMs"])
    cleaned["actuatorHoldMs"] = max(100, cleaned["actuatorHoldMs"])
    cleaned["actuatorReturnMs"] = max(100, cleaned["actuatorReturnMs"])
    cleaned["actionCooldownMs"] = max(0, cleaned["actionCooldownMs"])

    return cleaned


class HardwareActionController(HardwareActions):
    def __init__(self, config=None):
        super().__init__(clean_hardware_config(config or {}))

    def configure(self, config):
        return super().configure(clean_hardware_config(config or {}))

    def set_config(self, config):
        return self.configure(config)

    def update_config(self, config):
        return self.configure(config)

    def test(self):
        return self.trigger(reason="test-action")

    def test_action(self):
        return self.trigger(reason="test-action")

    def trigger_async(self, *args, **kwargs):
        """
        Compatibility method expected by web_final_app.py.

        It starts the action in a background thread through self.trigger().
        Accepts flexible arguments because older UI/backend versions may call it
        with reason, number, or config/hardware.
        """
        hardware = kwargs.pop("hardware", None)
        config = kwargs.pop("config", None)
        reason = kwargs.pop("reason", "trigger")
        number = kwargs.pop("number", None)

        if config is not None and hardware is None:
            hardware = config

        # Flexible positional compatibility
        for arg in args:
            if isinstance(arg, dict):
                hardware = arg
            elif isinstance(arg, int):
                number = arg
            elif isinstance(arg, str):
                reason = arg

        return self.trigger(hardware=hardware, reason=reason, number=number)

    def trigger_for_match(self, number=None, hardware=None, reason="match"):
        return self.trigger(hardware=hardware, reason=reason, number=number)

    def trigger_match(self, number=None, hardware=None):
        return self.trigger(hardware=hardware, reason="match", number=number)

