import json
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_HARDWARE_CONFIG = {
    "dryRun": True,
    "buzzerEnabled": True,
    "buzzerPin": 17,
    "buzzerDurationMs": 350,
    "buzzerDelayMs": 0,
    "actuatorEnabled": True,
    "actuatorPin": 18,
    "actuatorDelayMs": 0,
    "actuatorOutPercent": 35,
    "actuatorHoldMs": 350,
    "actuatorReturnMs": 350,
    "servoMinPulseMs": 1.0,
    "servoMaxPulseMs": 2.0,
    "actionCooldownMs": 1200,
    "allowOverlap": False,
}


def clean_hardware_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    cfg = dict(DEFAULT_HARDWARE_CONFIG)
    cfg.update(payload)

    def b(key: str) -> bool:
        return bool(cfg.get(key))

    def i(key: str, lo: int, hi: int) -> int:
        try:
            value = int(cfg.get(key, DEFAULT_HARDWARE_CONFIG[key]))
        except Exception:
            value = int(DEFAULT_HARDWARE_CONFIG[key])
        return max(lo, min(hi, value))

    def f(key: str, lo: float, hi: float) -> float:
        try:
            value = float(cfg.get(key, DEFAULT_HARDWARE_CONFIG[key]))
        except Exception:
            value = float(DEFAULT_HARDWARE_CONFIG[key])
        return max(lo, min(hi, value))

    return {
        "dryRun": b("dryRun"),
        "buzzerEnabled": b("buzzerEnabled"),
        "buzzerPin": i("buzzerPin", 2, 27),
        "buzzerDurationMs": i("buzzerDurationMs", 50, 5000),
        "buzzerDelayMs": i("buzzerDelayMs", 0, 10000),
        "actuatorEnabled": b("actuatorEnabled"),
        "actuatorPin": i("actuatorPin", 2, 27),
        "actuatorDelayMs": i("actuatorDelayMs", 0, 10000),
        "actuatorOutPercent": i("actuatorOutPercent", 0, 100),
        "actuatorHoldMs": i("actuatorHoldMs", 50, 5000),
        "actuatorReturnMs": i("actuatorReturnMs", 50, 5000),
        "servoMinPulseMs": f("servoMinPulseMs", 0.5, 2.5),
        "servoMaxPulseMs": f("servoMaxPulseMs", 1.5, 3.0),
        "actionCooldownMs": i("actionCooldownMs", 0, 30000),
        "allowOverlap": b("allowOverlap"),
    }


class HardwareActionController:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = clean_hardware_config(config)
        self.lock = threading.Lock()
        self.running = False
        self.last_action_time = 0.0
        self.last_action: dict[str, Any] | None = None
        self.last_error: str | None = None
        self._buzzer = None
        self._servo = None
        self._device_signature = None

    def close(self):
        with self.lock:
            for device in (self._buzzer, self._servo):
                try:
                    if device is not None:
                        device.close()
                except Exception:
                    pass
            self._buzzer = None
            self._servo = None
            self._device_signature = None

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        cleaned = clean_hardware_config(config)
        with self.lock:
            changed = cleaned != self.config
            self.config = cleaned
        if changed:
            self.close()
        return self.get_config()

    def get_config(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.config)

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "lastActionTime": self.last_action_time,
                "lastAction": dict(self.last_action) if self.last_action else None,
                "lastError": self.last_error,
                "dryRun": bool(self.config.get("dryRun", True)),
            }

    def _ensure_devices(self):
        cfg = self.config
        signature = (
            cfg["dryRun"], cfg["buzzerPin"], cfg["actuatorPin"],
            cfg["servoMinPulseMs"], cfg["servoMaxPulseMs"],
        )
        if signature == self._device_signature:
            return

        self.close()
        self._device_signature = signature

        if cfg["dryRun"]:
            return

        try:
            from gpiozero import DigitalOutputDevice, Servo
        except Exception as exc:
            self.last_error = f"gpiozero unavailable: {exc}"
            raise

        self._buzzer = DigitalOutputDevice(cfg["buzzerPin"], active_high=True, initial_value=False)
        self._servo = Servo(
            cfg["actuatorPin"],
            min_pulse_width=cfg["servoMinPulseMs"] / 1000.0,
            max_pulse_width=cfg["servoMaxPulseMs"] / 1000.0,
            frame_width=0.020,
            initial_value=-1,
        )

    @staticmethod
    def _percent_to_servo_value(percent: int) -> float:
        percent = max(0, min(100, int(percent)))
        return -1.0 + (2.0 * percent / 100.0)

    def _run_buzzer(self, cfg: dict[str, Any]):
        time.sleep(cfg["buzzerDelayMs"] / 1000.0)
        if cfg["dryRun"] or not cfg["buzzerEnabled"]:
            print("[HARDWARE] dry buzzer")
            return
        self._ensure_devices()
        self._buzzer.on()
        time.sleep(cfg["buzzerDurationMs"] / 1000.0)
        self._buzzer.off()

    def _run_actuator(self, cfg: dict[str, Any]):
        time.sleep(cfg["actuatorDelayMs"] / 1000.0)
        if cfg["dryRun"] or not cfg["actuatorEnabled"]:
            print("[HARDWARE] dry actuator")
            return
        self._ensure_devices()
        out_value = self._percent_to_servo_value(cfg["actuatorOutPercent"])
        self._servo.value = out_value
        time.sleep(cfg["actuatorHoldMs"] / 1000.0)
        self._servo.value = -1
        time.sleep(cfg["actuatorReturnMs"] / 1000.0)

    def trigger_async(self, *, number: int | None, list_name: str = "", reason: str = "match") -> dict[str, Any]:
        cfg = self.get_config()
        now = time.monotonic()
        with self.lock:
            elapsed_ms = (now - self.last_action_time) * 1000.0
            if self.running and not cfg["allowOverlap"]:
                return {"accepted": False, "reason": "action_already_running"}
            if elapsed_ms < cfg["actionCooldownMs"]:
                return {"accepted": False, "reason": "action_cooldown"}
            self.running = True

        def worker():
            action = {
                "number": number,
                "listName": list_name,
                "reason": reason,
                "startedAt": time.time(),
                "ok": False,
            }
            try:
                threads = []
                if cfg["buzzerEnabled"]:
                    threads.append(threading.Thread(target=self._run_buzzer, args=(cfg,), daemon=True))
                if cfg["actuatorEnabled"]:
                    threads.append(threading.Thread(target=self._run_actuator, args=(cfg,), daemon=True))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                action["ok"] = True
                self.last_error = None
            except Exception as exc:
                action["ok"] = False
                action["error"] = str(exc)
                self.last_error = str(exc)
                print(f"[HARDWARE] trigger failed: {exc}")
            finally:
                action["finishedAt"] = time.time()
                with self.lock:
                    self.running = False
                    self.last_action_time = time.monotonic()
                    self.last_action = action

        threading.Thread(target=worker, daemon=True).start()
        return {"accepted": True, "reason": "started", "dryRun": cfg["dryRun"]}
