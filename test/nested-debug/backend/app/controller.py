import threading
import time
import json
from pathlib import Path
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import cv2

from .camera import Camera
from .ocr import NumberOCR
from .pico_serial import PicoSerial
from .settings import settings
from .stability import StableNumberDetector


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


ROBOT_SETTINGS_FILE = Path("runtime/robot-settings.json")

DEFAULT_ROBOT_SETTINGS = {
    "cameraEnabled": True,
    "soundEnabled": True,
    "armEnabled": True,
    "soundDelaySeconds": 0.0,
    "armDelaySeconds": 0.0,
    "armExtensionPercent": 100,
}


class RouletteController:
    def __init__(self):
        self.camera = Camera(
            settings.camera_source,
            settings.camera_width,
            settings.camera_height,
            settings.camera_fps,
        )
        self.ocr = NumberOCR(settings.debug_save_frames)
        self.pico = PicoSerial(settings.pico_port, settings.pico_baud)
        self.stability = StableNumberDetector(settings.stable_frames)

        self.target_number = settings.target_number
        self.stable_frames = settings.stable_frames
        self.roi = (settings.roi_x, settings.roi_y, settings.roi_w, settings.roi_h)
        self.cooldown = settings.match_cooldown_seconds
        self.pico_command = settings.pico_command
        self.robot_settings = self._load_robot_settings()

        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

        self.session: dict[str, Any] | None = None
        self.readings: deque[dict[str, Any]] = deque(maxlen=2000)
        self.latest_reading: dict[str, Any] | None = None
        self._last_recorded_number: int | None = None
        self._last_recorded_time = 0.0

        self.last_raw_text = ""
        self.last_detected_number: int | None = None
        self.last_stable_number: int | None = None
        self.last_trigger_time = 0.0
        self.trigger_count = 0
        self.last_trigger_ok: bool | None = None
        self.last_trigger_error: str | None = None
        self.last_frame_jpeg: bytes | None = None

        # Prevent repeated triggers while the same stable target remains on screen.
        self.armed = True

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return True
            self.running = True
            self.stability.reset()
            self._last_recorded_number = None
            self._last_recorded_time = 0.0

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        with self.lock:
            self.running = False

        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None

        self.camera.close()

    def start_session(
        self,
        mode: str,
        target_number: int,
        camera_enabled: bool,
        sound_enabled: bool,
    ) -> dict[str, Any]:
        self.set_target_number(target_number)
        session = {
            "id": f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "mode": mode,
            "targetNumber": target_number,
            "status": "monitoring",
            "actionEnabled": True,
            "startedAt": utc_now_iso(),
            "cameraEnabled": camera_enabled,
            "soundEnabled": sound_enabled,
        }
        with self.lock:
            self.session = session
            self.latest_reading = None
            self._last_recorded_number = None
            self._last_recorded_time = 0.0
            self.armed = True
            self.stability.reset()

        if mode == "real" and camera_enabled:
            self.start()

        return self.public_session()

    def stop_session(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            if self.session and self.session.get("id") == session_id:
                self.session["status"] = "stopped"
                self.session["stoppedAt"] = utc_now_iso()
                stopped = dict(self.session)
            else:
                stopped = {
                    "id": session_id,
                    "mode": "real",
                    "targetNumber": self.target_number,
                    "status": "stopped",
                    "actionEnabled": False,
                    "startedAt": utc_now_iso(),
                    "stoppedAt": utc_now_iso(),
                }

        self.stop()
        return self._public_session_from_dict(stopped)

    def public_session(self) -> dict[str, Any]:
        with self.lock:
            session = dict(self.session) if self.session else {
                "id": "no-session",
                "mode": "real",
                "targetNumber": self.target_number,
                "status": "ready",
                "actionEnabled": True,
                "startedAt": utc_now_iso(),
            }
        return self._public_session_from_dict(session)

    def _public_session_from_dict(self, session: dict[str, Any]) -> dict[str, Any]:
        result = {
            "id": session["id"],
            "mode": session.get("mode", "real"),
            "targetNumber": session.get("targetNumber", self.target_number),
            "status": session.get("status", "ready"),
            "actionEnabled": session.get("actionEnabled", True),
            "startedAt": session.get("startedAt", utc_now_iso()),
        }
        if session.get("stoppedAt"):
            result["stoppedAt"] = session["stoppedAt"]
        return result

    def set_target_number(self, number: int) -> None:
        if not 0 <= number <= 36:
            raise ValueError("target_number must be between 0 and 36")
        with self.lock:
            self.target_number = number
            if self.session:
                self.session["targetNumber"] = number
            self.armed = True
            self.stability.reset()
            self._last_recorded_number = None

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        with self.lock:
            self.roi = (x, y, w, h)

    def _clean_robot_settings(self, payload: dict) -> dict:
        return {
            "cameraEnabled": bool(payload.get("cameraEnabled", True)),
            "soundEnabled": bool(payload.get("soundEnabled", True)),
            "armEnabled": bool(payload.get("armEnabled", True)),
            "soundDelaySeconds": max(0.0, min(10.0, float(payload.get("soundDelaySeconds", 0.0)))),
            "armDelaySeconds": max(0.0, min(10.0, float(payload.get("armDelaySeconds", 0.0)))),
        }

    def _load_robot_settings(self) -> dict:
        try:
            if ROBOT_SETTINGS_FILE.exists():
                raw = json.loads(ROBOT_SETTINGS_FILE.read_text())
                return self._clean_robot_settings(raw)
        except Exception as exc:
            print(f"[SETTINGS] could not load robot settings: {exc}")

        return dict(DEFAULT_ROBOT_SETTINGS)

    def _save_robot_settings(self) -> None:
        ROBOT_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ROBOT_SETTINGS_FILE.write_text(json.dumps(self.robot_settings, indent=2))

    def get_robot_settings(self) -> dict:
        with self.lock:
            return dict(self.robot_settings)

    def set_robot_settings(self, payload: dict) -> dict:
        cleaned = self._clean_robot_settings(payload)

        with self.lock:
            self.robot_settings = cleaned
            self._save_robot_settings()

        print(f"[SETTINGS] saved robot settings: {cleaned}")
        return cleaned


    def _build_pico_trigger_command(self) -> str:
        with self.lock:
            cfg = dict(self.robot_settings)

        return (
            "TRIGGER "
            f"{cfg['soundDelaySeconds']} "
            f"{cfg['armDelaySeconds']} "
            f"{1 if cfg['soundEnabled'] else 0} "
            f"{1 if cfg['armEnabled'] else 0}"
        )

    def send_pico_test(self) -> bool:
        return self.pico.send(self._build_pico_trigger_command())

    def manual_trigger(self) -> bool:
        return self._send_trigger(reason="manual")

    def _send_trigger(self, reason: str) -> bool:
        ok = self.pico.send(self._build_pico_trigger_command())
        with self.lock:
            self.last_trigger_ok = ok
            self.last_trigger_time = time.time()
            if ok:
                self.trigger_count += 1
                self.last_trigger_error = None
            else:
                self.last_trigger_error = self.pico.last_error
        print(f"[TRIGGER] reason={reason} ok={ok}")
        return ok

    def _loop(self) -> None:
        while True:
            with self.lock:
                if not self.running:
                    break
                roi = self.roi
                target = self.target_number
                cooldown = self.cooldown

            frame = self.camera.read()
            if frame is None:
                time.sleep(0.2)
                continue

            number, raw_text = self.ocr.read_number(frame, roi)
            stable_number, is_stable = self.stability.update(number)
            now = time.time()

            # Draw debug overlay for the backend video endpoint.
            preview = frame.copy()
            x, y, w, h = roi
            if w > 0 and h > 0:
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            overlay = f"read={number} stable={stable_number} target={target} armed={self.armed}"
            cv2.putText(
                preview,
                overlay,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            ok_jpeg, jpeg = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            with self.lock:
                self.last_detected_number = number
                self.last_raw_text = raw_text
                self.last_stable_number = stable_number if is_stable else None
                if ok_jpeg:
                    self.last_frame_jpeg = jpeg.tobytes()

            if is_stable and stable_number is not None:
                self._record_stable_number(stable_number, target, now)

                if stable_number != target:
                    # Re-arm when another stable value appears.
                    with self.lock:
                        self.armed = True

                if stable_number == target:
                    should_trigger = False
                    with self.lock:
                        if (now - self.last_trigger_time) >= cooldown:
                            should_trigger = True
                            self.armed = False

                    if should_trigger:
                        # Ensure the frontend receives a fresh reading/action event
                        # for every real trigger, even when the same stable number
                        # remains visible across cooldown intervals.
                        with self.lock:
                            self._last_recorded_number = None
                        self._record_stable_number(stable_number, target, now)
                        self._send_trigger(reason=f"stable_match_{stable_number}")

            if not is_stable:
                with self.lock:
                    self._last_recorded_number = None

            time.sleep(0.01)

    def _record_stable_number(self, number: int, target: int, now: float) -> None:
        with self.lock:
            # Store one reading per stable plateau to avoid filling history with duplicates.
            if self._last_recorded_number == number and (now - self._last_recorded_time) < self.cooldown:
                return

            session_id = self.session["id"] if self.session else None
            reading = {
                "id": f"reading-{uuid.uuid4().hex[:12]}",
                "sessionId": session_id,
                "number": number,
                "confidence": 1.0,
                "capturedAt": utc_now_iso(),
                "source": "real-camera",
                "persisted": True,
                "isMatch": number == target,
            }
            self.readings.appendleft(reading)
            self.latest_reading = reading
            self._last_recorded_number = number
            self._last_recorded_time = now

    def create_test_reading(self, target_number: int, force_match: bool) -> dict[str, Any]:
        if not 0 <= target_number <= 36:
            raise ValueError("targetNumber must be between 0 and 36")

        number = target_number if force_match else ((target_number + 1) % 37)
        reading = {
            "id": f"test-{uuid.uuid4().hex[:12]}",
            "sessionId": self.session["id"] if self.session else None,
            "number": number,
            "confidence": 1.0,
            "capturedAt": utc_now_iso(),
            "source": "manual-test",
            "persisted": False,
            "isMatch": number == target_number,
        }
        with self.lock:
            self.latest_reading = reading
        return reading

    def clear_readings(self) -> None:
        with self.lock:
            self.readings.clear()
            self.latest_reading = None
            self._last_recorded_number = None
            self._last_recorded_time = 0.0

    def list_readings(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 2000))
        with self.lock:
            items = list(self.readings)
        if session_id:
            items = [item for item in items if item.get("sessionId") == session_id]
        return items[:limit]

    def get_latest_reading(self) -> dict[str, Any] | None:
        with self.lock:
            return dict(self.latest_reading) if self.latest_reading else None

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            return self.last_frame_jpeg

    def system_status(self) -> dict[str, str]:
        with self.lock:
            camera = "ready" if self.camera.status().get("is_open") or self.running else "disabled"
            pico_status = self.pico.status()
            pico = "connected" if pico_status.get("is_open") else "unavailable"
            actuator = "safe" if self.last_trigger_error is None else "fault"
            backend = "connected"
        return {
            "camera": camera,
            "backend": backend,
            "pico": pico,
            "actuator": actuator,
            "alarm": "ready",
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "target_number": self.target_number,
                "stable_frames_required": self.stable_frames,
                "history": self.stability.snapshot(),
                "last_detected_number": self.last_detected_number,
                "last_stable_number": self.last_stable_number,
                "last_raw_ocr_text": self.last_raw_text,
                "latest_reading": dict(self.latest_reading) if self.latest_reading else None,
                "readings_count": len(self.readings),
                "session": dict(self.session) if self.session else None,
                "roi": {
                    "x": self.roi[0],
                    "y": self.roi[1],
                    "w": self.roi[2],
                    "h": self.roi[3],
                },
                "armed": self.armed,
                "trigger_count": self.trigger_count,
                "last_trigger_ok": self.last_trigger_ok,
                "last_trigger_time": self.last_trigger_time,
                "last_trigger_error": self.last_trigger_error,
                "camera": self.camera.status(),
                "pico": self.pico.status(),
            }
