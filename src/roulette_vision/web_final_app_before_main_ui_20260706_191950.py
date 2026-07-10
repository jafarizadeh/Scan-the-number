import argparse, csv, json, threading, time, uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_rect_from_shape, roi_metrics
from .cnn_number_model import CNNNumberModel
from .hardware_actions import HardwareActionController, DEFAULT_HARDWARE_CONFIG, clean_hardware_config


APP_SETTINGS_PATH = Path("data/app_settings.json")
RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

DEFAULT_APP_SETTINGS = {
    "selectedListId": "list-1",
    "lists": [
        {"id": "list-1", "name": "List 1", "numbers": [0,8,10,11,13,20,29,35,36]},
        {"id": "list-2", "name": "List 2", "numbers": [1,4,7,14,22]},
        {"id": "list-3", "name": "List 3", "numbers": [3,6,9,12,15,18]},
    ],
    "hardware": dict(DEFAULT_HARDWARE_CONFIG),
}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def clean_numbers(values):
    out, seen = [], set()

    if isinstance(values, str):
        import re
        values = re.findall(r"\d+", values)

    for v in values or []:
        try:
            n = int(v)
        except Exception:
            continue

        if 0 <= n <= 36 and n not in seen:
            out.append(n)
            seen.add(n)

    return sorted(out)


def clean_lists(payload):
    raw = payload.get("lists", []) if isinstance(payload, dict) else []
    lists = []

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue

        lid = str(item.get("id") or f"list-{uuid.uuid4().hex[:8]}")
        name = str(item.get("name") or f"List {i+1}").strip()[:60]
        nums = clean_numbers(item.get("numbers", []))

        lists.append({"id": lid, "name": name, "numbers": nums})

    if not lists:
        lists = list(DEFAULT_APP_SETTINGS["lists"])

    selected = str(payload.get("selectedListId") or lists[0]["id"]) if isinstance(payload, dict) else lists[0]["id"]

    if selected not in {x["id"] for x in lists}:
        selected = lists[0]["id"]

    return {"selectedListId": selected, "lists": lists}


class AppStore:
    def __init__(self, path=APP_SETTINGS_PATH):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data = self.load()

    def load(self):
        data = json.loads(json.dumps(DEFAULT_APP_SETTINGS))

        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                data.update(raw)
        except Exception as exc:
            print("[APP] settings load failed:", exc)

        clean = clean_lists(data)
        clean["hardware"] = clean_hardware_config(data.get("hardware", {}))
        return clean

    def save_locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get(self):
        with self.lock:
            return json.loads(json.dumps(self.data))

    def update_lists(self, payload):
        clean = clean_lists(payload)

        with self.lock:
            self.data["lists"] = clean["lists"]
            self.data["selectedListId"] = clean["selectedListId"]
            self.save_locked()
            return json.loads(json.dumps(self.data))

    def update_hardware(self, payload):
        clean = clean_hardware_config(payload)

        with self.lock:
            self.data["hardware"] = clean
            self.save_locked()
            return dict(clean)

    def selected_list(self):
        data = self.get()
        sid = data.get("selectedListId")

        for item in data.get("lists", []):
            if item.get("id") == sid:
                return item

        return data["lists"][0]


class FinalVisionHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)

        model_cfg = cfg.get("cnn_model", {})
        self.model = CNNNumberModel(
            model_cfg.get("model_path", "data/models/roulette_mobilenet_best_single.onnx"),
            img_size=int(model_cfg.get("img_size", 160)),
        )

        self.store = AppStore()
        self.actions = HardwareActionController(self.store.get().get("hardware", {}))

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.status = {"ready": False, "state": "STARTING"}

        self.running = False
        self.monitoring = False
        self.camera_enabled = True
        self.thread = None

        self.last_output_number = None
        self.candidate_label = None
        self.candidate_predictions = []
        self.candidate_streak = 0

        self.latest_decision = None
        self.latest_match = None
        self.session_id = None
        self.selected_list = self.store.selected_list()

        self.history = deque(maxlen=2000)
        self.events = deque(maxlen=120)

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)

        self.event_log = Path("data/logs/final_events.csv")
        if not self.event_log.exists():
            with self.event_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "id", "timestamp", "number", "score", "margin",
                    "list_id", "list_name", "is_match", "action_accepted", "roi_path"
                ])

    def start(self):
        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.camera.close()
        self.actions.close()

    def set_camera_enabled(self, enabled):
        with self.lock:
            self.camera_enabled = bool(enabled)
            if not self.camera_enabled:
                self.monitoring = False
                self.reset_candidate()
        return self.public_status()

    def start_monitoring(self, selected_list_id=None):
        data = self.store.get()

        if selected_list_id:
            data["selectedListId"] = selected_list_id
            self.store.update_lists(data)

        self.selected_list = self.store.selected_list()

        with self.lock:
            self.camera_enabled = True
            self.monitoring = True
            self.session_id = f"session-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.last_output_number = None
            self.latest_match = None
            self.reset_candidate()

        return self.public_status()

    def stop_monitoring(self):
        with self.lock:
            self.monitoring = False
            self.reset_candidate()
        return self.public_status()

    def reset_candidate(self):
        self.candidate_label = None
        self.candidate_predictions = []
        self.candidate_streak = 0

    def quick_state(self, roi, metrics):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        white = float(((v >= 145) & (s <= 130)).mean())
        green = float(((h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)).mean())
        red = float(((((h <= 12) | (h >= 165)) & (s >= 45) & (v >= 45))).mean())

        if white >= 0.003 or green >= 0.03 or red >= 0.03 or metrics["std"] >= 18.0:
            return "RESULT_VISIBLE"

        return "LOW_SIGNAL"

    def save_roi(self, number):
        with self.lock:
            roi = None if self.latest_roi is None else self.latest_roi.copy()

        if roi is None:
            return ""

        folder = Path("data/evidence")
        folder.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = folder / f"final_{number}_{ts}.png"

        cv2.imwrite(str(path), roi)
        return str(path)

    def record_number(self, number, score, margin):
        selected = self.selected_list or self.store.selected_list()
        target_set = set(int(x) for x in selected.get("numbers", []))

        is_match = int(number) in target_set
        roi_path = self.save_roi(number)
        event_id = f"ev-{uuid.uuid4().hex[:10]}"

        action_result = {"accepted": False, "reason": "not_match"}

        if is_match:
            action_result = self.actions.trigger_async(
                number=number,
                list_name=selected.get("name", ""),
                reason="number_in_selected_list",
            )

        event = {
            "id": event_id,
            "timestamp": now_iso(),
            "number": int(number),
            "score": float(score),
            "margin": float(margin),
            "listId": selected.get("id"),
            "listName": selected.get("name"),
            "targetNumbers": list(selected.get("numbers", [])),
            "isMatch": bool(is_match),
            "action": action_result,
            "roiPath": roi_path,
        }

        with self.lock:
            self.history.appendleft(event)
            self.events.appendleft(event)
            self.last_output_number = int(number)
            if is_match:
                self.latest_match = event

        with self.event_log.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                event_id, event["timestamp"], number,
                round(score, 4), round(margin, 4),
                selected.get("id"), selected.get("name"),
                is_match, action_result.get("accepted"), roi_path
            ])

        return event

    def finalize_candidate(self):
        preds = list(self.candidate_predictions)

        if not preds:
            self.reset_candidate()
            return None

        label = int(self.candidate_label)
        avg_score = sum(float(p["score"]) for p in preds) / len(preds)
        avg_margin = sum(float(p["margin"]) for p in preds) / len(preds)

        cfg = self.cfg.get("final_app", {})
        event = None

        if (
            avg_score >= float(cfg.get("min_confirm_score", 0.85))
            and avg_margin >= float(cfg.get("min_confirm_margin", 0.20))
        ):
            event = self.record_number(label, avg_score, avg_margin)

        self.reset_candidate()
        return event

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        cfg = self.cfg.get("final_app", {})
        min_stable = int(cfg.get("min_stable_frames", 2))
        min_score = float(cfg.get("min_live_score", 0.70))
        min_margin = float(cfg.get("min_live_margin", 0.10))

        while self.running:
            frame = self.camera.read()

            if frame is None:
                time.sleep(0.03)
                continue

            roi = crop_relative_roi(frame, self.cfg["roi"])
            metrics = roi_metrics(roi)

            with self.lock:
                cam_on = self.camera_enabled
                monitoring = self.monitoring

            state = "CAMERA_OFF" if not cam_on else self.quick_state(roi, metrics)

            decision = None
            event = None

            if cam_on and monitoring:
                try:
                    decision = self.model.predict(roi)
                    self.latest_decision = decision

                    label = int(decision["label"])
                    score = float(decision["score"])
                    margin = float(decision["margin"])

                    usable = state == "RESULT_VISIBLE" and score >= min_score and margin >= min_margin

                    if usable:
                        # نسخه فعلی: عدد تکراری پشت سر هم عمداً ثبت نمی‌شود.
                        if self.last_output_number is not None and label == int(self.last_output_number):
                            self.reset_candidate()
                        else:
                            if self.candidate_label == label:
                                self.candidate_streak += 1
                                self.candidate_predictions.append(decision)
                            else:
                                self.candidate_label = label
                                self.candidate_streak = 1
                                self.candidate_predictions = [decision]

                            if self.candidate_streak >= min_stable:
                                event = self.finalize_candidate()
                    else:
                        self.reset_candidate()

                except Exception as exc:
                    state = f"ERROR: {exc}"
                    self.reset_candidate()

            status = {
                "ready": True,
                "cameraEnabled": cam_on,
                "monitoring": monitoring,
                "sessionId": self.session_id,
                "state": state,
                "selectedList": self.selected_list,
                "decision": decision or self.latest_decision,
                "candidateLabel": self.candidate_label,
                "candidateStreak": self.candidate_streak,
                "lastOutputNumber": self.last_output_number,
                "latestMatch": self.latest_match,
                "history": list(self.history)[:240],
                "event": event,
                "metrics": metrics,
                "hardware": self.actions.status(),
            }

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_roi = roi.copy()
                self.status = json.loads(json.dumps(status, default=str))

            time.sleep(delay)

    def public_status(self):
        with self.lock:
            return json.loads(json.dumps(self.status, default=str))

    def get_frame(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            cam_on = self.camera_enabled

        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        if not cam_on:
            black = np.zeros_like(frame)
            cv2.putText(black, "CAMERA OFF", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255,255,255), 4, cv2.LINE_AA)
            return black

        x, y, w, h = roi_rect_from_shape(frame.shape, self.cfg["roi"])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        return frame

    def get_roi(self):
        with self.lock:
            roi = None if self.latest_roi is None else self.latest_roi.copy()
            cam_on = self.camera_enabled

        if roi is None:
            roi = np.zeros((160, 120, 3), dtype=np.uint8)

        if not cam_on:
            return np.zeros_like(roi)

        return roi

    def clear_history(self):
        with self.lock:
            self.history.clear()
            self.latest_match = None
        return {"ok": True}


def encode_jpeg(frame, quality=84):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roulette Vision</title>
<style>
:root{
  --bg:#eef3f8;
  --panel:#f8fbff;
  --panel-2:#ffffff;
  --line:#d7e2ee;
  --line-2:#c3d2e3;
  --text:#0f2944;
  --muted:#6f8298;
  --blue:#2f80ed;
  --blue-2:#e8f1ff;
  --green:#21b66f;
  --green-2:#eaf9f1;
  --red:#e23a49;
  --red-2:#ffecef;
  --blackchip:#1f2d3d;
  --shadow:0 8px 22px rgba(22,51,84,.08);
  --radius:18px;
  --radius-sm:14px;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:linear-gradient(180deg,#f6f9fc 0%,#edf3f8 100%);color:var(--text);font-family:Inter,Arial,sans-serif}
body{min-width:1180px}
button,input{font:inherit}
.app{padding:14px 16px 18px}
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  gap:14px;margin-bottom:12px
}
.brand{
  display:flex;align-items:center;gap:12px;font-size:26px;font-weight:900;letter-spacing:.2px
}
.brand-logo{
  width:30px;height:30px;border-radius:50%;
  background:radial-gradient(circle at 35% 35%, #ffd96a 0 18%, #db3d3d 19% 34%, #1c2a39 35% 56%, #23b969 57% 72%, #f7f7f7 73% 100%);
  border:2px solid #203246;
  box-shadow:0 2px 6px rgba(0,0,0,.12)
}
.header-actions{display:flex;align-items:center;gap:10px}
.head-btn,.icon-btn,.status-pill,.stop-btn{
  border:1px solid var(--line-2);
  background:rgba(255,255,255,.72);
  color:var(--text);
  border-radius:15px;
  height:44px;
  padding:0 16px;
  display:inline-flex;
  align-items:center;
  gap:9px;
  font-weight:800;
  box-shadow:var(--shadow);
  cursor:pointer;
}
.head-btn.off{opacity:.72}
.head-btn.active{background:var(--panel-2)}
.status-pill{
  background:#eff8f2;
  color:#1a8c57;
  cursor:default;
}
.status-dot{
  width:10px;height:10px;border-radius:50%;background:#1fb467;display:inline-block
}
.stop-btn{
  background:linear-gradient(180deg,#f14549,#dd2435);
  color:#fff;border-color:#d83b45;min-width:136px;justify-content:center
}
.icon-btn{width:44px;justify-content:center;padding:0}

.top-grid{
  display:grid;
  grid-template-columns:270px 1.22fr 240px 480px;
  gap:12px;
  align-items:stretch;
}
.panel{
  background:rgba(255,255,255,.72);
  border:1px solid var(--line);
  border-radius:22px;
  box-shadow:var(--shadow);
}
.panel-title{
  color:#425c77;
  font-size:13px;
  font-weight:900;
  letter-spacing:1px;
  text-transform:uppercase;
  margin-bottom:12px
}
.controls-panel{padding:14px}
.control-card{
  border:1px solid var(--line);
  background:var(--panel-2);
  border-radius:18px;
  padding:14px 16px;
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:12px;
  min-height:62px
}
.control-left{display:flex;align-items:center;gap:12px;font-size:17px;font-weight:900}
.control-icon{
  width:30px;height:30px;border-radius:10px;background:#f2f6fb;border:1px solid var(--line);display:flex;align-items:center;justify-content:center
}
.switch{
  width:50px;height:30px;border-radius:999px;background:#b9c8d8;position:relative;cursor:pointer;transition:.18s
}
.switch::after{
  content:"";position:absolute;top:3px;left:3px;width:24px;height:24px;border-radius:50%;background:#fff;box-shadow:0 2px 6px rgba(0,0,0,.18);transition:.18s
}
.switch.on{background:#28bf73}
.switch.on::after{left:23px}
.status-box{
  border:1px solid var(--line);
  background:linear-gradient(180deg,#fbfdff 0%,#f4f8fc 100%);
  border-radius:18px;padding:14px 16px;margin-top:2px
}
.status-mini{display:flex;align-items:center;gap:8px;font-weight:900;color:#51708f;font-size:13px;text-transform:uppercase}
.status-value{
  margin-top:10px;font-size:20px;font-weight:900;color:#1eb367
}
.status-sub{margin-top:4px;color:#6f8398;font-size:13px}

.camera-panel{padding:14px}
.camera-frame{
  position:relative;
  background:#07111a;
  border:1px solid #20374d;
  border-radius:20px;
  overflow:hidden;
  height:258px;
}
.camera-frame img{
  width:100%;height:100%;object-fit:cover;display:block;
}
.live-badge{
  position:absolute;left:14px;top:12px;background:#144f2e;color:#fff;
  border-radius:12px;padding:6px 12px;font-size:13px;font-weight:900;
  box-shadow:0 5px 14px rgba(20,79,46,.25)
}
.camera-disabled{
  position:absolute;inset:0;background:rgba(243,247,251,.82);display:none;align-items:center;justify-content:center;
  color:#8aa0b6;font-size:24px;font-weight:900;backdrop-filter:blur(2px)
}
.camera-frame.off .camera-disabled{display:flex}
.camera-actions{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.big-action{
  height:58px;border:none;border-radius:18px;font-size:18px;font-weight:900;color:#fff;cursor:pointer;box-shadow:var(--shadow)
}
.start-action{background:linear-gradient(180deg,#2fcb7b,#20b667)}
.stop-action{background:linear-gradient(180deg,#f35865,#e23748)}

.info-panel{padding:14px}
.info-grid{display:grid;grid-template-columns:1fr;gap:10px}
.info-card{
  min-height:78px;
  border:1px solid var(--line);
  background:var(--panel-2);
  border-radius:18px;
  padding:14px 16px;
}
.info-label{font-size:13px;font-weight:900;color:#5d7792;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px}
.info-value{font-size:18px;font-weight:900}
.info-big{font-size:20px;font-weight:900}
.scan-green{color:#1fb467}
.mono{font-variant-numeric:tabular-nums}
.dash{color:#93a5b7}

.list-panel{padding:14px}
.list-top{
  display:grid;grid-template-columns:1fr 130px;gap:10px;align-items:start
}
.list-field{
  min-height:52px;border:1px solid var(--line);background:var(--panel-2);
  border-radius:15px;padding:12px 15px;font-weight:900;font-size:18px;display:flex;align-items:center;justify-content:space-between
}
.select-btn{
  min-height:52px;border:1px solid #8ab1ff;background:linear-gradient(180deg,#fdfefe,#eef5ff);color:var(--blue);
  border-radius:15px;font-weight:900;cursor:pointer;box-shadow:var(--shadow)
}
.list-sub{margin:12px 0 10px;color:#5d7690;font-size:15px;font-weight:800}
.number-row,.quick-grid,.preview-chip-row{
  display:flex;gap:10px;flex-wrap:wrap
}
.number-chip{
  min-width:48px;height:38px;padding:0 10px;border-radius:13px;
  display:inline-flex;align-items:center;justify-content:center;
  font-size:16px;font-weight:900;color:#fff;
  border:2px solid rgba(255,255,255,.65);
  box-shadow:0 4px 10px rgba(12,26,44,.10), inset 0 1px 0 rgba(255,255,255,.18)
}
.number-chip.large{min-width:56px;height:42px;font-size:17px}
.number-chip.small{min-width:28px;height:28px;font-size:14px;padding:0 8px;border-radius:10px}
.num-red{background:linear-gradient(180deg,#f25463,#de3648)}
.num-black{background:linear-gradient(180deg,#2b3848,#1e2936)}
.num-green{background:linear-gradient(180deg,#32d07a,#1cb467)}
.preview-title{
  margin-top:18px;padding-top:14px;border-top:1px solid var(--line);font-size:14px;font-weight:900;color:#536e89;text-transform:uppercase
}
.line-link{
  margin-top:14px;width:100%;height:38px;border-radius:12px;border:1px solid #8ab1ff;background:#f4f8ff;color:#2d6fe3;font-weight:900;cursor:pointer
}

.history-panel{
  margin-top:12px;padding:12px 14px 16px
}
.history-head{
  display:flex;align-items:center;justify-content:space-between;margin-bottom:12px
}
.history-title{font-size:16px;font-weight:900;color:#425c77;text-transform:uppercase}
.history-hint{font-size:12px;color:#6f8397;font-weight:700}
.history-strip{
  display:flex;gap:16px;overflow-x:auto;overflow-y:hidden;scrollbar-width:none;
  user-select:none;padding-bottom:4px
}
.history-strip::-webkit-scrollbar{display:none}
.history-item{
  flex:0 0 auto;text-align:center
}
.history-chip{
  position:relative;
  width:70px;height:44px;border-radius:15px;border:2px solid rgba(255,255,255,.9);
  display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:900;color:#fff;
  box-shadow:0 7px 16px rgba(17,36,58,.14)
}
.history-chip.hit{
  outline:3px solid #f2c400;
  box-shadow:0 0 0 5px rgba(242,196,0,.18), 0 7px 16px rgba(17,36,58,.14)
}
.history-star{
  position:absolute;right:6px;top:4px;font-size:14px;color:#ffd83d
}
.history-time{margin-top:8px;font-size:12px;color:#6f8398;font-weight:700}
.history-name{margin-top:3px;font-size:11px;color:#94a4b6}

.workspace{
  margin-top:12px;
  display:grid;
  grid-template-columns:1fr 1fr 1fr;
  gap:12px;
}
.workspace-card{
  min-height:272px;
  background:rgba(255,255,255,.72);
  border:1px solid var(--line);
  border-radius:22px;
  box-shadow:var(--shadow);
  overflow:hidden;
}
.workspace-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 16px;border-bottom:1px solid var(--line)
}
.workspace-head h3{margin:0;font-size:15px;text-transform:uppercase;letter-spacing:.7px}
.close-x{border:none;background:transparent;color:#7f93a8;font-size:24px;line-height:1;cursor:pointer}
.workspace-body{padding:14px 16px}
.list-card{
  border:1px solid var(--line);
  background:var(--panel-2);
  border-radius:18px;
  padding:14px 16px;
  margin-bottom:12px;
  cursor:pointer;
  transition:.15s;
}
.list-card.active{border:2px solid #3e83f3;box-shadow:0 0 0 4px #edf4ff}
.list-card-title{font-size:18px;font-weight:900}
.list-card-count{margin-top:8px;color:#6f8296;font-size:13px;font-weight:800}
.list-card-preview{margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}

.num-editor-grid{
  display:grid;grid-template-columns:repeat(6,1fr);gap:10px
}
.num-pick{
  height:40px;border-radius:14px;border:2px solid rgba(255,255,255,.9);
  color:#fff;font-size:16px;font-weight:900;cursor:pointer;position:relative;
  box-shadow:0 5px 12px rgba(16,39,62,.10)
}
.num-pick.selected{
  outline:3px solid #3a80f0;
  box-shadow:0 0 0 5px rgba(58,128,240,.16), 0 5px 12px rgba(16,39,62,.10)
}
.num-pick.selected::after{
  content:"✓";position:absolute;right:8px;top:5px;font-size:13px
}
.editor-foot{
  display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:14px
}
.foot-note{font-size:13px;color:#708397;font-weight:800}
.action-row{display:flex;gap:10px}
.soft-btn,.blue-btn,.green-btn,.red-btn{
  height:40px;padding:0 16px;border-radius:12px;font-weight:900;cursor:pointer
}
.soft-btn{border:1px solid var(--line-2);background:#fff;color:var(--text)}
.blue-btn{border:1px solid #2f80ed;background:linear-gradient(180deg,#3f8cff,#2f79e6);color:#fff}
.green-btn{border:1px solid #1fa763;background:linear-gradient(180deg,#2dcc7d,#20b667);color:#fff}
.red-btn{border:1px solid #de4150;background:linear-gradient(180deg,#f25d69,#de3849);color:#fff}

.settings-layout{
  display:grid;grid-template-columns:102px 1fr;gap:16px;min-height:220px
}
.settings-tabs{
  border-right:1px solid var(--line);
  padding-right:12px
}
.settings-tab{
  width:100%;height:40px;border:none;background:transparent;color:#4a6480;font-weight:900;border-radius:10px;text-align:left;padding:0 12px;cursor:pointer;margin-bottom:8px
}
.settings-tab.active{background:linear-gradient(180deg,#3b8bff,#2c74df);color:#fff}
.settings-area{display:block}
.settings-top-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}
.settings-actions{display:flex;gap:10px}
.settings-list-box{
  border:1px solid var(--line);border-radius:18px;background:var(--panel-2);padding:14px 16px;margin-bottom:12px
}
.settings-list-box.active{border:2px solid #3e83f3;box-shadow:0 0 0 4px #edf4ff}
.settings-list-name{font-size:18px;font-weight:900}
.settings-list-meta{margin-top:6px;color:#73859a;font-size:13px;font-weight:700}
.hw-block{
  display:grid;grid-template-columns:1fr auto;gap:12px 10px;align-items:center;
  border:1px solid var(--line);background:var(--panel-2);border-radius:18px;padding:14px 16px;margin-bottom:14px
}
.hw-label{font-size:15px;font-weight:900}
.hw-note{
  border:1px solid var(--line);background:linear-gradient(180deg,#fbfdff,#f5f8fc);border-radius:18px;padding:14px 16px;color:#73859a;font-weight:800;line-height:1.35
}
.empty-note{color:#7a8ea3;font-size:15px;line-height:1.45}
.hidden{display:none !important}

.match-popup{
  position:fixed;inset:0;background:rgba(20,31,44,.28);display:flex;align-items:center;justify-content:center;z-index:50
}
.match-box{
  width:420px;max-width:92vw;background:#fff;border:2px solid #dbecdf;border-radius:28px;box-shadow:0 22px 60px rgba(26,60,90,.22);
  padding:24px 28px;text-align:center
}
.match-head{font-size:18px;font-weight:900;color:#28a764;text-transform:uppercase;letter-spacing:1px}
.match-number{
  margin:16px auto 8px;
  width:120px;height:120px;border-radius:28px;color:#fff;font-size:54px;font-weight:900;
  display:flex;align-items:center;justify-content:center
}
.match-text{font-size:16px;font-weight:800;color:#4f6983}
.popup-actions{margin-top:18px}
.popup-actions button{
  min-width:120px;height:42px;border-radius:14px;border:none;background:linear-gradient(180deg,#32c97c,#20b667);color:#fff;font-weight:900;cursor:pointer
}

@media (max-width:1400px){
  .top-grid{grid-template-columns:240px 1.18fr 220px 420px}
}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="brand">
      <div class="brand-logo"></div>
      <div>Roulette Vision</div>
    </div>
    <div class="header-actions">
      <button class="head-btn" id="hdrSoundBtn" onclick="toggleSound()">
        <span>🔊</span><span>Sound</span>
      </button>
      <button class="head-btn" id="hdrArmBtn" onclick="toggleArm()">
        <span>🦾</span><span>Arm</span>
      </button>
      <div class="status-pill" id="scanPill"><span class="status-dot"></span><span>Scanning</span></div>
      <button class="stop-btn" onclick="stopScan()">STOP</button>
      <button class="icon-btn" onclick="openSettingsPanel()">⚙️</button>
    </div>
  </div>

  <section class="top-grid">
    <aside class="panel controls-panel">
      <div class="panel-title">Controls</div>

      <div class="control-card">
        <div class="control-left"><div class="control-icon">📷</div><div>Camera</div></div>
        <div class="switch on" id="cameraSwitch" onclick="toggleCameraVisual()"></div>
      </div>

      <div class="control-card">
        <div class="control-left"><div class="control-icon">🔊</div><div>Sound</div></div>
        <div class="switch" id="soundSwitch" onclick="toggleSound()"></div>
      </div>

      <div class="control-card">
        <div class="control-left"><div class="control-icon">🦾</div><div>Arm</div></div>
        <div class="switch" id="armSwitch" onclick="toggleArm()"></div>
      </div>

      <div class="status-box">
        <div class="status-mini"><span class="status-dot" id="statusDot"></span><span>Status</span></div>
        <div class="status-value" id="statusText">SCANNING</div>
        <div class="status-sub" id="statusSub">System is active</div>
      </div>
    </aside>

    <section class="panel camera-panel">
      <div class="panel-title">Live Camera</div>
      <div class="camera-frame" id="cameraFrame">
        <img id="streamImg" src="/stream.mjpg" alt="Live camera">
        <div class="live-badge">LIVE</div>
        <div class="camera-disabled">CAMERA OFF</div>
      </div>
      <div class="camera-actions">
        <button class="big-action start-action" onclick="startScan()">▶ START</button>
        <button class="big-action stop-action" onclick="stopScan()">■ STOP</button>
      </div>
    </section>

    <section class="panel info-panel">
      <div class="panel-title">Current Info</div>
      <div class="info-grid">
        <div class="info-card">
          <div class="info-label">Scan</div>
          <div class="info-big scan-green" id="scanStateValue">SCANNING</div>
        </div>
        <div class="info-card">
          <div class="info-label">Last Number</div>
          <div class="info-big mono" id="lastNumberValue">--</div>
        </div>
        <div class="info-card">
          <div class="info-label">Last Match</div>
          <div class="info-big mono" id="lastMatchValue">--</div>
        </div>
      </div>
    </section>

    <section class="panel list-panel">
      <div class="panel-title">Active List</div>

      <div class="list-top">
        <div class="list-field">
          <span id="activeListName">List 1</span>
          <span>▾</span>
        </div>
        <button class="select-btn" onclick="openSelectPanel()">Select List</button>
      </div>

      <div class="list-sub" id="activeListMeta">0 numbers</div>
      <div class="number-row" id="activeListNumbers"></div>

      <div class="preview-title">Quick Preview</div>
      <div class="quick-grid" id="quickPreviewNumbers"></div>
      <button class="line-link" onclick="openNumbersPanel()">View All Numbers</button>
    </section>
  </section>

  <section class="panel history-panel">
    <div class="history-head">
      <div class="history-title">History</div>
      <div class="history-hint">Swipe left/right to scroll</div>
    </div>
    <div class="history-strip" id="historyStrip"></div>
  </section>

  <section class="workspace">
    <div class="workspace-card" id="panelSelectList">
      <div class="workspace-head">
        <h3>Select Active List</h3>
        <button class="close-x" onclick="hidePanel('panelSelectList')">×</button>
      </div>
      <div class="workspace-body" id="selectListBody"></div>
    </div>

    <div class="workspace-card" id="panelNumbers">
      <div class="workspace-head">
        <h3 id="numbersPanelTitle">Set Numbers</h3>
        <button class="close-x" onclick="hidePanel('panelNumbers')">×</button>
      </div>
      <div class="workspace-body">
        <div class="num-editor-grid" id="numberEditorGrid"></div>
        <div class="editor-foot">
          <div class="foot-note" id="editorCount">Selected: 0 numbers</div>
          <div class="action-row">
            <button class="soft-btn" onclick="clearEditorNumbers()">Clear All</button>
            <button class="blue-btn" onclick="saveEditorNumbers()">Save Numbers</button>
          </div>
        </div>
      </div>
    </div>

    <div class="workspace-card" id="panelSettings">
      <div class="workspace-head">
        <h3>Settings</h3>
        <button class="close-x" onclick="hidePanel('panelSettings')">×</button>
      </div>
      <div class="workspace-body">
        <div class="settings-layout">
          <div class="settings-tabs">
            <button class="settings-tab active" id="tabListsBtn" onclick="showSettingsTab('lists')">Lists</button>
            <button class="settings-tab" id="tabHardwareBtn" onclick="showSettingsTab('hardware')">Hardware</button>
            <button class="settings-tab" id="tabSystemBtn" onclick="showSettingsTab('system')">System</button>
          </div>
          <div class="settings-area">
            <div id="settingsTabLists"></div>
            <div id="settingsTabHardware" class="hidden"></div>
            <div id="settingsTabSystem" class="hidden"></div>
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

<div class="match-popup hidden" id="matchPopup">
  <div class="match-box">
    <div class="match-head">Match Found</div>
    <div class="match-number" id="popupNumber">7</div>
    <div class="match-text" id="popupText">Number found in selected list</div>
    <div class="popup-actions"><button onclick="closePopup()">OK</button></div>
  </div>
</div>

<script>
const RED_NUMBERS = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

const ui = {
  cameraVisible: true,
  lastPopupId: null,
  settingsTab: "lists",
  selectedSettingsListId: null,
  editorListId: null
};

let appSettings = null;
let liveStatus = null;

function numClass(n){
  if(Number(n) === 0) return "num-green";
  return RED_NUMBERS.has(Number(n)) ? "num-red" : "num-black";
}

function chipHTML(n, extraClass="", hit=false){
  return `<div class="number-chip ${numClass(n)} ${extraClass}">${n}</div>`;
}

function historyChipHTML(item){
  const hit = item.isMatch ? " hit" : "";
  const star = item.isMatch ? `<span class="history-star">★</span>` : "";
  return `
    <div class="history-item">
      <div class="history-chip ${numClass(item.number)}${hit}">
        ${item.number}
        ${star}
      </div>
      <div class="history-time">${shortTime(item.timestamp)}</div>
      <div class="history-name">${item.listName || ""}</div>
    </div>
  `;
}

function shortTime(v){
  if(!v) return "--";
  try{
    const s = String(v);
    return s.includes("T") ? s.split("T")[1] : s.slice(-8);
  }catch(e){
    return String(v);
  }
}

async function getJSON(url){
  const r = await fetch(url);
  return await r.json();
}

async function postJSON(url, data={}){
  const r = await fetch(url,{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(data)
  });
  return await r.json();
}

function currentList(){
  if(!appSettings) return null;
  return appSettings.lists.find(x => x.id === appSettings.selectedListId) || appSettings.lists[0] || null;
}

function getListById(id){
  if(!appSettings) return null;
  return appSettings.lists.find(x => x.id === id) || null;
}

function ensureSelectionState(){
  if(!appSettings) return;
  if(!ui.selectedSettingsListId){
    ui.selectedSettingsListId = appSettings.selectedListId || (appSettings.lists[0] && appSettings.lists[0].id);
  }
  if(!ui.editorListId){
    ui.editorListId = ui.selectedSettingsListId;
  }
}

function renderHeader(){
  const monitoring = !!(liveStatus && liveStatus.monitoring);
  document.getElementById("scanPill").innerHTML =
    monitoring
      ? `<span class="status-dot"></span><span>Scanning</span>`
      : `<span class="status-dot" style="background:#99a9ba"></span><span>Stopped</span>`;

  const hw = (appSettings && appSettings.hardware) ? appSettings.hardware : {};
  setSwitch("soundSwitch", !!hw.buzzerEnabled);
  setSwitch("armSwitch", !!hw.actuatorEnabled);

  const hdrSound = document.getElementById("hdrSoundBtn");
  const hdrArm = document.getElementById("hdrArmBtn");
  hdrSound.classList.toggle("off", !hw.buzzerEnabled);
  hdrArm.classList.toggle("off", !hw.actuatorEnabled);

  document.getElementById("statusText").textContent = monitoring ? "SCANNING" : "STOPPED";
  document.getElementById("statusText").style.color = monitoring ? "#1eb367" : "#d34b58";
  document.getElementById("statusSub").textContent = monitoring ? "System is active" : "Monitoring is stopped";
  document.getElementById("statusDot").style.background = monitoring ? "#1fb467" : "#9aaaba";

  document.getElementById("scanStateValue").textContent = monitoring ? "SCANNING" : "STOP";
  document.getElementById("scanStateValue").classList.toggle("scan-green", monitoring);
}

function renderCameraState(){
  const frame = document.getElementById("cameraFrame");
  const sw = document.getElementById("cameraSwitch");
  frame.classList.toggle("off", !ui.cameraVisible);
  setSwitchNode(sw, ui.cameraVisible);
}

function renderInfoCards(){
  const lastNum = liveStatus && liveStatus.lastOutputNumber != null ? liveStatus.lastOutputNumber : "--";
  const lastMatch = liveStatus && liveStatus.latestMatch ? liveStatus.latestMatch.number : "--";
  document.getElementById("lastNumberValue").textContent = lastNum;
  document.getElementById("lastMatchValue").textContent = lastMatch;
}

function renderActiveList(){
  const list = currentList();
  if(!list){
    document.getElementById("activeListName").textContent = "--";
    document.getElementById("activeListMeta").textContent = "0 numbers";
    document.getElementById("activeListNumbers").innerHTML = "";
    document.getElementById("quickPreviewNumbers").innerHTML = "";
    return;
  }

  document.getElementById("activeListName").textContent = list.name;
  document.getElementById("activeListMeta").textContent = `${list.numbers.length} numbers`;

  document.getElementById("activeListNumbers").innerHTML =
    list.numbers.map(n => chipHTML(n, "large")).join("");

  const quick = [];
  for(let i=0;i<=11;i++) quick.push(i);

  document.getElementById("quickPreviewNumbers").innerHTML =
    quick.map(n => chipHTML(n)).join("");
}

function renderHistory(){
  const items = (liveStatus && liveStatus.history) ? liveStatus.history.slice(0, 60) : [];
  document.getElementById("historyStrip").innerHTML = items.map(historyChipHTML).join("");
}

function renderSelectPanel(){
  const body = document.getElementById("selectListBody");
  if(!appSettings){
    body.innerHTML = `<div class="empty-note">Loading...</div>`;
    return;
  }
  body.innerHTML = appSettings.lists.map(l => `
    <div class="list-card ${appSettings.selectedListId === l.id ? "active" : ""}" onclick="selectActiveList('${l.id}')">
      <div class="list-card-title">${escapeHtml(l.name)}</div>
      <div class="list-card-count">${l.numbers.length} numbers</div>
      <div class="list-card-preview">${l.numbers.map(n => `<div class="number-chip ${numClass(n)} small">${n}</div>`).join("")}</div>
    </div>
  `).join("");
}

function renderNumbersPanel(){
  ensureSelectionState();
  const list = getListById(ui.editorListId) || currentList();
  if(!list){
    document.getElementById("numbersPanelTitle").textContent = "Set Numbers";
    document.getElementById("numberEditorGrid").innerHTML = "";
    document.getElementById("editorCount").textContent = "Selected: 0 numbers";
    return;
  }

  document.getElementById("numbersPanelTitle").textContent = `Set Numbers For ${list.name}`;
  document.getElementById("numberEditorGrid").innerHTML = Array.from({length:37},(_,n)=>n).map(n => {
    const sel = list.numbers.includes(n) ? "selected" : "";
    return `<button class="num-pick ${numClass(n)} ${sel}" onclick="toggleEditorNumber(${n})">${n}</button>`;
  }).join("");
  document.getElementById("editorCount").textContent = `Selected: ${list.numbers.length} numbers`;
}

function renderSettingsLists(){
  ensureSelectionState();
  const active = ui.selectedSettingsListId;
  const html = `
    <div class="settings-top-row">
      <div style="font-size:14px;font-weight:900;color:#617b95;text-transform:uppercase">List Management</div>
      <div class="settings-actions">
        <button class="blue-btn" onclick="addList()">+ Add List</button>
        <button class="soft-btn" onclick="renameSelectedList()">✎ Edit</button>
        <button class="soft-btn" style="color:#d83e4d;border-color:#f0b8bf" onclick="deleteSelectedList()">🗑 Delete</button>
      </div>
    </div>
    ${(appSettings.lists || []).map(l => `
      <div class="settings-list-box ${l.id===active ? "active":""}" onclick="selectSettingsList('${l.id}')">
        <div class="settings-list-name">${escapeHtml(l.name)}</div>
        <div class="settings-list-meta">${l.numbers.length} numbers</div>
        <div class="list-card-preview" style="margin-top:10px">
          ${l.numbers.map(n => `<div class="number-chip ${numClass(n)} small">${n}</div>`).join("")}
        </div>
      </div>
    `).join("")}
    <div class="action-row" style="margin-top:12px">
      <button class="blue-btn" onclick="editSelectedListNumbers()">Fill Numbers</button>
      <button class="soft-btn" onclick="clearSelectedListNumbers()">Clear Numbers</button>
      <button class="green-btn" onclick="saveListsToServer()">Save Lists</button>
    </div>
  `;
  document.getElementById("settingsTabLists").innerHTML = html;
}

function renderSettingsHardware(){
  const hw = appSettings.hardware || {};
  document.getElementById("settingsTabHardware").innerHTML = `
    <div class="hw-block">
      <div class="hw-label">Dry run</div>
      <div class="switch ${hw.dryRun ? "on" : ""}" onclick="toggleHardwareField('dryRun')"></div>

      <div class="hw-label">Sound</div>
      <div class="switch ${hw.buzzerEnabled ? "on" : ""}" onclick="toggleHardwareField('buzzerEnabled')"></div>

      <div class="hw-label">Arm</div>
      <div class="switch ${hw.actuatorEnabled ? "on" : ""}" onclick="toggleHardwareField('actuatorEnabled')"></div>

      <div class="hw-label">Save hardware</div>
      <button class="green-btn" onclick="saveHardwareToServer()">Save Hardware</button>

      <div class="hw-label">Test action</div>
      <button class="soft-btn" onclick="testAction()">Test Action</button>
    </div>
    <div class="hw-note">
      GPIO defaults:<br>
      Buzzer: GPIO${hw.buzzerPin ?? 17}<br>
      Arm: GPIO${hw.actuatorPin ?? 18}
    </div>
  `;
}

function renderSettingsSystem(){
  const st = liveStatus || {};
  document.getElementById("settingsTabSystem").innerHTML = `
    <div class="hw-note">
      Monitoring: <b>${st.monitoring ? "ON" : "OFF"}</b><br>
      State: <b>${escapeHtml(st.state || "-")}</b><br>
      Last output: <b>${st.lastOutputNumber != null ? st.lastOutputNumber : "--"}</b><br>
      Last match: <b>${st.latestMatch ? st.latestMatch.number : "--"}</b><br>
      History items: <b>${(st.history || []).length}</b><br><br>
      <div class="action-row">
        <button class="soft-btn" onclick="clearHistory()">Clear History</button>
      </div>
    </div>
  `;
}

function renderSettingsTabs(){
  document.getElementById("tabListsBtn").classList.toggle("active", ui.settingsTab === "lists");
  document.getElementById("tabHardwareBtn").classList.toggle("active", ui.settingsTab === "hardware");
  document.getElementById("tabSystemBtn").classList.toggle("active", ui.settingsTab === "system");

  document.getElementById("settingsTabLists").classList.toggle("hidden", ui.settingsTab !== "lists");
  document.getElementById("settingsTabHardware").classList.toggle("hidden", ui.settingsTab !== "hardware");
  document.getElementById("settingsTabSystem").classList.toggle("hidden", ui.settingsTab !== "system");
}

function renderAll(){
  if(!appSettings) return;
  renderHeader();
  renderCameraState();
  renderInfoCards();
  renderActiveList();
  renderHistory();
  renderSelectPanel();
  renderNumbersPanel();
  renderSettingsLists();
  renderSettingsHardware();
  renderSettingsSystem();
  renderSettingsTabs();
}

function setSwitch(id, on){
  const el = document.getElementById(id);
  if(el) setSwitchNode(el, on);
}
function setSwitchNode(el, on){
  el.classList.toggle("on", !!on);
}

async function loadSettings(){
  appSettings = await getJSON("/api/settings");
  ensureSelectionState();
  renderAll();
}

async function pollStatus(){
  try{
    liveStatus = await getJSON("/api/status");
    renderAll();
    maybeShowMatch();
  }catch(e){}
}

function maybeShowMatch(){
  if(!liveStatus || !liveStatus.latestMatch) return;
  const item = liveStatus.latestMatch;
  if(item.id === ui.lastPopupId) return;
  ui.lastPopupId = item.id;
  document.getElementById("popupNumber").textContent = item.number;
  document.getElementById("popupNumber").className = `match-number ${numClass(item.number)}`;
  document.getElementById("popupText").textContent = `${item.number} is in the selected list`;
  document.getElementById("matchPopup").classList.remove("hidden");
}

function closePopup(){
  document.getElementById("matchPopup").classList.add("hidden");
}

function toggleCameraVisual(){
  ui.cameraVisible = !ui.cameraVisible;
  renderCameraState();
}

async function toggleSound(){
  if(!appSettings) return;
  appSettings.hardware.buzzerEnabled = !appSettings.hardware.buzzerEnabled;
  await saveHardwareToServer(false);
}

async function toggleArm(){
  if(!appSettings) return;
  appSettings.hardware.actuatorEnabled = !appSettings.hardware.actuatorEnabled;
  await saveHardwareToServer(false);
}

async function saveHardwareToServer(reRender=true){
  appSettings.hardware = await postJSON("/api/hardware", appSettings.hardware);
  if(reRender) renderAll(); else renderAll();
}

async function testAction(){
  await postJSON("/api/test-action", {});
}

async function startScan(){
  if(!appSettings) return;
  liveStatus = await postJSON("/api/start", {selectedListId: appSettings.selectedListId});
  renderAll();
}

async function stopScan(){
  liveStatus = await postJSON("/api/stop", {});
  renderAll();
}

async function clearHistory(){
  await postJSON("/api/history/clear", {});
  await pollStatus();
}

function openSelectPanel(){
  document.getElementById("panelSelectList").classList.remove("hidden");
}

function openNumbersPanel(){
  ui.editorListId = appSettings.selectedListId;
  document.getElementById("panelNumbers").classList.remove("hidden");
  renderNumbersPanel();
}

function openSettingsPanel(){
  document.getElementById("panelSettings").classList.remove("hidden");
  showSettingsTab("lists");
}

function hidePanel(id){
  document.getElementById(id).classList.add("hidden");
}

async function selectActiveList(id){
  appSettings.selectedListId = id;
  await saveListsToServer(false);
  ui.selectedSettingsListId = id;
  ui.editorListId = id;
  renderAll();
}

function showSettingsTab(name){
  ui.settingsTab = name;
  renderSettingsTabs();
}

function selectSettingsList(id){
  ui.selectedSettingsListId = id;
  renderSettingsLists();
}

function editSelectedListNumbers(){
  ui.editorListId = ui.selectedSettingsListId || appSettings.selectedListId;
  document.getElementById("panelNumbers").classList.remove("hidden");
  renderNumbersPanel();
}

function clearSelectedListNumbers(){
  const list = getListById(ui.selectedSettingsListId);
  if(!list) return;
  list.numbers = [];
  renderSettingsLists();
  if(ui.editorListId === list.id) renderNumbersPanel();
}

function addList(){
  const idx = (appSettings.lists || []).length + 1;
  const newList = {
    id: "list-" + Math.random().toString(16).slice(2,10),
    name: "List " + idx,
    numbers: []
  };
  appSettings.lists.push(newList);
  ui.selectedSettingsListId = newList.id;
  ui.editorListId = newList.id;
  renderAll();
}

function renameSelectedList(){
  const list = getListById(ui.selectedSettingsListId);
  if(!list) return;
  const next = prompt("List name:", list.name);
  if(next && next.trim()){
    list.name = next.trim().slice(0,60);
    renderAll();
  }
}

function deleteSelectedList(){
  if(!ui.selectedSettingsListId) return;
  if((appSettings.lists || []).length <= 1){
    alert("At least one list must remain.");
    return;
  }
  appSettings.lists = appSettings.lists.filter(x => x.id !== ui.selectedSettingsListId);
  if(appSettings.selectedListId === ui.selectedSettingsListId){
    appSettings.selectedListId = appSettings.lists[0].id;
  }
  ui.selectedSettingsListId = appSettings.selectedListId;
  ui.editorListId = ui.selectedSettingsListId;
  renderAll();
}

function toggleEditorNumber(n){
  const list = getListById(ui.editorListId);
  if(!list) return;
  const idx = list.numbers.indexOf(n);
  if(idx >= 0) list.numbers.splice(idx,1);
  else list.numbers.push(n);
  list.numbers.sort((a,b)=>a-b);
  renderNumbersPanel();
  renderSettingsLists();
  if(currentList() && currentList().id === list.id) renderActiveList();
}

function clearEditorNumbers(){
  const list = getListById(ui.editorListId);
  if(!list) return;
  list.numbers = [];
  renderNumbersPanel();
  renderSettingsLists();
  if(currentList() && currentList().id === list.id) renderActiveList();
}

async function saveEditorNumbers(){
  await saveListsToServer(false);
  renderAll();
}

async function saveListsToServer(reRender=true){
  const payload = {
    selectedListId: appSettings.selectedListId,
    lists: appSettings.lists.map(l => ({
      id:l.id,
      name:l.name,
      numbers:(l.numbers || []).map(Number).filter(n => n>=0 && n<=36).sort((a,b)=>a-b)
    }))
  };
  appSettings = await postJSON("/api/lists", payload);
  ensureSelectionState();
  if(reRender) renderAll(); else renderAll();
}

function toggleHardwareField(name){
  appSettings.hardware[name] = !appSettings.hardware[name];
  renderSettingsHardware();
  renderHeader();
}

function escapeHtml(v){
  return String(v ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;");
}

function enableDragScroll(el){
  let isDown = false;
  let startX = 0;
  let scrollLeft = 0;

  el.addEventListener("mousedown", e => {
    isDown = true;
    startX = e.pageX - el.offsetLeft;
    scrollLeft = el.scrollLeft;
  });
  el.addEventListener("mouseleave", ()=> isDown = false);
  el.addEventListener("mouseup", ()=> isDown = false);
  el.addEventListener("mousemove", e => {
    if(!isDown) return;
    e.preventDefault();
    const x = e.pageX - el.offsetLeft;
    const walk = (x - startX) * 1.3;
    el.scrollLeft = scrollLeft - walk;
  });

  let touchX = 0;
  let touchScroll = 0;
  el.addEventListener("touchstart", e => {
    touchX = e.touches[0].clientX;
    touchScroll = el.scrollLeft;
  }, {passive:true});
  el.addEventListener("touchmove", e => {
    const walk = (e.touches[0].clientX - touchX) * 1.25;
    el.scrollLeft = touchScroll - walk;
  }, {passive:true});
}

async function boot(){
  await loadSettings();
  await pollStatus();
  enableDragScroll(document.getElementById("historyStrip"));
  setInterval(pollStatus, 350);
}
boot();
</script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    hub = None

    def log_message(self, fmt, *args):
        return

    def _json_body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/settings":
            data = self.hub.store.get()
            data["hardwareStatus"] = self.hub.actions.status()
            self.send_json(data)

        elif path == "/api/status":
            self.send_json(self.hub.public_status())

        elif path == "/stream.mjpg":
            self.stream("frame")

        elif path == "/roi.mjpg":
            self.stream("roi")

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            if path == "/api/lists":
                data = self.hub.store.update_lists(self._json_body())
                self.hub.selected_list = self.hub.store.selected_list()
                self.send_json(data)

            elif path == "/api/hardware":
                saved = self.hub.store.update_hardware(self._json_body())
                self.hub.actions.update_config(saved)
                self.send_json(saved)

            elif path == "/api/camera":
                body = self._json_body()
                self.send_json(self.hub.set_camera_enabled(bool(body.get("enabled", True))))

            elif path == "/api/start":
                body = self._json_body()
                self.send_json(self.hub.start_monitoring(body.get("selectedListId")))

            elif path == "/api/stop":
                self.send_json(self.hub.stop_monitoring())

            elif path == "/api/history/clear":
                self.send_json(self.hub.clear_history())

            elif path == "/api/test-action":
                self.send_json(self.hub.actions.trigger_async(number=None, list_name="manual", reason="manual_test"))

            else:
                self.send_error(404)

        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)

    def stream(self, mode):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            frame = self.hub.get_roi() if mode == "roi" else self.hub.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            if mode == "roi":
                frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

            jpg = encode_jpeg(frame)

            if not jpg:
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.06)
            except Exception:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    cfg.setdefault("final_app", {
        "min_stable_frames": 2,
        "min_live_score": 0.70,
        "min_live_margin": 0.10,
        "min_confirm_score": 0.85,
        "min_confirm_margin": 0.20,
    })

    hub = FinalVisionHub(cfg, source=args.source)
    Handler.hub = hub
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"Roulette final app: http://192.168.1.251:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
