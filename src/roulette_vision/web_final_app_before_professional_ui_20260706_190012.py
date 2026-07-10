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
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Roulette Vision</title>
<style>
:root{
  --bg:#edf3f8;
  --panel:#ffffff;
  --panel2:#f7fafc;
  --border:#c9d7e4;
  --text:#15283a;
  --muted:#6f8193;
  --blue:#2f80ed;
  --green:#1fbf75;
  --red:#d94452;
  --black:#26313b;
  --match:#ffcc33;
  --shadow:0 3px 10px rgba(30,55,80,.12);
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{
  margin:0;
  width:800px;
  height:480px;
  overflow:hidden;
  background:var(--bg);
  color:var(--text);
  font-family:Arial,Helvetica,sans-serif;
  user-select:none;
}
body{display:flex;flex-direction:column}
header{
  height:36px;
  flex:0 0 36px;
  background:#ffffff;
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 10px;
}
h1{margin:0;font-size:20px;font-weight:900;color:#10263a}
.headerBtns{display:flex;gap:6px}
button{
  border:1px solid #b8c9d8;
  background:#f8fbfd;
  color:#13283a;
  border-radius:11px;
  height:28px;
  padding:0 10px;
  font-size:13px;
  font-weight:900;
  cursor:pointer;
  box-shadow:0 1px 3px rgba(0,0,0,.06);
}
button.green{background:#1fbf75;color:white;border-color:#18a865}
button.red{background:#d94452;color:white;border-color:#c63745}
button.blue{background:#2f80ed;color:white;border-color:#2a72d3}
button.ghost{background:#fff}
button:active{transform:scale(.97)}
.main{
  height:444px;
  padding:7px;
  display:flex;
  flex-direction:column;
  gap:7px;
}
.top{
  height:214px;
  display:grid;
  grid-template-columns:145px 1fr 202px;
  gap:7px;
}
.panel{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:15px;
  box-shadow:var(--shadow);
  overflow:hidden;
}
.left{
  padding:8px;
  display:flex;
  flex-direction:column;
  gap:7px;
}
.title{
  font-size:11px;
  color:var(--muted);
  font-weight:900;
  text-transform:uppercase;
  letter-spacing:.5px;
}
.toggleRow{
  height:39px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  background:var(--panel2);
  border:1px solid #d5e0ea;
  border-radius:13px;
  padding:6px 8px;
}
.toggleRow label{font-size:14px;font-weight:900}
.switch{
  appearance:none;
  width:43px;
  height:24px;
  border-radius:20px;
  background:#b6c6d5;
  position:relative;
  outline:none;
}
.switch:checked{background:#23bf74}
.switch:after{
  content:"";
  position:absolute;
  width:18px;
  height:18px;
  top:3px;
  left:3px;
  background:#fff;
  border-radius:50%;
  box-shadow:0 1px 4px rgba(0,0,0,.25);
  transition:.18s;
}
.switch:checked:after{left:22px}
.left .wideBtn{width:100%;height:27px}
.statusMini{
  flex:1;
  min-height:34px;
  background:#f7fafc;
  border:1px solid #d5e0ea;
  border-radius:13px;
  padding:6px 8px;
}
.statusMini .k{font-size:10px;color:var(--muted);font-weight:800}
.statusMini .v{font-size:17px;font-weight:900;margin-top:1px}

.center{
  padding:7px;
  display:grid;
  grid-template-columns:1fr 145px;
  gap:7px;
}
.cameraBox{
  position:relative;
  border-radius:13px;
  overflow:hidden;
  border:1px solid #c3d2df;
  background:#111;
}
.cameraBox img{
  width:100%;
  height:100%;
  display:block;
  object-fit:cover;
}
.badge{
  position:absolute;
  left:7px;
  top:7px;
  background:rgba(255,255,255,.90);
  border:1px solid #b9c7d3;
  color:#1a2e40;
  border-radius:9px;
  padding:3px 7px;
  font-size:11px;
  font-weight:900;
}
.liveSide{
  display:flex;
  flex-direction:column;
  gap:7px;
}
.liveCard{
  flex:1;
  background:#f7fafc;
  border:1px solid #d5e0ea;
  border-radius:13px;
  padding:8px;
  display:flex;
  flex-direction:column;
  justify-content:center;
}
.liveCard .k{
  color:var(--muted);
  font-size:10px;
  font-weight:900;
  margin-bottom:3px;
}
.liveCard .big{
  font-size:29px;
  font-weight:900;
  line-height:1;
}
.liveCard .mid{
  font-size:18px;
  font-weight:900;
}
.right{
  padding:8px;
  display:flex;
  flex-direction:column;
  gap:7px;
}
.listButton{
  height:36px;
  border-radius:13px;
  background:#f7fafc;
  border:1px solid #c8d7e4;
  padding:0 10px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  font-size:15px;
  font-weight:900;
}
.listInfo{
  font-size:11px;
  color:var(--muted);
  font-weight:800;
}
.numberGrid{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:5px;
  overflow:hidden;
}
.numTile{
  height:32px;
  border-radius:11px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:17px;
  font-weight:900;
  color:white;
  border:2px solid rgba(255,255,255,.9);
  box-shadow:0 2px 5px rgba(0,0,0,.12);
}
.num-red{background:var(--red)}
.num-black{background:var(--black)}
.num-green{background:var(--green)}
.targetActive{
  border-color:#ffffff;
  box-shadow:0 0 0 2px rgba(47,128,237,.25),0 2px 5px rgba(0,0,0,.12);
}

.historyPanel{
  flex:1;
  min-height:0;
  padding:8px;
  display:flex;
  flex-direction:column;
}
.histHead{
  height:24px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:5px;
}
.histHead h3{
  margin:0;
  font-size:18px;
  font-weight:900;
}
.hint{font-size:11px;color:var(--muted);font-weight:700}
.track{
  flex:1;
  min-height:0;
  overflow-x:auto;
  overflow-y:hidden;
  display:flex;
  flex-direction:row-reverse;
  align-items:flex-start;
  gap:7px;
  padding:2px 2px 7px 2px;
  touch-action:pan-x;
  scrollbar-width:none;
  -ms-overflow-style:none;
}
.track::-webkit-scrollbar{display:none}
.histItem{
  width:56px;
  min-width:56px;
  height:70px;
  border-radius:14px;
  border:2px solid rgba(255,255,255,.95);
  color:white;
  text-align:center;
  padding:5px 3px;
  box-shadow:0 3px 8px rgba(0,0,0,.13);
}
.histItem .n{
  font-size:25px;
  font-weight:900;
  line-height:1;
}
.histItem .t{
  font-size:9px;
  margin-top:4px;
  opacity:.9;
}
.histItem .tag{
  margin-top:3px;
  font-size:9px;
  font-weight:900;
}
.histMatch{
  border:3px solid var(--match)!important;
  box-shadow:0 0 0 3px rgba(255,204,51,.25),0 4px 12px rgba(255,204,51,.45);
}
.histMatch .tag{color:var(--match)}

.modal{
  position:fixed;
  inset:0;
  display:none;
  z-index:50;
  background:rgba(30,50,70,.28);
  backdrop-filter:blur(3px);
  align-items:center;
  justify-content:center;
}
.modal.show{display:flex}
.modalBox{
  background:#fff;
  color:var(--text);
  border:1px solid #c7d7e4;
  border-radius:22px;
  box-shadow:0 18px 60px rgba(40,60,90,.30);
  overflow:hidden;
}
.modalHeader{
  height:44px;
  background:#f6f9fc;
  border-bottom:1px solid #d9e4ee;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 12px;
}
.modalHeader h2{margin:0;font-size:18px;font-weight:900}
.modalBody{padding:12px}
.listModalBox{width:500px;max-height:430px}
.listChoiceGrid{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:9px;
  max-height:340px;
  overflow:auto;
  scrollbar-width:none;
}
.listChoiceGrid::-webkit-scrollbar{display:none}
.listChoice{
  min-height:70px;
  border-radius:16px;
  border:2px solid #d2dfeb;
  background:#f9fcff;
  padding:10px;
  text-align:left;
}
.listChoice.active{
  border-color:#2f80ed;
  box-shadow:0 0 0 3px rgba(47,128,237,.16);
}
.listChoice .name{font-size:17px;font-weight:900}
.listChoice .count{font-size:12px;color:var(--muted);margin-top:5px}
.listChoice .preview{font-size:12px;color:#34495e;margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.settingsBox{
  width:760px;
  height:450px;
}
.settingsLayout{
  height:406px;
  display:grid;
  grid-template-columns:210px 1fr 190px;
  gap:10px;
}
.settingsColumn{
  min-height:0;
  overflow:auto;
  scrollbar-width:none;
}
.settingsColumn::-webkit-scrollbar{display:none}
.listManagerCard{
  border:2px solid #d2dfeb;
  background:#f9fcff;
  border-radius:15px;
  padding:9px;
  margin-bottom:8px;
}
.listManagerCard.active{
  border-color:#2f80ed;
  box-shadow:0 0 0 3px rgba(47,128,237,.15);
}
.listManagerCard .name{font-size:15px;font-weight:900}
.listManagerCard .count{font-size:11px;color:var(--muted);margin-top:4px}
.editorTitle{
  font-size:15px;
  font-weight:900;
  margin-bottom:8px;
}
.numPickGrid{
  display:grid;
  grid-template-columns:repeat(6,1fr);
  gap:6px;
}
.pickTile{
  height:34px;
  border-radius:12px;
  border:2px solid #d5e0ea;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:16px;
  font-weight:900;
  color:white;
}
.pickTile.selected{
  border-color:#ffcc33;
  box-shadow:0 0 0 3px rgba(255,204,51,.35);
}
.hardwareCard{
  background:#f9fcff;
  border:1px solid #d5e0ea;
  border-radius:15px;
  padding:10px;
  margin-bottom:8px;
}
.hardwareCard label{
  display:flex;
  justify-content:space-between;
  align-items:center;
  font-size:13px;
  font-weight:900;
  margin-bottom:8px;
}
.popup{
  position:fixed;
  inset:0;
  display:none;
  align-items:center;
  justify-content:center;
  z-index:90;
  background:rgba(0,0,0,.35);
}
.popup.show{display:flex}
.popupBox{
  width:380px;
  background:#ffffff;
  border:4px solid #1fbf75;
  border-radius:28px;
  box-shadow:0 18px 70px rgba(31,191,117,.45);
  text-align:center;
  padding:22px;
}
.popupBox h2{margin:0;font-size:25px;color:#1a9f62}
.popupNumber{
  font-size:98px;
  font-weight:900;
  line-height:1;
  margin:5px 0;
}
.popupText{
  font-size:18px;
  font-weight:900;
}
</style>
</head>
<body>
<header>
  <h1>Roulette Vision</h1>
  <div class="headerBtns">
    <button onclick="clearHistory()">Clear</button>
    <button onclick="openSettings()">Settings</button>
  </div>
</header>

<div class="main">
  <section class="top">
    <div class="panel left">
      <div class="title">Controls</div>

      <div class="toggleRow">
        <label>Camera</label>
        <input id="cameraToggle" class="switch" type="checkbox" checked onchange="toggleCamera()">
      </div>

      <div class="toggleRow">
        <label>Sound</label>
        <input id="soundToggle" class="switch" type="checkbox" onchange="toggleSound()">
      </div>

      <div class="toggleRow">
        <label>Arm</label>
        <input id="armToggle" class="switch" type="checkbox" onchange="toggleArm()">
      </div>

      <button class="green wideBtn" onclick="startScan()">Start</button>
      <button class="red wideBtn" onclick="stopScan()">Stop</button>

      <div class="statusMini">
        <div class="k">Status</div>
        <div id="scanMini" class="v">STOP</div>
      </div>
    </div>

    <div class="panel center">
      <div class="cameraBox">
        <img id="stream" src="/stream.mjpg">
        <div id="badge" class="badge">READY</div>
      </div>

      <div class="liveSide">
        <div class="liveCard">
          <div class="k">Scan</div>
          <div id="scanBig" class="big">STOP</div>
        </div>
        <div class="liveCard">
          <div class="k">Last Number</div>
          <div id="lastNum" class="big">--</div>
        </div>
        <div class="liveCard">
          <div class="k">Last Match</div>
          <div id="lastMatch" class="mid">--</div>
        </div>
      </div>
    </div>

    <div class="panel right">
      <div class="title">Selected List</div>
      <button id="listButton" class="listButton" onclick="openListPopup()">
        <span id="listButtonText">List 1</span>
        <span>▸</span>
      </button>
      <div id="listInfo" class="listInfo">-</div>
      <div id="targetGrid" class="numberGrid"></div>
    </div>
  </section>

  <section class="panel historyPanel">
    <div class="histHead">
      <h3>History</h3>
      <div class="hint">Swipe numbers horizontally</div>
    </div>
    <div id="historyTrack" class="track"></div>
  </section>
</div>

<div id="listModal" class="modal">
  <div class="modalBox listModalBox">
    <div class="modalHeader">
      <h2>Select list</h2>
      <button onclick="closeListPopup()">Close</button>
    </div>
    <div class="modalBody">
      <div id="listChoiceGrid" class="listChoiceGrid"></div>
    </div>
  </div>
</div>

<div id="settingsModal" class="modal">
  <div class="modalBox settingsBox">
    <div class="modalHeader">
      <h2>Settings</h2>
      <div>
        <button class="green" onclick="saveLists()">Save</button>
        <button onclick="closeSettings()">Back</button>
      </div>
    </div>

    <div class="modalBody">
      <div class="settingsLayout">
        <div class="settingsColumn">
          <div class="editorTitle">Lists</div>
          <div id="settingsListCards"></div>
          <button class="blue" style="width:100%;margin-top:6px" onclick="addList()">Add List</button>
          <button class="red" style="width:100%;margin-top:6px" onclick="deleteSelectedList()">Delete List</button>
        </div>

        <div class="settingsColumn">
          <div id="editorTitle" class="editorTitle">Numbers</div>
          <div id="numPickGrid" class="numPickGrid"></div>
        </div>

        <div class="settingsColumn">
          <div class="editorTitle">Hardware</div>

          <div class="hardwareCard">
            <label>Dry run <input id="dryRun" class="switch" type="checkbox"></label>
            <label>Sound <input id="buzzerEnabled" class="switch" type="checkbox"></label>
            <label>Arm <input id="actuatorEnabled" class="switch" type="checkbox"></label>
            <button class="green" style="width:100%" onclick="saveHardware()">Save Hardware</button>
            <button style="width:100%;margin-top:7px" onclick="testAction()">Test Action</button>
          </div>

          <div class="hardwareCard">
            <div class="hint">GPIO defaults:</div>
            <div class="hint">Buzzer: GPIO17</div>
            <div class="hint">Arm: GPIO18</div>
            <br>
            <div id="msg" class="hint"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="popup" class="popup">
  <div class="popupBox">
    <h2>TARGET FOUND</h2>
    <div id="popupNumber" class="popupNumber">--</div>
    <div class="popupText">Buzzer + Arm + Popup</div>
  </div>
</div>

<script>
let appSettings=null;
let lastMatchId=null;
let editingListId=null;

const RED=new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

function rclass(n){
  n=Number(n);
  if(n===0) return 'num-green';
  return RED.has(n)?'num-red':'num-black';
}

function selectedObj(){
  if(!appSettings || !appSettings.lists || !appSettings.lists.length) return null;
  return appSettings.lists.find(x=>x.id===appSettings.selectedListId) || appSettings.lists[0];
}

function editingObj(){
  if(!appSettings || !appSettings.lists || !appSettings.lists.length) return null;
  return appSettings.lists.find(x=>x.id===editingListId) || selectedObj();
}

function numTile(n,active){
  return `<div class="numTile ${rclass(n)} ${active?'targetActive':''}">${n}</div>`;
}

function histTile(r){
  let t=(r.timestamp||'').split('T').pop().slice(0,8);
  return `<div class="histItem ${rclass(r.number)} ${r.isMatch?'histMatch':''}">
    <div class="n">${r.number}</div>
    <div class="t">${t}</div>
    <div class="tag">${r.isMatch?'MATCH':''}</div>
  </div>`;
}

async function loadSettings(){
  appSettings=await (await fetch('/api/settings')).json();
  editingListId=appSettings.selectedListId;
  renderAll();
}

function renderAll(){
  renderMainList();
  renderListPopup();
  renderSettingsLists();
  renderNumberEditor();
  renderHardware();
}

function renderMainList(){
  let l=selectedObj();
  if(!l) return;

  document.getElementById('listButtonText').textContent=l.name;
  document.getElementById('listInfo').textContent=`${l.name} | ${l.numbers.length} numbers`;
  document.getElementById('targetGrid').innerHTML=l.numbers.map(n=>numTile(n,true)).join('');
}

function renderListPopup(){
  let box=document.getElementById('listChoiceGrid');
  if(!appSettings || !box) return;

  box.innerHTML=(appSettings.lists||[]).map(l=>{
    let active=l.id===appSettings.selectedListId;
    let preview=(l.numbers||[]).join(', ');
    return `<button class="listChoice ${active?'active':''}" onclick="selectMainList('${l.id}')">
      <div class="name">${l.name}</div>
      <div class="count">${l.numbers.length} numbers</div>
      <div class="preview">${preview}</div>
    </button>`;
  }).join('');
}

function renderSettingsLists(){
  let box=document.getElementById('settingsListCards');
  if(!appSettings || !box) return;

  box.innerHTML=(appSettings.lists||[]).map(l=>{
    let active=l.id===editingListId;
    return `<button class="listManagerCard ${active?'active':''}" style="width:100%;text-align:left" onclick="editList('${l.id}')">
      <div class="name">${l.name}</div>
      <div class="count">${l.numbers.length} numbers</div>
    </button>`;
  }).join('');
}

function renderNumberEditor(){
  let l=editingObj();
  let box=document.getElementById('numPickGrid');
  if(!l || !box) return;

  editingListId=l.id;
  document.getElementById('editorTitle').textContent=`Numbers in ${l.name}`;

  let set=new Set((l.numbers||[]).map(Number));
  let html='';

  for(let n=0;n<=36;n++){
    html += `<button class="pickTile ${rclass(n)} ${set.has(n)?'selected':''}" onclick="toggleNumber(${n})">${n}</button>`;
  }

  box.innerHTML=html;
}

function renderHardware(){
  let h=(appSettings&&appSettings.hardware)||{};
  document.getElementById('soundToggle').checked=!!h.buzzerEnabled;
  document.getElementById('armToggle').checked=!!h.actuatorEnabled;

  document.getElementById('dryRun').checked=!!h.dryRun;
  document.getElementById('buzzerEnabled').checked=!!h.buzzerEnabled;
  document.getElementById('actuatorEnabled').checked=!!h.actuatorEnabled;
}

function openListPopup(){
  renderListPopup();
  document.getElementById('listModal').classList.add('show');
}

function closeListPopup(){
  document.getElementById('listModal').classList.remove('show');
}

async function selectMainList(id){
  appSettings.selectedListId=id;
  await saveListsSilent();
  closeListPopup();
}

function openSettings(){
  editingListId=appSettings.selectedListId;
  renderAll();
  document.getElementById('settingsModal').classList.add('show');
}

function closeSettings(){
  document.getElementById('settingsModal').classList.remove('show');
  loadSettings();
}

function editList(id){
  editingListId=id;
  renderSettingsLists();
  renderNumberEditor();
}

function toggleNumber(n){
  let l=editingObj();
  if(!l) return;

  let set=new Set((l.numbers||[]).map(Number));

  if(set.has(n)) set.delete(n);
  else set.add(n);

  l.numbers=[...set].sort((a,b)=>a-b);
  renderNumberEditor();

  if(l.id===appSettings.selectedListId) renderMainList();
}

async function addList(){
  let idx=(appSettings.lists||[]).length+1;
  let l={
    id:'list-'+Math.random().toString(16).slice(2,10),
    name:'List '+idx,
    numbers:[]
  };
  appSettings.lists.push(l);
  editingListId=l.id;
  appSettings.selectedListId=l.id;
  await saveListsSilent();
}

async function deleteSelectedList(){
  if(!editingListId) return;
  if((appSettings.lists||[]).length<=1){
    document.getElementById('msg').textContent='At least one list is required';
    return;
  }

  appSettings.lists=appSettings.lists.filter(x=>x.id!==editingListId);

  if(appSettings.selectedListId===editingListId){
    appSettings.selectedListId=appSettings.lists[0].id;
  }

  editingListId=appSettings.selectedListId;
  await saveListsSilent();
}

async function saveListsSilent(){
  appSettings=await (await fetch('/api/lists',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(appSettings)
  })).json();

  if(!editingListId) editingListId=appSettings.selectedListId;

  renderAll();
}

async function saveLists(){
  await saveListsSilent();
  document.getElementById('msg').textContent='Lists saved';
}

async function saveHardware(){
  let h={...(appSettings.hardware||{})};
  h.dryRun=document.getElementById('dryRun').checked;
  h.buzzerEnabled=document.getElementById('buzzerEnabled').checked;
  h.actuatorEnabled=document.getElementById('actuatorEnabled').checked;

  appSettings.hardware=await (await fetch('/api/hardware',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(h)
  })).json();

  renderHardware();
  document.getElementById('msg').textContent='Hardware saved';
}

async function saveHardwarePartial(patch){
  let h={...(appSettings.hardware||{}),...patch};

  appSettings.hardware=await (await fetch('/api/hardware',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(h)
  })).json();

  renderHardware();
}

async function toggleSound(){
  await saveHardwarePartial({buzzerEnabled:document.getElementById('soundToggle').checked});
}

async function toggleArm(){
  await saveHardwarePartial({actuatorEnabled:document.getElementById('armToggle').checked});
}

async function toggleCamera(){
  await fetch('/api/camera',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:document.getElementById('cameraToggle').checked})
  });
}

async function testAction(){
  await fetch('/api/test-action',{method:'POST'});
  document.getElementById('msg').textContent='Test action sent';
}

async function startScan(){
  await fetch('/api/start',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({selectedListId:appSettings.selectedListId})
  });
}

async function stopScan(){
  await fetch('/api/stop',{method:'POST'});
}

async function clearHistory(){
  await fetch('/api/history/clear',{method:'POST'});
}

function popup(n,id){
  if(id===lastMatchId) return;

  lastMatchId=id;
  document.getElementById('popupNumber').textContent=n;

  let p=document.getElementById('popup');
  p.classList.add('show');

  setTimeout(()=>p.classList.remove('show'),2300);
}

async function poll(){
  try{
    let st=await (await fetch('/api/status')).json();

    document.getElementById('cameraToggle').checked=!!st.cameraEnabled;

    let scan=st.monitoring?'SCAN':'STOP';
    document.getElementById('scanMini').textContent=scan;
    document.getElementById('scanBig').textContent=scan;
    document.getElementById('badge').textContent=st.state||'READY';

    document.getElementById('lastNum').textContent=(st.history&&st.history.length)?st.history[0].number:'--';
    document.getElementById('lastMatch').textContent=st.latestMatch?st.latestMatch.number:'--';

    document.getElementById('historyTrack').innerHTML=(st.history||[]).slice(0,80).map(histTile).join('');

    if(st.selectedList && appSettings && st.selectedList.id!==appSettings.selectedListId){
      appSettings.selectedListId=st.selectedList.id;
      renderMainList();
      renderListPopup();
    }

    if(st.latestMatch) popup(st.latestMatch.number,st.latestMatch.id);
  }catch(e){}
}

loadSettings();
setInterval(poll,300);
poll();
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
