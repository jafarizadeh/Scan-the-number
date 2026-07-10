import argparse
import csv
import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_rect_from_shape, roi_metrics
from .cnn_number_model import CNNNumberModel
from .hardware_actions import HardwareActionController, DEFAULT_HARDWARE_CONFIG, clean_hardware_config

APP_SETTINGS_PATH = Path("data/app_settings.json")

DEFAULT_TARGET_LISTS = {
    "selectedListId": "list-1",
    "lists": [
        {"id": "list-1", "name": "List 1", "numbers": [0, 8, 10, 11, 13, 20, 29, 35, 36]},
    ],
    "hardware": dict(DEFAULT_HARDWARE_CONFIG),
}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def clean_numbers(values):
    out = []
    seen = set()
    for value in values or []:
        try:
            n = int(value)
        except Exception:
            continue
        if 0 <= n <= 36 and n not in seen:
            out.append(n)
            seen.add(n)
    return sorted(out)


def clean_lists(payload):
    raw_lists = payload.get("lists", []) if isinstance(payload, dict) else []
    lists = []
    for idx, item in enumerate(raw_lists):
        if not isinstance(item, dict):
            continue
        lid = str(item.get("id") or f"list-{uuid.uuid4().hex[:8]}")
        name = str(item.get("name") or f"List {idx + 1}").strip()[:60]
        nums = clean_numbers(item.get("numbers", []))
        lists.append({"id": lid, "name": name, "numbers": nums})
    if not lists:
        lists = list(DEFAULT_TARGET_LISTS["lists"])
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
        data = dict(DEFAULT_TARGET_LISTS)
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text())
                data.update(raw)
        except Exception as exc:
            print(f"[APP] settings load failed: {exc}")
        clean = clean_lists(data)
        clean["hardware"] = clean_hardware_config(data.get("hardware", {}))
        return clean

    def save(self):
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
            self.save()
            return json.loads(json.dumps(self.data))

    def update_hardware(self, payload):
        clean = clean_hardware_config(payload)
        with self.lock:
            self.data["hardware"] = clean
            self.save()
            return dict(clean)

    def selected_list(self):
        data = self.get()
        selected = data.get("selectedListId")
        for item in data.get("lists", []):
            if item.get("id") == selected:
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
        self.events = deque(maxlen=100)

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)
        self.event_log = Path("data/logs/final_events.csv")
        if not self.event_log.exists():
            with self.event_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["id", "timestamp", "number", "score", "margin", "list_id", "list_name", "is_match", "action_accepted", "roi_path"])

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

    def start_monitoring(self, selected_list_id=None):
        data = self.store.get()
        if selected_list_id:
            data["selectedListId"] = selected_list_id
            self.store.update_lists(data)
        self.selected_list = self.store.selected_list()
        with self.lock:
            self.monitoring = True
            self.session_id = f"session-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.last_output_number = None
            self.reset_candidate()
            self.latest_match = None
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
        white_digit_ratio = float(((v >= 145) & (s <= 130)).mean())
        green_ratio = float(((h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)).mean())
        red_ratio = float(((((h <= 12) | (h >= 165)) & (s >= 45) & (v >= 45))).mean())
        if white_digit_ratio >= 0.003 or green_ratio >= 0.03 or red_ratio >= 0.03 or metrics["std"] >= 18.0:
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
            action_result = self.actions.trigger_async(number=number, list_name=selected.get("name", ""), reason="number_in_selected_list")
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
            csv.writer(f).writerow([event_id, event["timestamp"], number, round(score, 4), round(margin, 4), selected.get("id"), selected.get("name"), is_match, action_result.get("accepted"), roi_path])
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
        if avg_score >= float(cfg.get("min_confirm_score", 0.85)) and avg_margin >= float(cfg.get("min_confirm_margin", 0.20)):
            event = self.record_number(label, avg_score, avg_margin)
        else:
            event = None
        self.reset_candidate()
        return event

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)
        cfg = self.cfg.get("final_app", {})
        min_stable_frames = int(cfg.get("min_stable_frames", 2))
        min_live_score = float(cfg.get("min_live_score", 0.70))
        min_live_margin = float(cfg.get("min_live_margin", 0.10))
        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.03)
                continue
            roi = crop_relative_roi(frame, self.cfg["roi"])
            metrics = roi_metrics(roi)
            state = self.quick_state(roi, metrics)
            decision = None
            event = None
            if self.monitoring:
                try:
                    decision = self.model.predict(roi)
                    label = int(decision["label"])
                    score = float(decision["score"])
                    margin = float(decision["margin"])
                    self.latest_decision = decision
                    usable = state == "RESULT_VISIBLE" and score >= min_live_score and margin >= min_live_margin
                    if usable:
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
                            if self.candidate_streak >= min_stable_frames:
                                event = self.finalize_candidate()
                    else:
                        self.reset_candidate()
                except Exception as exc:
                    state = f"ERROR: {exc}"
                    self.reset_candidate()
            status = {
                "ready": True,
                "monitoring": self.monitoring,
                "sessionId": self.session_id,
                "state": state,
                "selectedList": self.selected_list,
                "decision": decision or self.latest_decision,
                "candidateLabel": self.candidate_label,
                "candidateStreak": self.candidate_streak,
                "lastOutputNumber": self.last_output_number,
                "latestMatch": self.latest_match,
                "history": list(self.history)[:200],
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
        if frame is not None:
            x, y, w, h = roi_rect_from_shape(frame.shape, self.cfg["roi"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame

    def get_roi(self):
        with self.lock:
            return None if self.latest_roi is None else self.latest_roi.copy()

    def clear_history(self):
        with self.lock:
            self.history.clear()
            self.latest_match = None
        return {"ok": True}


def encode_jpeg(frame, quality=85):
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
  --bg:#eef4fa;
  --panel:#ffffff;
  --soft:#f7fbff;
  --line:#cfe0ef;
  --text:#0f2944;
  --muted:#56708c;
  --green:#25bf6b;
  --red:#e43a47;
  --black:#182333;
  --blue:#2f80ed;
  --gold:#ffcc33;
  --shadow:0 5px 15px rgba(35,60,90,.10);
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
body{padding:8px}
.app{
  width:784px;
  height:464px;
  display:grid;
  grid-template-rows:48px 205px 195px;
  gap:8px;
}

/* TOP */
.topbar{
  background:rgba(255,255,255,.86);
  border:1px solid var(--line);
  border-radius:17px;
  box-shadow:var(--shadow);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 10px;
}
.brand{
  display:flex;
  align-items:center;
  gap:10px;
}
.logoBox{
  width:36px;
  height:36px;
  border-radius:50%;
  background:#fff;
  border:1px dashed #8fb3d8;
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:hidden;
}
.logoBox img{
  width:100%;
  height:100%;
  object-fit:contain;
}
.logoFallback{
  font-size:9px;
  color:#6c97c1;
  font-weight:900;
  text-align:center;
  line-height:1;
}
.brandTitle{
  font-size:23px;
  font-weight:900;
  white-space:nowrap;
}
.topActions{
  display:flex;
  align-items:center;
  gap:7px;
}
.topBtn,.scanPill{
  height:36px;
  border-radius:13px;
  border:1px solid var(--line);
  background:#fff;
  color:var(--text);
  font-size:13px;
  font-weight:900;
  padding:0 11px;
  display:flex;
  align-items:center;
  gap:7px;
  box-shadow:0 3px 8px rgba(40,65,95,.07);
}
.topBtn{
  cursor:pointer;
}
.scanPill{
  min-width:104px;
  background:#eef9f3;
  color:#159552;
}
.statusDot{
  width:9px;
  height:9px;
  border-radius:50%;
  background:var(--green);
}
.icon{
  width:19px;
  height:19px;
  stroke:currentColor;
  fill:none;
  stroke-width:2.2;
  stroke-linecap:round;
  stroke-linejoin:round;
}

/* MAIN GRID */
.mainGrid{
  display:grid;
  grid-template-columns:150px 218px 154px 238px;
  gap:8px;
}
.card{
  background:rgba(255,255,255,.88);
  border:1px solid var(--line);
  border-radius:18px;
  box-shadow:var(--shadow);
  overflow:hidden;
}
.cardTitle{
  font-size:11px;
  font-weight:900;
  color:#355579;
  letter-spacing:.7px;
  text-transform:uppercase;
  padding:9px 10px 0 10px;
}

/* CONTROLS */
.controls{
  padding:0 9px 9px 9px;
  display:flex;
  flex-direction:column;
  gap:7px;
}
.controlRow{
  height:42px;
  border:1px solid var(--line);
  border-radius:14px;
  background:var(--soft);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 9px;
}
.controlLeft{
  display:flex;
  align-items:center;
  gap:8px;
  font-size:14px;
  font-weight:900;
}
.controlLeft svg{
  width:20px;
  height:20px;
}
.switch{
  width:45px;
  height:25px;
  border-radius:18px;
  background:#b7c8d9;
  position:relative;
  cursor:pointer;
}
.switch:after{
  content:"";
  position:absolute;
  width:21px;
  height:21px;
  top:2px;
  left:2px;
  border-radius:50%;
  background:#fff;
  box-shadow:0 2px 6px rgba(0,0,0,.22);
  transition:.16s;
}
.switch.on{
  background:var(--green);
}
.switch.on:after{
  left:22px;
}
.statusBox{
  flex:1;
  min-height:55px;
  border:1px solid var(--line);
  border-radius:15px;
  background:var(--soft);
  padding:9px;
}
.statusLabel{
  display:flex;
  align-items:center;
  gap:7px;
  color:#355579;
  font-size:11px;
  font-weight:900;
  text-transform:uppercase;
}
.statusValue{
  margin-top:8px;
  font-size:18px;
  line-height:1;
  font-weight:900;
}
.statusSub{
  margin-top:4px;
  color:var(--muted);
  font-size:11px;
}

/* CAMERA */
.cameraPanel{
  padding:0 9px 9px 9px;
  display:flex;
  flex-direction:column;
}
.cameraFrame{
  position:relative;
  height:125px;
  margin-top:8px;
  border-radius:14px;
  overflow:hidden;
  background:#05090e;
  border:1px solid #203449;
}
.cameraFrame img{
  width:100%;
  height:100%;
  object-fit:cover;
  display:block;
}
.liveBadge{
  position:absolute;
  top:7px;
  left:7px;
  background:#118847;
  color:#fff;
  padding:4px 8px;
  font-size:11px;
  font-weight:900;
  border-radius:8px;
}
.cameraButtons{
  margin-top:8px;
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:8px;
}
.scanBtn{
  height:38px;
  border:0;
  border-radius:12px;
  color:#fff;
  font-size:14px;
  font-weight:900;
  cursor:pointer;
  box-shadow:0 4px 10px rgba(35,60,90,.12);
}
.startBtn{background:linear-gradient(180deg,#2dca77,#19ad61)}
.stopBtn{background:linear-gradient(180deg,#ef4652,#d92d3a)}

/* INFO */
.infoPanel{
  padding:0 9px 9px 9px;
  display:flex;
  flex-direction:column;
  gap:7px;
}
.infoBox{
  flex:1;
  border:1px solid var(--line);
  border-radius:13px;
  background:var(--soft);
  padding:9px;
  display:flex;
  flex-direction:column;
  justify-content:center;
}
.infoLabel{
  font-size:11px;
  font-weight:900;
  color:#355579;
  text-transform:uppercase;
}
.infoVal{
  margin-top:6px;
  font-size:16px;
  font-weight:900;
}
.infoVal.green{color:#159552}
.lastNumber{
  align-self:flex-end;
  min-width:54px;
  height:36px;
  border-radius:10px;
  background:linear-gradient(180deg,#273549,#101923);
  color:#fff;
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:18px;
  font-weight:900;
  margin-top:-4px;
}

/* ACTIVE LIST */
.listPanel{
  padding:0 10px 9px 10px;
  display:flex;
  flex-direction:column;
}
.listTop{
  margin-top:8px;
  display:grid;
  grid-template-columns:1fr 88px;
  gap:7px;
}
.fakeSelect{
  height:38px;
  border:1px solid var(--line);
  border-radius:12px;
  background:var(--soft);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 10px;
  font-size:15px;
  font-weight:900;
}
.selectBtn{
  height:38px;
  border-radius:12px;
  border:1px solid #62a0ff;
  background:#fff;
  color:#1d68ea;
  font-size:12px;
  font-weight:900;
}
.listMeta{
  margin:8px 0 6px 0;
  font-size:12px;
  color:#355579;
  font-weight:900;
}
.numRow{
  display:flex;
  flex-wrap:wrap;
  gap:6px;
}
.divider{
  height:1px;
  background:#d9e5ef;
  margin:8px 0 6px 0;
}
.quickTitle{
  color:#355579;
  font-size:11px;
  font-weight:900;
  text-transform:uppercase;
  margin-bottom:6px;
}

/* NUMBER CHIPS */
.numChip{
  min-width:36px;
  height:29px;
  padding:0 8px;
  border-radius:9px;
  color:#fff;
  font-size:14px;
  font-weight:900;
  display:flex;
  align-items:center;
  justify-content:center;
  border:2px solid rgba(255,255,255,.55);
  box-shadow:0 3px 8px rgba(20,40,65,.14), inset 0 1px 0 rgba(255,255,255,.22);
  position:relative;
}
.numRed{background:linear-gradient(180deg,#f24a56,#dc303d)}
.numBlack{background:linear-gradient(180deg,#263449,#101923)}
.numGreen{background:linear-gradient(180deg,#34cc76,#1bae61)}
.numMatch{
  outline:3px solid var(--gold);
  box-shadow:0 0 0 4px rgba(255,204,51,.22);
}
.numMatch:after{
  content:"★";
  position:absolute;
  top:-8px;
  right:1px;
  color:var(--gold);
  font-size:11px;
}

/* HISTORY */
.historyPanel{
  padding:9px 10px 10px 10px;
  display:flex;
  flex-direction:column;
}
.historyHead{
  height:20px;
  display:flex;
  align-items:center;
  justify-content:space-between;
}
.historyTitle{
  color:#355579;
  font-size:12px;
  font-weight:900;
  text-transform:uppercase;
}
.historyHint{
  color:#5d7693;
  font-size:11px;
  font-weight:800;
}
.historyGrid{
  flex:1;
  overflow-x:auto;
  overflow-y:hidden;
  scrollbar-width:none;
  display:grid;
  grid-auto-flow:column;
  grid-template-rows:repeat(5, 1fr);
  grid-auto-columns:56px;
  gap:5px 12px;
  padding:5px 2px 2px 2px;
  touch-action:pan-x;
}
.historyGrid::-webkit-scrollbar{display:none}
.histItem{
  width:56px;
  height:30px;
  display:flex;
  flex-direction:column;
  align-items:center;
}
.histItem .numChip{
  width:54px;
  min-width:54px;
  height:22px;
  font-size:12px;
  border-radius:7px;
}
.timeLabel{
  margin-top:1px;
  font-size:8.5px;
  line-height:1;
  color:#607894;
  font-weight:800;
}

/* MODAL */
.modal{
  position:fixed;
  inset:0;
  background:rgba(40,60,80,.26);
  display:none;
  align-items:center;
  justify-content:center;
  z-index:30;
}
.modal.show{display:flex}
.modalBox{
  background:#fff;
  border:1px solid var(--line);
  border-radius:18px;
  box-shadow:0 20px 60px rgba(30,50,80,.25);
  width:430px;
  max-height:410px;
  overflow:hidden;
}
.modalHead{
  height:46px;
  border-bottom:1px solid var(--line);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 12px;
}
.modalHead h2{
  margin:0;
  font-size:18px;
}
.closeBtn{
  height:32px;
  padding:0 12px;
  border-radius:10px;
  border:1px solid var(--line);
  background:#fff;
  font-weight:900;
}
.modalBody{
  padding:12px;
  max-height:360px;
  overflow:auto;
}
.listChoice{
  width:100%;
  min-height:58px;
  border:1px solid var(--line);
  background:#fff;
  border-radius:14px;
  margin-bottom:8px;
  padding:8px 10px;
  display:flex;
  align-items:center;
  justify-content:space-between;
}
.listChoice.active{
  border:2px solid var(--blue);
  box-shadow:0 0 0 3px rgba(47,128,237,.16);
}
.listChoiceName{
  font-size:17px;
  font-weight:900;
}
.listChoiceMeta{
  margin-top:3px;
  color:var(--muted);
  font-size:12px;
  font-weight:800;
}
.popupBox{
  width:320px;
  text-align:center;
  padding:22px;
}
.popupTitle{
  color:var(--green);
  font-size:20px;
  font-weight:900;
}
.popupNumber{
  width:100px;
  height:78px;
  margin:14px auto;
  border-radius:17px;
  color:#fff;
  font-size:46px;
  font-weight:900;
  display:flex;
  align-items:center;
  justify-content:center;
}
</style>
</head>

<body>
<div class="app">

  <header class="topbar">
    <div class="brand">
      <div class="logoBox">
        <img src="/assets/logo.png" onerror="this.style.display='none';document.getElementById('logoFallback').style.display='block'">
        <div id="logoFallback" class="logoFallback">logo<br>path</div>
      </div>
      <div class="brandTitle">Roulette Vision</div>
    </div>

    <div class="topActions">
      <button class="topBtn" onclick="toggleSound()">
        <svg class="icon" viewBox="0 0 24 24">
          <path d="M11 5 6 9H3v6h3l5 4V5z"></path>
          <path d="M15.5 8.5a5 5 0 0 1 0 7"></path>
        </svg>
        Sound
      </button>

      <button class="topBtn" onclick="toggleArm()">
        <svg class="icon" viewBox="0 0 24 24">
          <path d="M8 20h8"></path>
          <path d="M12 4v7"></path>
          <path d="M6 14l4-4"></path>
          <path d="M14 14 20 8"></path>
          <path d="M17 5h3v3"></path>
        </svg>
        Arm
      </button>

      <div class="scanPill" id="scanPill">
        <span class="statusDot"></span>
        <span>Stopped</span>
      </div>

      <button class="settingsBtn topBtn" onclick="openSettings()">
        <svg class="icon" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.6-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1A2 2 0 1 1 7.1 4l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 2.9V3a2 2 0 1 1 4 0v-.1a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1A2 2 0 1 1 20 6.9l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"></path>
        </svg>
        Settings
      </button>
    </div>
  </header>

  <main class="mainGrid">

    <section class="card controls-col">
      <div class="panel-title">Controls</div>

      <div class="toggle-card">
        <div class="toggle-left">
          <svg class="icon" viewBox="0 0 24 24"><path d="M4 7h16v12H4z"></path><path d="M9 7l1.5-3h3L15 7"></path><circle cx="12" cy="13" r="3.3"></circle></svg>
          Camera
        </div>
        <div id="cameraSwitch" class="switch on" onclick="toggleCamera()"></div>
      </div>

      <div class="toggle-card">
        <div class="toggle-left">
          <svg class="icon" viewBox="0 0 24 24"><path d="M11 5 6 9H3v6h3l5 4V5z"></path><path d="M15.5 8.5a5 5 0 0 1 0 7"></path></svg>
          Sound
        </div>
        <div id="soundSwitch" class="switch" onclick="toggleSound()"></div>
      </div>

      <div class="toggle-card">
        <div class="toggle-left">
          <svg class="icon" viewBox="0 0 24 24"><path d="M8 20h8"></path><path d="M12 4v7"></path><path d="M6 14l4-4"></path><path d="M14 14 20 8"></path></svg>
          Arm
        </div>
        <div id="armSwitch" class="switch" onclick="toggleArm()"></div>
      </div>

      <div class="status-card">
        <div class="status-line"><span class="dot" id="statusDot"></span>Status</div>
        <div id="statusValue" class="status-value">STOPPED</div>
        <div id="statusSub" class="status-text">System is stopped</div>
      </div>
    </section>

    <section class="card camera-panel">
      <div class="panel-title">Live Camera</div>

      <div class="camera-frame" id="cameraFrame">
        <img id="cameraImg" src="/stream.mjpg">
        <div class="live-badge">LIVE</div>
      </div>

      <div class="camera-buttons">
        <button class="btn btn-start" onclick="startScan()">▶ START</button>
        <button class="btn btn-stop" onclick="stopScan()">■ STOP</button>
      </div>
    </section>

    <section class="card info-panel">
      <div class="panel-title">Current Info</div>

      <div class="info-card">
        <div class="info-label">Scan</div>
        <div id="scanState" class="info-value">STOP</div>
      </div>

      <div class="info-card">
        <div class="info-label">Last Number</div>
        <div id="lastNumber" class="last-number-box">--</div>
      </div>

      <div class="info-card">
        <div class="info-label">Last Match</div>
        <div id="lastMatch" class="info-value" style="font-size:22px">--</div>
      </div>
    </section>

    <section class="card list-panel">
      <div class="panel-title">Active List</div>

      <div class="list-top">
        <div class="list-select"><span id="listName">List 1</span><span class="chev">⌄</span></div>
        <button class="list-btn" onclick="openListModal()">Select</button>
      </div>

      <div id="listMeta" class="subtle">0 numbers</div>
      <div id="listNumbers" class="number-row"></div>

      <div class="divider"></div>

      <div class="quickTitle">Quick Preview</div>
      <div id="quickPreview" class="number-row"></div>
    </section>

  </main>

  <section class="card history-panel">
    <div class="history-top">
      <div class="history-title">History</div>
      <div class="history-note">Swipe left/right to scroll</div>
    </div>

    <div id="historyGrid" class="history-grid"></div>
  </section>

</div>

<div id="listModal" class="modal">
  <div class="modalBox">
    <div class="modalHead">
      <h2>Select Active List</h2>
      <button class="closeBtn" onclick="closeListModal()">Close</button>
    </div>
    <div id="listChoices" class="modalBody"></div>
  </div>
</div>

<div id="matchModal" class="modal">
  <div class="modalBox popupBox">
    <div class="popupTitle">TARGET FOUND</div>
    <div id="popupNumber" class="popupNumber">--</div>
    <div id="popupText">Number found in selected list</div>
    <br>
    <button class="closeBtn" onclick="closeMatchModal()">OK</button>
  </div>
</div>

<script>
let appSettings=null;
let cameraOn=true;
let lastMatchId=null;

const RED=new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);

function cls(n){
  n=Number(n);
  if(n===0) return "numGreen";
  return RED.has(n) ? "numRed" : "numBlack";
}

function chip(n, extra=""){
  return `<div class="numChip ${cls(n)} ${extra}">${n}</div>`;
}

function histItem(x){
  const t=(x.timestamp||"").split("T").pop().slice(0,8);
  return `
    <div class="histItem">
      <div class="numChip ${cls(x.number)} ${x.isMatch?'numMatch':''}">${x.number}</div>
      <div class="timeLabel">${t}</div>
    </div>`;
}

function currentList(){
  if(!appSettings || !appSettings.lists || !appSettings.lists.length) return null;
  return appSettings.lists.find(x=>x.id===appSettings.selectedListId) || appSettings.lists[0];
}

async function getJSON(url){
  return await (await fetch(url)).json();
}

async function postJSON(url,data={}){
  return await (await fetch(url,{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(data)
  })).json();
}

function setSwitch(id,on){
  const e=document.getElementById(id);
  if(e) e.classList.toggle("on",!!on);
}

async function loadSettings(){
  appSettings=await getJSON("/api/settings");
  renderSettings();
}

function renderSettings(){
  const hw=appSettings.hardware||{};
  setSwitch("soundSwitch",!!hw.buzzerEnabled);
  setSwitch("armSwitch",!!hw.actuatorEnabled);
  renderList();
  renderListChoices();
}

function renderList(){
  const l=currentList();
  if(!l) return;

  document.getElementById("listName").textContent=l.name;
  document.getElementById("listMeta").textContent=`${l.numbers.length} numbers`;
  document.getElementById("listNumbers").innerHTML=l.numbers.map(n=>chip(n)).join("");

  const quick=[0,1,2,3,4,5,6,7,8,9,10,11];
  document.getElementById("quickPreview").innerHTML=quick.map(n=>chip(n)).join("");
}

function renderListChoices(){
  if(!appSettings) return;
  document.getElementById("listChoices").innerHTML=appSettings.lists.map(l=>{
    return `
      <button class="listChoice ${l.id===appSettings.selectedListId?'active':''}" onclick="selectList('${l.id}')">
        <div>
          <div class="listChoiceName">${l.name}</div>
          <div class="listChoiceMeta">${l.numbers.length} numbers</div>
        </div>
        <div class="number-row">${l.numbers.slice(0,6).map(n=>chip(n)).join("")}</div>
      </button>`;
  }).join("");
}

async function selectList(id){
  appSettings.selectedListId=id;
  appSettings=await postJSON("/api/lists",appSettings);
  renderSettings();
  closeListModal();
}

function openListModal(){
  renderListChoices();
  document.getElementById("listModal").classList.add("show");
}

function closeListModal(){
  document.getElementById("listModal").classList.remove("show");
}

async function toggleSound(){
  if(!appSettings) return;
  appSettings.hardware.buzzerEnabled=!appSettings.hardware.buzzerEnabled;
  appSettings.hardware=await postJSON("/api/hardware",appSettings.hardware);
  renderSettings();
}

async function toggleArm(){
  if(!appSettings) return;
  appSettings.hardware.actuatorEnabled=!appSettings.hardware.actuatorEnabled;
  appSettings.hardware=await postJSON("/api/hardware",appSettings.hardware);
  renderSettings();
}

function toggleCamera(){
  cameraOn=!cameraOn;
  setSwitch("cameraSwitch",cameraOn);
  document.getElementById("cameraFrame").style.opacity=cameraOn ? "1" : ".25";
}

async function startScan(){
  if(!appSettings) return;
  await postJSON("/api/start",{selectedListId:appSettings.selectedListId});
}

async function stopScan(){
  await postJSON("/api/stop");
}

function openSettings(){
  alert("Settings page will be connected in the next step.");
}

function updateStatus(st){
  const scanning=!!st.monitoring;

  document.getElementById("scanPill").innerHTML=scanning
    ? `<span class="statusDot"></span><span>Scanning</span>`
    : `<span class="statusDot" style="background:#9aacbf"></span><span>Stopped</span>`;

  document.getElementById("statusValue").textContent=scanning ? "SCANNING" : "STOPPED";
  document.getElementById("statusValue").style.color=scanning ? "#1a9f54" : "#db3242";
  document.getElementById("statusSub").textContent=scanning ? "System is active" : "System is stopped";

  document.getElementById("scanState").textContent=scanning ? "SCANNING" : "STOP";
  document.getElementById("scanState").style.color=scanning ? "#1a9f54" : "#db3242";

  document.getElementById("lastNumber").textContent=st.lastOutputNumber ?? "--";
  document.getElementById("lastMatch").textContent=st.latestMatch ? st.latestMatch.number : "--";

  document.getElementById("historyGrid").innerHTML=(st.history||[]).slice(0,100).map(histItem).join("");

  if(st.selectedList && appSettings && st.selectedList.id!==appSettings.selectedListId){
    appSettings.selectedListId=st.selectedList.id;
    renderList();
  }

  if(st.latestMatch && st.latestMatch.id!==lastMatchId){
    lastMatchId=st.latestMatch.id;
    showMatch(st.latestMatch.number);
  }
}

async function poll(){
  try{
    const st=await getJSON("/api/status");
    updateStatus(st);
  }catch(e){}
}

function showMatch(n){
  const e=document.getElementById("popupNumber");
  e.textContent=n;
  e.className=`popupNumber ${cls(n)}`;
  document.getElementById("popupText").textContent=`Number ${n} is in selected list`;
  document.getElementById("matchModal").classList.add("show");
  setTimeout(closeMatchModal,2200);
}

function closeMatchModal(){
  document.getElementById("matchModal").classList.remove("show");
}

function enableDragScroll(el){
  let down=false,startX=0,scrollLeft=0;

  el.addEventListener("mousedown",e=>{
    down=true;
    startX=e.pageX-el.offsetLeft;
    scrollLeft=el.scrollLeft;
  });

  el.addEventListener("mouseup",()=>down=false);
  el.addEventListener("mouseleave",()=>down=false);

  el.addEventListener("mousemove",e=>{
    if(!down) return;
    e.preventDefault();
    const x=e.pageX-el.offsetLeft;
    el.scrollLeft=scrollLeft-(x-startX)*1.2;
  });

  let touchX=0,touchScroll=0;

  el.addEventListener("touchstart",e=>{
    touchX=e.touches[0].clientX;
    touchScroll=el.scrollLeft;
  },{passive:true});

  el.addEventListener("touchmove",e=>{
    el.scrollLeft=touchScroll-(e.touches[0].clientX-touchX)*1.2;
  },{passive:true});
}

async function boot(){
  await loadSettings();
  await poll();
  enableDragScroll(document.getElementById("historyGrid"));
  setInterval(poll,350);
}

boot();
</script>
</body>
</html>
'''


class Handler(BaseHTTPRequestHandler):
    hub: FinalVisionHub = None

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
            self.end_headers(); self.wfile.write(body)
        elif path == "/api/settings":
            data = self.hub.store.get()
            data["hardwareStatus"] = self.hub.actions.status()
            self.send_json(data)
        elif path == "/api/status":
            self.send_json(self.hub.public_status())
        elif path.startswith("/assets/"):
            asset = Path("data/assets") / Path(path).name
            if asset.exists() and asset.is_file():
                body = asset.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)
        elif path == "/assets/logo.png":
            asset = Path("data/assets/logo.png")
            if not asset.exists():
                self.send_error(404)
                return
            data = asset.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
                self.send_json(self.hub.store.update_lists(self._json_body()))
                self.hub.selected_list = self.hub.store.selected_list()
            elif path == "/api/hardware":
                saved = self.hub.store.update_hardware(self._json_body())
                self.hub.actions.update_config(saved)
                self.send_json(saved)
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
                time.sleep(0.05); continue
            if mode == "roi":
                frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            jpg = encode_jpeg(frame)
            if not jpg:
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg + b"\r\n")
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
    cfg.setdefault("final_app", {"min_stable_frames": 2, "min_live_score": 0.70, "min_live_margin": 0.10, "min_confirm_score": 0.85, "min_confirm_margin": 0.20})
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
        server.server_close(); hub.close()

if __name__ == "__main__":
    main()
