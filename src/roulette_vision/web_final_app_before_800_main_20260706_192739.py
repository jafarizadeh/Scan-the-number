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
<meta name="viewport" content="width=1500, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Roulette Vision</title>
<style>
:root{--bg:#f3f7fb;--card:#fff;--line:#d8e4ef;--line2:#c8d7e6;--text:#11283d;--muted:#647b92;--blue:#2f80ed;--green:#1fb86f;--red:#dd2f3e;--black:#182431;--shadow:0 8px 24px rgba(31,58,88,.10);--gold:#ffcc33}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}html,body{margin:0;width:100%;height:100%;background:linear-gradient(180deg,#fbfdff,#eef4f9);color:var(--text);font-family:Arial,Helvetica,sans-serif;overflow:hidden;user-select:none}body{padding:14px 22px}button{font-family:inherit;cursor:pointer}.app{width:100%;height:100%;display:flex;flex-direction:column;gap:10px}.header{height:50px;display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:12px;font-size:27px;font-weight:900}.logo{width:36px;height:36px;border-radius:50%;object-fit:cover}.logoFallback{width:36px;height:36px;border-radius:50%;display:none;align-items:center;justify-content:center;border:2px dashed #94b3d4;color:#4774a5;font-size:10px;font-weight:900;text-align:center;line-height:1}.headerActions{display:flex;align-items:center;gap:14px}.topBtn,.statusPill{height:42px;padding:0 18px;border-radius:13px;border:1px solid var(--line2);background:#fff;color:var(--text);font-size:14px;font-weight:900;display:flex;align-items:center;gap:9px;box-shadow:0 3px 10px rgba(31,58,88,.06)}.statusPill{min-width:140px;justify-content:center;background:#ecf8f2;color:#0b8f50}.statusDot{width:10px;height:10px;border-radius:50%;background:var(--green)}.settingsBtn{min-width:150px}.icon{width:22px;height:22px;stroke:currentColor;fill:none;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}.topGrid{height:360px;display:grid;grid-template-columns:278px 1.12fr 250px 465px;gap:10px}.card{background:rgba(255,255,255,.86);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow);overflow:hidden}.cardTitle{padding:12px 14px 0;font-size:13px;font-weight:900;color:#425d78;letter-spacing:.8px;text-transform:uppercase}.controls{padding:0 13px 13px}.controlRow{height:56px;margin-top:12px;padding:0 14px;border:1px solid var(--line);border-radius:14px;background:linear-gradient(180deg,#fff,#f7fbff);display:flex;align-items:center;justify-content:space-between}.controlLeft{display:flex;align-items:center;gap:13px;font-size:15px;font-weight:900}.controlIcon{width:26px;height:26px;color:#20354b}.switch{width:50px;height:26px;border-radius:20px;background:#b9cadb;position:relative;transition:.18s}.switch:after{content:"";position:absolute;width:22px;height:22px;border-radius:50%;background:#fff;top:2px;left:2px;box-shadow:0 2px 8px rgba(0,0,0,.25);transition:.18s}.switch.on{background:var(--green)}.switch.on:after{left:26px}.statusBox{margin-top:13px;height:100px;border:1px solid var(--line);border-radius:15px;background:linear-gradient(180deg,#fff,#f7fbff);padding:13px}.statusLabel{font-size:13px;color:#425d78;font-weight:900;text-transform:uppercase;display:flex;align-items:center;gap:8px}.statusText{margin-top:14px;font-size:21px;font-weight:900;color:var(--green)}.statusSub{margin-top:6px;color:#4d657d;font-size:13px}.cameraCard{padding:0 14px 14px}.cameraBox{height:250px;margin-top:11px;position:relative;background:#05080c;border-radius:14px;overflow:hidden;border:1px solid #1d3147}.cameraBox img{width:100%;height:100%;object-fit:cover;display:block}.liveBadge{position:absolute;top:11px;left:11px;padding:6px 11px;border-radius:9px;background:#10592f;color:white;font-size:13px;font-weight:900}.scanButtons{margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:18px}.scanBtn{height:42px;border:0;border-radius:12px;color:white;font-size:16px;font-weight:900}.startBtn{background:linear-gradient(180deg,#2dcc7d,#18ae63)}.stopBtn{background:linear-gradient(180deg,#ef3d48,#d92535)}.infoCard{padding:0 13px 13px}.infoBox{height:78px;margin-top:11px;border:1px solid var(--line);border-radius:12px;background:linear-gradient(180deg,#fff,#f7fbff);padding:14px;display:grid;grid-template-columns:1fr auto;align-items:center}.infoLabel{color:#425d78;font-size:13px;font-weight:900;text-transform:uppercase}.infoValue{font-size:18px;font-weight:900}.infoValue.big{min-width:64px;height:46px;border-radius:10px;background:linear-gradient(180deg,#263241,#0f1720);color:#fff;display:flex;align-items:center;justify-content:center;font-size:24px}.infoValue.green{color:var(--green)}.listCard{padding:0 14px 13px}.listTop{margin-top:11px;display:grid;grid-template-columns:1fr 130px;gap:12px}.listSelectFake{height:52px;border:1px solid var(--line);border-radius:12px;background:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 14px;font-size:17px;font-weight:900}.selectListBtn{height:52px;border-radius:12px;border:1px solid #2f80ed;background:#fff;color:#1d6ee5;font-size:14px;font-weight:900}.listMeta{margin:13px 0 10px;color:#425d78;font-size:14px;font-weight:900}.numRow{display:flex;gap:10px;flex-wrap:wrap}.quickTitle{margin-top:16px;padding-top:13px;border-top:1px solid var(--line);color:#425d78;font-size:13px;font-weight:900;text-transform:uppercase}.viewAllBtn{margin-top:12px;height:36px;width:100%;border-radius:10px;border:1px solid #2f80ed;background:#fff;color:#1d6ee5;font-weight:900}.numChip{min-width:52px;height:36px;padding:0 10px;border-radius:9px;color:white;font-size:17px;font-weight:900;display:flex;align-items:center;justify-content:center;border:2px solid rgba(255,255,255,.65);box-shadow:0 4px 10px rgba(18,36,58,.14),inset 0 1px 0 rgba(255,255,255,.22)}.numSmall{min-width:34px;height:24px;font-size:12px;border-radius:7px}.numRed{background:linear-gradient(180deg,#ef3d48,#d72f3b)}.numBlack{background:linear-gradient(180deg,#263241,#0e1720)}.numGreen{background:linear-gradient(180deg,#2dcc7d,#18ae63)}.historyCard{height:276px;padding:0 14px 12px}.historyHead{height:44px;display:flex;align-items:center;justify-content:space-between}.historyTitle{color:#425d78;font-size:13px;font-weight:900;text-transform:uppercase}.historyHint{color:#61788f;font-size:13px;font-weight:800}.historyGrid{height:220px;overflow-x:auto;overflow-y:hidden;display:grid;grid-auto-flow:column;grid-template-rows:repeat(5,1fr);grid-auto-columns:74px;gap:7px 18px;padding:2px 8px 4px;scrollbar-width:none;touch-action:pan-x}.historyGrid::-webkit-scrollbar{display:none}.historyItem{width:74px;height:38px;text-align:center}.historyNum{width:74px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;color:white;font-size:15px;font-weight:900;box-shadow:0 4px 9px rgba(18,36,58,.14);position:relative}.historyNum.match{outline:3px solid var(--gold);box-shadow:0 0 0 4px rgba(255,204,51,.22)}.historyNum.match:after{content:"★";position:absolute;top:-6px;right:4px;color:var(--gold);font-size:12px}.historyTime{font-size:10px;margin-top:2px;color:#5b7187}.modal{position:fixed;inset:0;background:rgba(30,50,70,.22);display:none;align-items:center;justify-content:center;z-index:50;backdrop-filter:blur(3px)}.modal.show{display:flex}.modalBox{background:white;border-radius:22px;box-shadow:0 20px 70px rgba(30,50,80,.25);border:1px solid var(--line);overflow:hidden}.modalHead{height:54px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 16px}.modalHead h2{margin:0;font-size:20px}.modalBody{padding:15px}.listModal{width:530px}.listChoice{height:75px;width:100%;border:1px solid var(--line);background:#fff;border-radius:14px;margin-bottom:10px;padding:10px 14px;display:grid;grid-template-columns:1fr auto;align-items:center;text-align:left}.listChoice.active{border:2px solid #2f80ed;box-shadow:0 0 0 4px #eaf2ff}.listChoiceName{font-size:18px;font-weight:900}.listChoiceMeta{margin-top:4px;color:#6f8298;font-size:13px;font-weight:800}.popupMatch{width:390px;text-align:center;padding:24px}.popupTitle{color:var(--green);font-size:22px;font-weight:900}.popupNumber{width:120px;height:90px;margin:16px auto;border-radius:18px;color:white;font-size:52px;font-weight:900;display:flex;align-items:center;justify-content:center}.closeBtn{height:38px;min-width:90px;border-radius:10px;border:1px solid var(--line2);background:white;font-weight:900}
</style>
</head>
<body>
<div class="app">
<header class="header"><div class="brand"><img class="logo" src="/assets/logo.png" onerror="this.style.display='none';document.getElementById('logoFallback').style.display='flex'"><div id="logoFallback" class="logoFallback">logo<br>path</div><div>Roulette Vision</div></div><div class="headerActions"><button class="topBtn" onclick="toggleSound()"><svg class="icon" viewBox="0 0 24 24"><path d="M4 10v4h4l5 5V5l-5 5H4z"/><path d="M16 8.5a5 5 0 0 1 0 7"/><path d="M19 6a9 9 0 0 1 0 12"/></svg>Sound</button><button class="topBtn" onclick="toggleArm()"><svg class="icon" viewBox="0 0 24 24"><path d="M7 20h10"/><path d="M9 20v-4l4-4"/><path d="M13 12l3 3"/><path d="M15 4l5 5-4 4-5-5z"/><path d="M5 8l4 4"/></svg>Arm</button><div class="statusPill" id="scanPill"><span class="statusDot"></span><span>Scanning</span></div><button class="topBtn settingsBtn" onclick="openSettings()"><svg class="icon" viewBox="0 0 24 24"><path d="M12 15.5A3.5 3.5 0 1 0 12 8a3.5 3.5 0 0 0 0 7.5z"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 0 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 0 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.6-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1A2 2 0 0 1 7.1 4l.1.1a1.7 1.7 0 0 0 1.9.3h.1A1.7 1.7 0 0 0 10 2.9V3a2 2 0 0 1 4 0v-.1a1.7 1.7 0 0 0 1 1.6h.1a1.7 1.7 0 0 0 1.9-.3l.1-.1A2 2 0 0 1 20 6.9l-.1.1a1.7 1.7 0 0 0-.3 1.9v.1a1.7 1.7 0 0 0 1.6 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>Settings</button></div></header>
<section class="topGrid"><div class="card controls"><div class="cardTitle">Controls</div><div class="controlRow"><div class="controlLeft"><svg class="controlIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h4l2-3h4l2 3h4v13H4z"/><circle cx="12" cy="13" r="4"/></svg>Camera</div><div class="switch on" id="cameraSwitch" onclick="toggleCamera()"></div></div><div class="controlRow"><div class="controlLeft"><svg class="controlIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 10v4h4l5 5V5l-5 5H4z"/><path d="M16 8.5a5 5 0 0 1 0 7"/></svg>Sound</div><div class="switch" id="soundSwitch" onclick="toggleSound()"></div></div><div class="controlRow"><div class="controlLeft"><svg class="controlIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 20h10"/><path d="M9 20v-4l4-4"/><path d="M13 12l3 3"/><path d="M15 4l5 5-4 4-5-5z"/></svg>Arm</div><div class="switch" id="armSwitch" onclick="toggleArm()"></div></div><div class="statusBox"><div class="statusLabel"><span class="statusDot"></span>Status</div><div class="statusText" id="statusText">SCANNING</div><div class="statusSub" id="statusSub">System is active</div></div></div><div class="card cameraCard"><div class="cardTitle">Live Camera</div><div class="cameraBox" id="cameraBox"><img src="/stream.mjpg"><div class="liveBadge">LIVE</div></div><div class="scanButtons"><button class="scanBtn startBtn" onclick="startScan()">▷ START</button><button class="scanBtn stopBtn" onclick="stopScan()">■ STOP</button></div></div><div class="card infoCard"><div class="cardTitle">Current Info</div><div class="infoBox"><div class="infoLabel">Scan</div><div class="infoValue green" id="scanState">SCANNING</div></div><div class="infoBox"><div class="infoLabel">Last Number</div><div class="infoValue big" id="lastNumber">--</div></div><div class="infoBox"><div class="infoLabel">Last Match</div><div class="infoValue" id="lastMatch">--</div></div></div><div class="card listCard"><div class="cardTitle">Active List</div><div class="listTop"><div class="listSelectFake"><span id="listName">List 1</span><span>⌄</span></div><button class="selectListBtn" onclick="openListModal()">Select List</button></div><div class="listMeta" id="listMeta">0 numbers</div><div class="numRow" id="listNumbers"></div><div class="quickTitle">Quick Preview</div><div class="numRow" id="quickPreview"></div><button class="viewAllBtn" onclick="openSettings()">View All Numbers</button></div></section><section class="card historyCard"><div class="historyHead"><div class="historyTitle">History</div><div class="historyHint">Swipe left/right to scroll</div></div><div class="historyGrid" id="historyGrid"></div></section></div>
<div class="modal" id="listModal"><div class="modalBox listModal"><div class="modalHead"><h2>Select Active List</h2><button class="closeBtn" onclick="closeListModal()">Close</button></div><div class="modalBody" id="listChoices"></div></div></div><div class="modal" id="matchModal"><div class="modalBox popupMatch"><div class="popupTitle">TARGET FOUND</div><div class="popupNumber" id="popupNumber">--</div><div id="popupText">Number found in selected list</div><br><button class="closeBtn" onclick="closeMatchModal()">OK</button></div></div>
<script>
let appSettings=null;let lastMatchId=null;let cameraOn=true;const RED=new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);function colorClass(n){n=Number(n);if(n===0)return"numGreen";return RED.has(n)?"numRed":"numBlack"}function chip(n,cls=""){return`<div class="numChip ${colorClass(n)} ${cls}">${n}</div>`}function hitem(x){const t=(x.timestamp||"").split("T").pop().slice(0,8);return`<div class="historyItem"><div class="historyNum ${colorClass(x.number)} ${x.isMatch?'match':''}">${x.number}</div><div class="historyTime">${t}</div></div>`}function currentList(){if(!appSettings||!appSettings.lists||!appSettings.lists.length)return null;return appSettings.lists.find(x=>x.id===appSettings.selectedListId)||appSettings.lists[0]}async function getJSON(url){return await(await fetch(url)).json()}async function postJSON(url,data={}){return await(await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)})).json()}async function loadSettings(){appSettings=await getJSON("/api/settings");renderSettings()}function renderSettings(){const hw=appSettings.hardware||{};setSwitch("soundSwitch",!!hw.buzzerEnabled);setSwitch("armSwitch",!!hw.actuatorEnabled);renderList();renderListModal()}function setSwitch(id,on){const el=document.getElementById(id);if(el)el.classList.toggle("on",!!on)}function renderList(){const l=currentList();if(!l)return;document.getElementById("listName").textContent=l.name;document.getElementById("listMeta").textContent=`${l.numbers.length} numbers`;document.getElementById("listNumbers").innerHTML=l.numbers.map(n=>chip(n)).join("");const quick=[0,1,2,3,4,5,6,7,8,9,10,11];document.getElementById("quickPreview").innerHTML=quick.map(n=>chip(n)).join("")}function renderListModal(){if(!appSettings)return;document.getElementById("listChoices").innerHTML=appSettings.lists.map(l=>`<button class="listChoice ${l.id===appSettings.selectedListId?'active':''}" onclick="selectList('${l.id}')"><div><div class="listChoiceName">${l.name}</div><div class="listChoiceMeta">${l.numbers.length} numbers</div></div><div class="numRow">${l.numbers.slice(0,8).map(n=>chip(n,'numSmall')).join("")}</div></button>`).join("")}async function selectList(id){appSettings.selectedListId=id;appSettings=await postJSON("/api/lists",appSettings);renderSettings();closeListModal()}function openListModal(){renderListModal();document.getElementById("listModal").classList.add("show")}function closeListModal(){document.getElementById("listModal").classList.remove("show")}async function toggleSound(){appSettings.hardware.buzzerEnabled=!appSettings.hardware.buzzerEnabled;appSettings.hardware=await postJSON("/api/hardware",appSettings.hardware);renderSettings()}async function toggleArm(){appSettings.hardware.actuatorEnabled=!appSettings.hardware.actuatorEnabled;appSettings.hardware=await postJSON("/api/hardware",appSettings.hardware);renderSettings()}function toggleCamera(){cameraOn=!cameraOn;setSwitch("cameraSwitch",cameraOn);document.getElementById("cameraBox").style.opacity=cameraOn?"1":".25"}async function startScan(){await postJSON("/api/start",{selectedListId:appSettings.selectedListId})}async function stopScan(){await postJSON("/api/stop")}function openSettings(){alert("Settings page will be implemented next.")}async function poll(){try{const st=await getJSON("/api/status");const scanning=!!st.monitoring;document.getElementById("scanPill").innerHTML=scanning?`<span class="statusDot"></span><span>Scanning</span>`:`<span class="statusDot" style="background:#9aa9b8"></span><span>Stopped</span>`;document.getElementById("statusText").textContent=scanning?"SCANNING":"STOPPED";document.getElementById("statusText").style.color=scanning?"#1fb86f":"#dd2f3e";document.getElementById("statusSub").textContent=scanning?"System is active":"System is stopped";document.getElementById("scanState").textContent=scanning?"SCANNING":"STOP";document.getElementById("scanState").style.color=scanning?"#1fb86f":"#dd2f3e";document.getElementById("lastNumber").textContent=st.lastOutputNumber??"--";document.getElementById("lastMatch").textContent=st.latestMatch?st.latestMatch.number:"--";document.getElementById("historyGrid").innerHTML=(st.history||[]).slice(0,90).map(hitem).join("");if(st.selectedList&&appSettings&&st.selectedList.id!==appSettings.selectedListId){appSettings.selectedListId=st.selectedList.id;renderList()}if(st.latestMatch&&st.latestMatch.id!==lastMatchId){lastMatchId=st.latestMatch.id;showMatch(st.latestMatch.number)}}catch(e){}}function showMatch(n){const box=document.getElementById("popupNumber");box.textContent=n;box.className=`popupNumber ${colorClass(n)}`;document.getElementById("popupText").textContent=`Number ${n} is in selected list`;document.getElementById("matchModal").classList.add("show");setTimeout(closeMatchModal,2500)}function closeMatchModal(){document.getElementById("matchModal").classList.remove("show")}function enableDragScroll(el){let down=false,startX=0,scrollLeft=0;el.addEventListener("mousedown",e=>{down=true;startX=e.pageX-el.offsetLeft;scrollLeft=el.scrollLeft});el.addEventListener("mouseup",()=>down=false);el.addEventListener("mouseleave",()=>down=false);el.addEventListener("mousemove",e=>{if(!down)return;e.preventDefault();const x=e.pageX-el.offsetLeft;el.scrollLeft=scrollLeft-(x-startX)*1.2});let touchX=0,touchScroll=0;el.addEventListener("touchstart",e=>{touchX=e.touches[0].clientX;touchScroll=el.scrollLeft},{passive:true});el.addEventListener("touchmove",e=>{el.scrollLeft=touchScroll-(e.touches[0].clientX-touchX)*1.2},{passive:true})}async function boot(){await loadSettings();await poll();enableDragScroll(document.getElementById("historyGrid"));setInterval(poll,350)}boot();
</script>
</body>
</html>'''


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
