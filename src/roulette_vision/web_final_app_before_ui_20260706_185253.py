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
:root{--bg:#0e1822;--panel:#162331;--panel2:#1c2b3b;--border:#35506b;--text:#f3f7fb;--muted:#9cb0c3;--match:#f7d154}
*{box-sizing:border-box}
html,body{margin:0;width:800px;height:480px;overflow:hidden;background:var(--bg);color:var(--text);font-family:Arial,Helvetica,sans-serif}
body{display:flex;flex-direction:column}
header{height:42px;flex:0 0 42px;display:flex;align-items:center;justify-content:space-between;padding:0 10px;background:#13202c;border-bottom:1px solid #263b4f}
h1{margin:0;font-size:21px;font-weight:900}
button{background:#23374a;color:#fff;border:1px solid #47647f;border-radius:10px;height:30px;padding:0 10px;font-weight:800;cursor:pointer}
button.green{background:#155f3d}button.red{background:#7a2830}
.main{height:438px;display:flex;flex-direction:column;padding:7px;gap:7px}
.top{height:246px;display:grid;grid-template-columns:168px 1fr 210px;gap:7px}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;box-shadow:0 2px 8px rgba(0,0,0,.25);overflow:hidden}
.left{padding:10px;display:flex;flex-direction:column;gap:9px}
.title{font-size:12px;color:var(--muted);font-weight:900;text-transform:uppercase;letter-spacing:.4px}
.toggleRow{display:flex;align-items:center;justify-content:space-between;background:var(--panel2);border:1px solid #30485d;border-radius:12px;padding:7px 9px}
.toggleRow label{font-size:14px;font-weight:800}
.switch{appearance:none;width:46px;height:25px;border-radius:18px;background:#42596f;position:relative;outline:none;cursor:pointer}
.switch:checked{background:#27b36a}
.switch:after{content:"";position:absolute;top:3px;left:3px;width:19px;height:19px;border-radius:50%;background:#fff;transition:.18s}
.switch:checked:after{left:24px}
.statusMini{background:#0f1821;border:1px solid #2d4154;border-radius:12px;padding:7px 9px}
.statusMini .k{font-size:11px;color:var(--muted)}.statusMini .v{font-size:19px;font-weight:900;margin-top:2px}
.center{padding:7px;display:grid;grid-template-columns:1fr 176px;gap:7px}
.cameraBox{position:relative;background:#000;border:1px solid #2f455a;border-radius:12px;overflow:hidden}
.cameraBox img{width:100%;height:100%;object-fit:cover;display:block}
.badge{position:absolute;left:7px;top:7px;background:rgba(0,0,0,.65);border:1px solid rgba(255,255,255,.18);border-radius:9px;padding:4px 7px;font-size:12px;font-weight:900}
.liveSide{display:flex;flex-direction:column;gap:7px}
.liveCard{flex:1;background:#0f1821;border:1px solid #2d4154;border-radius:12px;padding:8px;display:flex;flex-direction:column;justify-content:center}
.liveCard .k{font-size:11px;color:var(--muted);font-weight:800}.liveCard .big{font-size:35px;font-weight:900;line-height:1}.liveCard .mid{font-size:18px;font-weight:900}
.right{padding:9px;display:flex;flex-direction:column;gap:7px}
select{width:100%;height:33px;border-radius:10px;border:1px solid #45617c;background:#0f1821;color:#fff;font-size:13px;padding:0 7px}
.listInfo{font-size:12px;color:var(--muted)}
.numberGrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;overflow:auto;padding-right:1px}
.numTile{height:38px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:900;border:2px solid rgba(255,255,255,.14);box-shadow:0 2px 8px rgba(0,0,0,.25)}
.num-red{background:#b8303d;color:#fff}.num-black{background:#252d35;color:#fff}.num-green{background:#18a85c;color:#fff}
.targetActive{border:2px solid #fff;box-shadow:0 0 11px rgba(255,255,255,.28)}
.historyPanel{flex:1;display:flex;flex-direction:column;padding:9px;min-height:0}
.histHead{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.histHead h3{margin:0;font-size:18px}.hint{font-size:12px;color:var(--muted)}
.track{flex:1;min-height:0;overflow-x:auto;overflow-y:hidden;display:flex;flex-direction:row-reverse;gap:7px;padding-bottom:3px}
.histItem{width:62px;min-width:62px;border-radius:14px;border:2px solid rgba(255,255,255,.14);padding:5px 4px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.histItem .n{font-size:27px;font-weight:900;line-height:1.05}.histItem .t{font-size:10px;margin-top:4px;opacity:.9}.histItem .tag{font-size:10px;font-weight:900;margin-top:3px}
.histMatch{border:3px solid var(--match)!important;box-shadow:0 0 13px rgba(247,209,84,.62)}
.histMatch .tag{color:var(--match)}
.popup{position:fixed;inset:0;background:rgba(0,0,0,.72);display:none;align-items:center;justify-content:center;z-index:99}
.popup.show{display:flex}.popupBox{width:410px;border-radius:26px;background:#13251a;border:3px solid #5fff9b;box-shadow:0 0 28px rgba(95,255,155,.45);text-align:center;padding:23px}.popupBox h2{margin:0;font-size:26px;color:#5fff9b}.popupNumber{font-size:105px;font-weight:900;line-height:1}.popupText{font-size:20px;font-weight:900}
.settings{position:fixed;inset:0;background:#0e1822;z-index:50;display:none;padding:10px;overflow:auto}
.settings.show{display:block}.settingsGrid{display:grid;grid-template-columns:1fr 260px;gap:10px}.listCard{background:#162331;border:1px solid #35506b;border-radius:12px;padding:9px;margin-bottom:8px}
textarea,input{background:#0f1821;color:#fff;border:1px solid #45617c;border-radius:8px;padding:7px}textarea{width:100%;height:68px}.hidden{display:none!important}
</style>
</head>
<body>
<header><h1>Roulette Vision</h1><div><button onclick="clearHistory()">Clear</button> <button onclick="openSettings()">Settings</button></div></header>
<div class="main">
<section class="top">
<div class="panel left">
<div class="title">Controls</div>
<div class="toggleRow"><label>Camera</label><input id="cameraToggle" class="switch" type="checkbox" checked onchange="toggleCamera()"></div>
<div class="toggleRow"><label>Sound</label><input id="soundToggle" class="switch" type="checkbox" onchange="toggleSound()"></div>
<div class="toggleRow"><label>Arm</label><input id="armToggle" class="switch" type="checkbox" onchange="toggleArm()"></div>
<button class="green" onclick="startScan()">Start Scan</button>
<button class="red" onclick="stopScan()">Stop</button>
<div class="statusMini"><div class="k">Status</div><div id="scanMini" class="v">STOP</div></div>
</div>
<div class="panel center">
<div class="cameraBox"><img id="stream" src="/stream.mjpg"><div id="badge" class="badge">READY</div></div>
<div class="liveSide">
<div class="liveCard"><div class="k">Scan</div><div id="scanBig" class="big">STOP</div></div>
<div class="liveCard"><div class="k">Last Number</div><div id="lastNum" class="big">--</div></div>
<div class="liveCard"><div class="k">Last Match</div><div id="lastMatch" class="mid">--</div></div>
</div>
</div>
<div class="panel right">
<div class="title">Selected List</div>
<select id="selectedList" onchange="changeSelectedList()"></select>
<div id="listInfo" class="listInfo">-</div>
<div id="targetGrid" class="numberGrid"></div>
</div>
</section>
<section class="panel historyPanel">
<div class="histHead"><h3>History</h3><div class="hint">Newest on the right</div></div>
<div id="historyTrack" class="track"></div>
</section>
</div>
<div id="popup" class="popup"><div class="popupBox"><h2>TARGET FOUND</h2><div id="popupNumber" class="popupNumber">--</div><div class="popupText">Buzzer + Arm + Popup</div></div></div>
<section id="settings" class="settings">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px"><h2 style="margin:0">Settings</h2><button onclick="closeSettings()">Back</button></div>
<div class="settingsGrid">
<div><h3>Target Lists</h3><div id="lists"></div><button onclick="addList()">Add List</button> <button class="green" onclick="saveLists()">Save Lists</button></div>
<div><h3>Hardware</h3>
<label><input id="dryRun" type="checkbox"> Dry run</label><br><br>
<label><input id="buzzerEnabled" type="checkbox"> Buzzer enabled</label><br>
<label>Buzzer GPIO <input id="buzzerPin" type="number" style="width:80px"></label><br>
<label>Buzzer ms <input id="buzzerDurationMs" type="number" style="width:80px"></label><br><br>
<label><input id="actuatorEnabled" type="checkbox"> Arm enabled</label><br>
<label>Arm GPIO <input id="actuatorPin" type="number" style="width:80px"></label><br>
<label>Out % <input id="actuatorOutPercent" type="number" style="width:80px"></label><br>
<label>Home % <input id="actuatorHomePercent" type="number" style="width:80px"></label><br>
<label>Hold ms <input id="actuatorHoldMs" type="number" style="width:80px"></label><br><br>
<button class="green" onclick="saveHardware()">Save Hardware</button>
<button onclick="testAction()">Test Action</button>
<p id="msg" class="hint"></p>
</div>
</div>
</section>
<script>
let appSettings=null,lastMatchId=null;
const RED=new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
function cls(n){n=Number(n);return n===0?'num-green':(RED.has(n)?'num-red':'num-black')}
function numTile(n,a=false){return `<div class="numTile ${cls(n)} ${a?'targetActive':''}">${n}</div>`}
function histTile(r){let t=(r.timestamp||'').split('T').pop().slice(0,8);return `<div class="histItem ${cls(r.number)} ${r.isMatch?'histMatch':''}"><div class="n">${r.number}</div><div class="t">${t}</div><div class="tag">${r.isMatch?'MATCH':''}</div></div>`}
function numsFromText(t){return [...new Set((t.match(/\d+/g)||[]).map(Number).filter(n=>n>=0&&n<=36))].sort((a,b)=>a-b)}
async function loadSettings(){appSettings=await (await fetch('/api/settings')).json();renderSelect();renderLists();renderHardware()}
function selectedObj(){return (appSettings.lists||[]).find(x=>x.id===appSettings.selectedListId)||appSettings.lists[0]}
function renderSelect(){let s=document.getElementById('selectedList');s.innerHTML='';(appSettings.lists||[]).forEach(l=>{let o=document.createElement('option');o.value=l.id;o.textContent=l.name;if(l.id===appSettings.selectedListId)o.selected=true;s.appendChild(o)});renderTargets()}
function renderTargets(){let l=selectedObj();if(!l)return;document.getElementById('listInfo').textContent=`${l.name} | ${l.numbers.length} numbers`;document.getElementById('targetGrid').innerHTML=l.numbers.map(n=>numTile(n,true)).join('')}
function renderLists(){let c=document.getElementById('lists');c.innerHTML='';(appSettings.lists||[]).forEach((l,i)=>{let d=document.createElement('div');d.className='listCard';d.innerHTML=`<input data-name="${i}" value="${l.name}" style="width:100%;margin-bottom:6px"><textarea data-nums="${i}">${l.numbers.join(' ')}</textarea><br><button class="red" onclick="removeList(${i})">Remove</button>`;c.appendChild(d)})}
function renderHardware(){let h=appSettings.hardware||{};['dryRun','buzzerEnabled','actuatorEnabled'].forEach(k=>document.getElementById(k).checked=!!h[k]);['buzzerPin','buzzerDurationMs','actuatorPin','actuatorOutPercent','actuatorHomePercent','actuatorHoldMs'].forEach(k=>document.getElementById(k).value=h[k]??'');document.getElementById('soundToggle').checked=!!h.buzzerEnabled;document.getElementById('armToggle').checked=!!h.actuatorEnabled}
function addList(){appSettings.lists.push({id:'list-'+Math.random().toString(16).slice(2,10),name:'List '+(appSettings.lists.length+1),numbers:[]});renderLists();renderSelect()}
function removeList(i){appSettings.lists.splice(i,1);if(!appSettings.lists.length)addList();renderLists();renderSelect()}
async function saveLists(){appSettings.lists=appSettings.lists.map((l,i)=>({...l,name:document.querySelector(`[data-name="${i}"]`).value,numbers:numsFromText(document.querySelector(`[data-nums="${i}"]`).value)}));appSettings.selectedListId=document.getElementById('selectedList').value;appSettings=await (await fetch('/api/lists',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(appSettings)})).json();renderSelect();renderLists();document.getElementById('msg').textContent='Lists saved'}
async function changeSelectedList(){appSettings.selectedListId=document.getElementById('selectedList').value;appSettings=await (await fetch('/api/lists',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(appSettings)})).json();renderSelect()}
async function saveHardware(){let h={};['dryRun','buzzerEnabled','actuatorEnabled'].forEach(k=>h[k]=document.getElementById(k).checked);['buzzerPin','buzzerDurationMs','actuatorPin','actuatorOutPercent','actuatorHomePercent','actuatorHoldMs'].forEach(k=>h[k]=Number(document.getElementById(k).value));appSettings.hardware=await (await fetch('/api/hardware',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(h)})).json();renderHardware();document.getElementById('msg').textContent='Hardware saved'}
async function saveHardwarePartial(p){appSettings.hardware=await (await fetch('/api/hardware',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...appSettings.hardware,...p})})).json();renderHardware()}
async function toggleSound(){await saveHardwarePartial({buzzerEnabled:document.getElementById('soundToggle').checked})}
async function toggleArm(){await saveHardwarePartial({actuatorEnabled:document.getElementById('armToggle').checked})}
async function toggleCamera(){await fetch('/api/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:document.getElementById('cameraToggle').checked})})}
async function testAction(){await fetch('/api/test-action',{method:'POST'});document.getElementById('msg').textContent='Test action sent'}
async function startScan(){await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({selectedListId:document.getElementById('selectedList').value})})}
async function stopScan(){await fetch('/api/stop',{method:'POST'})}
async function clearHistory(){await fetch('/api/history/clear',{method:'POST'})}
function openSettings(){document.getElementById('settings').classList.add('show')}
function closeSettings(){document.getElementById('settings').classList.remove('show');loadSettings()}
function popup(n,id){if(id===lastMatchId)return;lastMatchId=id;document.getElementById('popupNumber').textContent=n;let p=document.getElementById('popup');p.classList.add('show');setTimeout(()=>p.classList.remove('show'),2300)}
async function poll(){try{let st=await (await fetch('/api/status')).json();document.getElementById('cameraToggle').checked=!!st.cameraEnabled;let sc=st.monitoring?'SCAN':'STOP';document.getElementById('scanMini').textContent=sc;document.getElementById('scanBig').textContent=sc;document.getElementById('badge').textContent=st.state||'READY';document.getElementById('lastNum').textContent=(st.history&&st.history.length)?st.history[0].number:'--';document.getElementById('lastMatch').textContent=st.latestMatch?st.latestMatch.number:'--';document.getElementById('historyTrack').innerHTML=(st.history||[]).slice(0,60).map(histTile).join('');if(st.latestMatch)popup(st.latestMatch.number,st.latestMatch.id)}catch(e){}}
loadSettings();setInterval(poll,300);poll();
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
