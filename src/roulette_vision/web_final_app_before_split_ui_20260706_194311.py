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
  <link rel="stylesheet" href="/ui/styles.css">
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/ui/app.js"></script>
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


    def serve_static_file(self, file_path):
        import mimetypes
        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        data = file_path.read_bytes()
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

        if file_path.suffix == ".js":
            mime = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif file_path.suffix == ".html":
            mime = "text/html; charset=utf-8"
        elif file_path.suffix == ".png":
            mime = "image/png"

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/ui/"):
            rel = path[len("/ui/"):].lstrip("/")
            if ".." in rel:
                self.send_error(404)
                return
            self.serve_static_file(Path("data/ui") / rel)
            return

        if path == "/assets/logo.png":
            self.serve_static_file(Path("data/assets/logo.png"))
            return

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
