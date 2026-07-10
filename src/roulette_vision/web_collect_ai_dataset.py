import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_rect_from_shape, roi_metrics


class DatasetHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.status = {
            "ready": False,
            "capture_active": False,
            "label": None,
            "remaining": 0,
            "saved": 0,
        }

        self.capture_active = False
        self.capture_label = None
        self.capture_remaining = 0
        self.capture_total = 0
        self.capture_interval = 0.08
        self.last_save_time = 0.0

        self.running = False
        self.thread = None

        Path("data/ai_dataset").mkdir(parents=True, exist_ok=True)

    def start(self):
        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def counts(self):
        result = {}
        root = Path("data/ai_dataset")

        for i in range(37):
            folder = root / str(i)
            result[str(i)] = len(list(folder.glob("*.png"))) if folder.exists() else 0

        return result

    def start_capture(self, label, count, interval_ms):
        label = int(label)
        count = int(count)

        if label < 0 or label > 36:
            raise ValueError("label must be between 0 and 36")

        with self.lock:
            self.capture_label = label
            self.capture_remaining = count
            self.capture_total = count
            self.capture_interval = max(0.02, float(interval_ms) / 1000.0)
            self.capture_active = True
            self.last_save_time = 0.0

        Path("data/ai_dataset", str(label)).mkdir(parents=True, exist_ok=True)

    def _loop(self):
        while self.running:
            frame = self.camera.read()

            if frame is None:
                time.sleep(0.03)
                continue

            roi = crop_relative_roi(frame, self.cfg["roi"])
            metrics = roi_metrics(roi)

            now = time.time()
            saved_now = False

            with self.lock:
                if self.capture_active and self.capture_remaining > 0:
                    if now - self.last_save_time >= self.capture_interval:
                        label = self.capture_label
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        path = Path("data/ai_dataset") / str(label) / f"{label}_{ts}.png"
                        cv2.imwrite(str(path), roi)

                        self.capture_remaining -= 1
                        self.last_save_time = now
                        saved_now = True

                        if self.capture_remaining <= 0:
                            self.capture_active = False

                self.latest_frame = frame.copy()
                self.latest_roi = roi.copy()
                self.status = {
                    "ready": True,
                    "capture_active": self.capture_active,
                    "label": self.capture_label,
                    "remaining": self.capture_remaining,
                    "total": self.capture_total,
                    "saved": self.capture_total - self.capture_remaining,
                    "counts": self.counts(),
                    "metrics": metrics,
                    "saved_now": saved_now,
                }

            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_roi(self):
        with self.lock:
            return None if self.latest_roi is None else self.latest_roi.copy()

    def get_status(self):
        with self.lock:
            return json.loads(json.dumps(self.status, default=str))

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.camera.close()


def encode_jpeg(frame, quality=85):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


def draw_overlay(frame, cfg, status):
    x, y, w, h = roi_rect_from_shape(frame.shape, cfg["roi"])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    text = "AI DATASET COLLECTOR"
    if status.get("capture_active"):
        text += f" | label={status.get('label')} saved={status.get('saved')} remaining={status.get('remaining')}"

    cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 1050), 55), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def make_handler(hub, cfg):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path == "/":
                self.index()
            elif path == "/stream.mjpg":
                self.stream("full")
            elif path == "/roi.mjpg":
                self.stream("roi")
            elif path == "/api/status":
                self.send_json(hub.get_status())
            elif path == "/api/start":
                try:
                    label = int(qs.get("label", ["0"])[0])
                    count = int(qs.get("count", ["100"])[0])
                    interval_ms = int(qs.get("interval_ms", ["80"])[0])
                    hub.start_capture(label, count, interval_ms)
                    self.send_json({"ok": True, "label": label, "count": count})
                except Exception as e:
                    self.send_json({"ok": False, "error": str(e)}, status=400)
            else:
                self.send_error(404)

        def send_json(self, data, status=200):
            body = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def index(self):
            options = "\n".join([f'<option value="{i}">{i}</option>' for i in range(37)])
            html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Roulette AI Dataset Collector</title>
<style>
body {{ margin:0; background:#101418; color:#f2f2f2; font-family:Arial,sans-serif; }}
header {{ padding:14px 20px; background:#1b232b; border-bottom:1px solid #34404a; }}
.grid {{ display:grid; grid-template-columns:minmax(680px,1fr) 480px; gap:16px; padding:16px; }}
.card {{ background:#1b232b; border:1px solid #34404a; border-radius:10px; padding:12px; }}
img {{ max-width:100%; border-radius:8px; border:1px solid #34404a; background:#000; }}
select,input,button {{ font-size:18px; padding:8px; margin:4px; }}
button {{ cursor:pointer; }}
pre {{ background:#0b0e11; padding:10px; border-radius:8px; overflow:auto; max-height:420px; }}
</style>
</head>
<body>
<header><h1>Roulette AI Dataset Collector</h1></header>
<div class="grid">
  <div class="card">
    <h2>Full Camera</h2>
    <img src="/stream.mjpg">
  </div>
  <div class="card">
    <h2>ROI</h2>
    <img src="/roi.mjpg" style="width:100%">

    <h2>Capture</h2>
    <label>Label:</label>
    <select id="label">{options}</select><br>
    <label>Count:</label>
    <input id="count" type="number" value="100" min="1" max="1000"><br>
    <label>Interval ms:</label>
    <input id="interval" type="number" value="80" min="20" max="1000"><br>
    <button onclick="startCapture()">Collect</button>

    <h2>Status</h2>
    <pre id="status">loading...</pre>
  </div>
</div>

<script>
async function startCapture(){{
  const label = document.getElementById('label').value;
  const count = document.getElementById('count').value;
  const interval = document.getElementById('interval').value;
  await fetch(`/api/start?label=${{label}}&count=${{count}}&interval_ms=${{interval}}`);
  refresh();
}}

async function refresh(){{
  const res = await fetch('/api/status');
  const data = await res.json();
  document.getElementById('status').textContent = JSON.stringify(data,null,2);
}}

setInterval(refresh,500);
refresh();
</script>
</body>
</html>
"""
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream(self, mode):
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                if mode == "roi":
                    frame = hub.get_roi()
                    if frame is not None:
                        frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                else:
                    frame = hub.get_frame()
                    if frame is not None:
                        frame = draw_overlay(frame, cfg, hub.get_status())

                if frame is None:
                    time.sleep(0.05)
                    continue

                jpg = encode_jpeg(frame)

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.05)
                except Exception:
                    break

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    hub = DatasetHub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("AI dataset collector running:")
    print(f"  http://192.168.1.251:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
