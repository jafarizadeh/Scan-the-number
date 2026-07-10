import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .engine import RouletteDetectionEngine
from .vision import crop_relative_roi, roi_metrics, roi_rect_from_shape


class DetectionHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.source = source

        self.camera = FrameSource(cfg["camera"], source=source)
        self.engine = RouletteDetectionEngine(cfg)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.latest_status = {
            "ready": False,
            "state": "STARTING",
            "armed": False,
            "metrics": {},
            "prediction": None,
            "preview_prediction": None,
            "event": None
        }

        self.events = deque(maxlen=50)
        self.running = False
        self.thread = None

    def start(self):
        if not self.engine.matcher.is_ready():
            labels = []
        else:
            try:
                labels = self.engine.matcher.labels()
            except Exception:
                labels = []

        print("Loaded template labels:", labels)

        self.camera.start()
        self.running = True

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.03)
                continue

            roi = crop_relative_roi(frame, self.cfg["roi"])

            try:
                result = self.engine.process(frame, roi)
            except Exception as e:
                result = {
                    "state": "ERROR",
                    "armed": False,
                    "metrics": roi_metrics(roi),
                    "empty_score": None,
                    "prediction": None,
                    "event": None,
                    "error": str(e)
                }

            preview = None
            try:
                if result.get("state") == "BRIGHT" and self.engine.matcher.is_ready():
                    preview = self.engine.matcher.predict(roi)
            except Exception:
                preview = None

            event = result.get("event")
            if event:
                self.events.appendleft(event)

            status = {
                "ready": True,
                "state": result.get("state"),
                "armed": result.get("armed"),
                "metrics": result.get("metrics"),
                "empty_score": result.get("empty_score"),
                "prediction": result.get("prediction"),
                "preview_prediction": preview,
                "event": event,
                "events": list(self.events),
                "templates_ready": self.engine.matcher.is_ready()
            }

            if "error" in result:
                status["error"] = result["error"]

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_roi = roi.copy()
                self.latest_status = status

            time.sleep(delay)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_roi(self):
        with self.lock:
            if self.latest_roi is None:
                return None
            return self.latest_roi.copy()

    def get_status(self):
        with self.lock:
            return json.loads(json.dumps(self.latest_status, default=str))

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.camera.close()


def encode_jpeg(frame, quality=85):
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        return None
    return buffer.tobytes()


def draw_overlay(frame, cfg, status):
    x, y, rw, rh = roi_rect_from_shape(frame.shape, cfg["roi"])

    cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 255), 2)
    cv2.putText(
        frame,
        "RESULT ROI",
        (x, max(20, y + rh + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    if "focus_roi" in cfg:
        fx, fy, fw, fh = roi_rect_from_shape(frame.shape, cfg["focus_roi"])
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "FOCUS ROI",
            (fx, max(20, fy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    state = status.get("state", "UNKNOWN")
    armed = status.get("armed", False)
    preview = status.get("preview_prediction")

    text = f"STATE: {state} | ARMED: {armed}"

    if preview:
        text += (
            f" | PREVIEW: {preview.get('label')} "
            f"s={preview.get('score', 0):.3f} "
            f"m={preview.get('margin', 0):.3f}"
        )

    cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 900), 55), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    event = status.get("event")
    if event:
        msg = f"CONFIRMED: {event.get('number')} | round {event.get('round_index')}"
        cv2.rectangle(frame, (10, 65), (520, 110), (0, 90, 0), -1)
        cv2.putText(
            frame,
            msg,
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame


def make_handler(hub, cfg):
    class LiveDetectionHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path

            if path == "/":
                self.send_index()
            elif path == "/stream.mjpg":
                self.send_stream("full")
            elif path == "/roi.mjpg":
                self.send_stream("roi")
            elif path == "/api/status":
                self.send_json(hub.get_status())
            else:
                self.send_error(404, "Not found")

        def send_json(self, data, status=200):
            body = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_index(self):
            html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Roulette Vision - Live Detection</title>
  <style>
    body {
      margin: 0;
      background: #101418;
      color: #f2f2f2;
      font-family: Arial, sans-serif;
    }
    header {
      padding: 14px 20px;
      background: #1b232b;
      border-bottom: 1px solid #34404a;
    }
    h1 {
      margin: 0;
      font-size: 22px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(680px, 1fr) 460px;
      gap: 16px;
      padding: 16px;
    }
    .card {
      background: #1b232b;
      border: 1px solid #34404a;
      border-radius: 10px;
      padding: 12px;
    }
    .card h2 {
      margin: 0 0 10px 0;
      font-size: 18px;
    }
    img {
      max-width: 100%;
      border-radius: 8px;
      border: 1px solid #34404a;
      background: #000;
    }
    .roi {
      width: 100%;
      image-rendering: auto;
    }
    .statusBox {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }
    .pill {
      background: #0b0e11;
      border: 1px solid #34404a;
      border-radius: 8px;
      padding: 10px;
      font-size: 15px;
    }
    .big {
      font-size: 28px;
      font-weight: bold;
    }
    .ok {
      color: #9ef09e;
    }
    .warn {
      color: #ffd27a;
    }
    .bad {
      color: #ff9999;
    }
    pre {
      background: #0b0e11;
      padding: 10px;
      border-radius: 8px;
      overflow: auto;
      font-size: 12px;
      max-height: 360px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    td, th {
      border-bottom: 1px solid #34404a;
      padding: 6px;
      text-align: left;
    }
    .hint {
      color: #aeb7c0;
      font-size: 14px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <header>
    <h1>Roulette Vision - Live Detection</h1>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Full Camera Frame</h2>
      <img src="/stream.mjpg">
      <p class="hint">
        کادر زرد Result ROI است. سیستم فقط بعد از دیدن EMPTY و سپس عدد روشن، نتیجه را ثبت می‌کند.
      </p>
    </div>

    <div class="card">
      <h2>Result ROI</h2>
      <img class="roi" src="/roi.mjpg">

      <h2>Live Status</h2>
      <div class="statusBox">
        <div class="pill">
          State<br>
          <span id="state" class="big warn">---</span>
        </div>
        <div class="pill">
          Armed<br>
          <span id="armed" class="big">---</span>
        </div>
        <div class="pill">
          Preview<br>
          <span id="preview" class="big">---</span>
        </div>
        <div class="pill">
          Last Confirmed<br>
          <span id="confirmed" class="big ok">---</span>
        </div>
      </div>

      <h2>Recent Confirmed Results</h2>
      <table>
        <thead>
          <tr>
            <th>Round</th>
            <th>Number</th>
            <th>Score</th>
            <th>Margin</th>
          </tr>
        </thead>
        <tbody id="events"></tbody>
      </table>

      <h2>Debug</h2>
      <pre id="debug">loading...</pre>
    </div>
  </div>

<script>
function fmt(x, digits=3) {
  if (x === null || x === undefined) return "---";
  if (typeof x === "number") return x.toFixed(digits);
  return x;
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    document.getElementById('state').textContent = data.state || '---';
    document.getElementById('armed').textContent = data.armed ? 'YES' : 'NO';

    if (data.preview_prediction) {
      document.getElementById('preview').textContent =
        data.preview_prediction.label +
        ' / ' +
        fmt(data.preview_prediction.score, 3) +
        ' / m=' +
        fmt(data.preview_prediction.margin, 3);
    } else {
      document.getElementById('preview').textContent = '---';
    }

    if (data.events && data.events.length > 0) {
      const e = data.events[0];
      document.getElementById('confirmed').textContent = e.number;
    } else {
      document.getElementById('confirmed').textContent = '---';
    }

    const tbody = document.getElementById('events');
    tbody.innerHTML = '';

    if (data.events) {
      for (const e of data.events.slice(0, 12)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${e.round_index}</td>
          <td><b>${e.number}</b></td>
          <td>${fmt(e.score, 3)}</td>
          <td>${fmt(e.margin, 3)}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    const debug = {
      state: data.state,
      armed: data.armed,
      empty_score: data.empty_score,
      metrics: data.metrics,
      prediction: data.prediction,
      preview_prediction: data.preview_prediction,
      event: data.event
    };

    document.getElementById('debug').textContent = JSON.stringify(debug, null, 2);
  } catch (e) {
    document.getElementById('debug').textContent = 'status unavailable';
  }
}

setInterval(refreshStatus, 500);
refreshStatus();
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

        def send_stream(self, mode):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                if mode == "roi":
                    frame = hub.get_roi()
                    if frame is not None:
                        frame = cv2.resize(
                            frame,
                            None,
                            fx=3,
                            fy=3,
                            interpolation=cv2.INTER_NEAREST
                        )
                else:
                    frame = hub.get_frame()
                    if frame is not None:
                        frame = draw_overlay(frame, cfg, hub.get_status())

                if frame is None:
                    time.sleep(0.05)
                    continue

                jpg = encode_jpeg(frame)
                if jpg is None:
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.05)
                except BrokenPipeError:
                    break
                except ConnectionResetError:
                    break

    return LiveDetectionHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    hub = DetectionHub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("Roulette Vision LIVE DETECTION web server is running:")
    print(f"  Local:   http://127.0.0.1:{args.port}")
    print(f"  Network: http://192.168.1.251:{args.port}")
    print("")
    print("Detection is running inside the web server.")
    print("Open the Network URL from Windows.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping live detection web server...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
