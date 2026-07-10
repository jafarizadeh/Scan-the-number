import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_metrics, roi_rect_from_shape


class CameraHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.source = source
        self.camera = FrameSource(cfg["camera"], source=source)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.thread = None

    def start(self):
        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        while self.running:
            frame = self.camera.read()
            if frame is not None:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(delay)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

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


def draw_roi_box(frame, cfg):
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

    return frame


def count_templates():
    result = {}
    for n in range(37):
        d = Path("data/templates") / str(n)
        result[str(n)] = len(list(d.glob("*.png"))) if d.exists() else 0
    return result


def save_roi_image(frame, cfg, out_path):
    roi = crop_relative_roi(frame, cfg["roi"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), roi)
    return roi


def make_handler(hub, cfg):
    class RouletteWebHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path == "/":
                self.send_index()
            elif path == "/stream.mjpg":
                self.send_stream(mode="full")
            elif path == "/roi.mjpg":
                self.send_stream(mode="roi")
            elif path == "/snapshot.jpg":
                self.send_snapshot(mode="full")
            elif path == "/roi_snapshot.jpg":
                self.send_snapshot(mode="roi")
            elif path == "/metrics":
                self.send_metrics()
            elif path == "/api/template-counts":
                self.send_json({"ok": True, "counts": count_templates()})
            elif path == "/api/capture-template":
                self.capture_template(query)
            elif path == "/api/capture-empty":
                self.capture_empty(query)
            else:
                self.send_error(404, "Not found")

        def send_json(self, data, status=200):
            body = json.dumps(data, indent=2).encode("utf-8")
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
  <title>Roulette Vision - Live Camera</title>
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
      grid-template-columns: minmax(640px, 1fr) 430px;
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
    pre {
      background: #0b0e11;
      padding: 10px;
      border-radius: 8px;
      overflow: auto;
      font-size: 12px;
      max-height: 380px;
    }
    .hint {
      color: #aeb7c0;
      font-size: 14px;
      line-height: 1.5;
    }
    input, button {
      font-size: 15px;
      border-radius: 7px;
      border: 1px solid #485866;
      padding: 8px;
      margin: 4px 4px 4px 0;
    }
    input {
      width: 70px;
      background: #0b0e11;
      color: #fff;
    }
    button {
      background: #25313b;
      color: #fff;
      cursor: pointer;
    }
    button:hover {
      background: #314252;
    }
    .danger {
      background: #4a2730;
    }
    .ok {
      color: #9ef09e;
      font-weight: bold;
    }
    .warn {
      color: #ffd27a;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <header>
    <h1>Roulette Vision - Live Camera</h1>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Full Camera Frame</h2>
      <img src="/stream.mjpg">
      <p class="hint">
        کادر زرد برای تشخیص عدد است. کادر سبز فقط برای کنترل فوکوس است.
      </p>

      <div class="card">
        <h2>Template Capture</h2>
        <div>
          <label>Number:</label>
          <input id="label" type="number" min="0" max="36" value="12">
          <button onclick="captureTemplate(1)">Save 1 Template</button>
          <button onclick="captureTemplate(40)">Collect 40 Templates</button>
          <button class="danger" onclick="captureEmpty(30)">Collect 30 EMPTY</button>
          <button onclick="refreshCounts()">Refresh Counts</button>
        </div>
        <p class="hint">
          عددی را وارد کن که الآن در Result ROI دیده می‌شود، سپس Collect 40 Templates را بزن.
          وقتی ناحیه نتیجه خالی/سیاه شد، Collect 30 EMPTY را بزن.
        </p>
        <pre id="actionResult">ready</pre>
        <h2>Template Counts</h2>
        <pre id="counts">loading...</pre>
      </div>
    </div>

    <div class="card">
      <h2>Result ROI</h2>
      <img class="roi" src="/roi.mjpg">

      <h2>ROI Metrics</h2>
      <pre id="metrics">loading...</pre>
    </div>
  </div>

<script>
async function refreshMetrics() {
  try {
    const res = await fetch('/metrics');
    const data = await res.json();
    document.getElementById('metrics').textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    document.getElementById('metrics').textContent = 'metrics unavailable';
  }
}

async function refreshCounts() {
  try {
    const res = await fetch('/api/template-counts');
    const data = await res.json();
    document.getElementById('counts').textContent = JSON.stringify(data.counts, null, 2);
  } catch (e) {
    document.getElementById('counts').textContent = 'counts unavailable';
  }
}

async function captureTemplate(count) {
  const label = document.getElementById('label').value;
  document.getElementById('actionResult').textContent = 'capturing template...';

  try {
    const res = await fetch(`/api/capture-template?label=${label}&count=${count}&interval=0.10`);
    const data = await res.json();
    document.getElementById('actionResult').textContent = JSON.stringify(data, null, 2);
    refreshCounts();
  } catch (e) {
    document.getElementById('actionResult').textContent = 'capture failed';
  }
}

async function captureEmpty(count) {
  document.getElementById('actionResult').textContent = 'capturing empty...';

  try {
    const res = await fetch(`/api/capture-empty?count=${count}&interval=0.08`);
    const data = await res.json();
    document.getElementById('actionResult').textContent = JSON.stringify(data, null, 2);
    refreshCounts();
  } catch (e) {
    document.getElementById('actionResult').textContent = 'empty capture failed';
  }
}

setInterval(refreshMetrics, 1000);
refreshMetrics();
refreshCounts();
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

        def send_snapshot(self, mode):
            frame = hub.get_frame()
            if frame is None:
                self.send_error(503, "No camera frame yet")
                return

            if mode == "roi":
                frame = crop_relative_roi(frame, cfg["roi"])
            else:
                frame = draw_roi_box(frame, cfg)

            jpg = encode_jpeg(frame)
            if jpg is None:
                self.send_error(500, "Could not encode JPEG")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)

        def send_stream(self, mode):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                frame = hub.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                if mode == "roi":
                    frame = crop_relative_roi(frame, cfg["roi"])
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=3,
                        fy=3,
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    frame = draw_roi_box(frame, cfg)

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

        def send_metrics(self):
            frame = hub.get_frame()
            if frame is None:
                self.send_json({"ready": False})
                return

            roi = crop_relative_roi(frame, cfg["roi"])
            x, y, rw, rh = roi_rect_from_shape(frame.shape, cfg["roi"])

            focus_cfg = cfg.get("focus_roi", cfg["roi"])
            focus_roi = crop_relative_roi(frame, focus_cfg)
            fx, fy, fw, fh = roi_rect_from_shape(frame.shape, focus_cfg)

            data = {
                "ready": True,
                "frame_shape": {
                    "height": frame.shape[0],
                    "width": frame.shape[1]
                },
                "roi_config": cfg["roi"],
                "roi_pixels": {
                    "x": x,
                    "y": y,
                    "w": rw,
                    "h": rh
                },
                "focus_roi_config": focus_cfg,
                "focus_roi_pixels": {
                    "x": fx,
                    "y": fy,
                    "w": fw,
                    "h": fh
                },
                "metrics": roi_metrics(roi),
                "focus_metrics": roi_metrics(focus_roi)
            }

            self.send_json(data)

        def capture_template(self, query):
            try:
                label = int(query.get("label", [""])[0])
                count = int(query.get("count", ["1"])[0])
                interval = float(query.get("interval", ["0.10"])[0])
            except Exception:
                self.send_json({"ok": False, "error": "Invalid label/count/interval"}, status=400)
                return

            if label < 0 or label > 36:
                self.send_json({"ok": False, "error": "Label must be between 0 and 36"}, status=400)
                return

            count = max(1, min(count, 200))
            interval = max(0.03, min(interval, 1.0))

            out_dir = Path("data/templates") / str(label)
            out_dir.mkdir(parents=True, exist_ok=True)

            saved = []
            metrics_list = []

            for i in range(count):
                frame = hub.get_frame()
                if frame is None:
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = out_dir / f"{label}_{timestamp}_{i:04d}.png"
                roi = save_roi_image(frame, cfg, path)
                saved.append(str(path))
                metrics_list.append(roi_metrics(roi))

                time.sleep(interval)

            self.send_json({
                "ok": True,
                "type": "template",
                "label": label,
                "requested": count,
                "saved_count": len(saved),
                "first_file": saved[0] if saved else None,
                "last_file": saved[-1] if saved else None,
                "counts": count_templates(),
                "last_metrics": metrics_list[-1] if metrics_list else None
            })

        def capture_empty(self, query):
            try:
                count = int(query.get("count", ["30"])[0])
                interval = float(query.get("interval", ["0.08"])[0])
            except Exception:
                self.send_json({"ok": False, "error": "Invalid count/interval"}, status=400)
                return

            count = max(1, min(count, 200))
            interval = max(0.03, min(interval, 1.0))

            out_dir = Path("data/empty")
            out_dir.mkdir(parents=True, exist_ok=True)

            saved = []
            metrics_list = []

            for i in range(count):
                frame = hub.get_frame()
                if frame is None:
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = out_dir / f"empty_{timestamp}_{i:04d}.png"
                roi = save_roi_image(frame, cfg, path)
                saved.append(str(path))
                metrics_list.append(roi_metrics(roi))

                time.sleep(interval)

            self.send_json({
                "ok": True,
                "type": "empty",
                "requested": count,
                "saved_count": len(saved),
                "first_file": saved[0] if saved else None,
                "last_file": saved[-1] if saved else None,
                "last_metrics": metrics_list[-1] if metrics_list else None
            })

    return RouletteWebHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    Path("data/templates").mkdir(parents=True, exist_ok=True)
    Path("data/empty").mkdir(parents=True, exist_ok=True)

    hub = CameraHub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("Roulette Vision web server is running:")
    print(f"  Local:   http://127.0.0.1:{args.port}")
    print(f"  Network: http://192.168.1.251:{args.port}")
    print("")
    print("Open the Network URL from Windows.")
    print("Use the web page to capture templates while watching the live camera.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping web server...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
