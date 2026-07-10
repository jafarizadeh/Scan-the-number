import argparse
import csv
import json
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_metrics, roi_rect_from_shape
from .vision import TemplateMatcher, EmptyMatcher

try:
    from .digit_recognizer import DigitRecognizer
except Exception:
    DigitRecognizer = None


class HybridHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)

        self.template_matcher = TemplateMatcher("data/templates", cfg["matching"])
        self.empty_matcher = EmptyMatcher("data/empty", cfg["matching"])
        self.digit_recognizer = DigitRecognizer("data/templates") if DigitRecognizer else None

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None

        self.events = deque(maxlen=80)
        self.review_items = deque(maxlen=80)

        self.armed = False
        self.empty_counter = 0
        self.votes = deque(maxlen=int(cfg["decision"]["vote_window"]))
        self.round_index = 0

        self.bright_frames_since_empty = 0
        self.last_status = {
            "ready": False,
            "state": "STARTING",
            "armed": False,
        }

        self.running = False
        self.thread = None

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)
        Path("data/review").mkdir(parents=True, exist_ok=True)

        self.log_path = Path("data/logs/hybrid_events.csv")
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round_index", "timestamp", "type", "number",
                    "score", "margin", "agreement",
                    "whole_label", "whole_score", "whole_margin",
                    "digit_label", "digit_score", "digit_margin",
                    "frame_path", "roi_path"
                ])

    def start(self):
        print("Whole-template labels:", self.template_matcher.labels())
        if self.digit_recognizer:
            print("Digit recognizer counts:", self.digit_recognizer.counts())
            print("Digit recognizer ready:", self.digit_recognizer.is_ready())
        else:
            print("Digit recognizer not available.")

        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def classify_state(self, roi):
        metrics = roi_metrics(roi)
        state_cfg = self.cfg["state"]

        empty_score = self.empty_matcher.score(roi)

        if empty_score is not None:
            if empty_score >= float(state_cfg["empty_template_score_min"]):
                return "EMPTY", metrics, empty_score

        is_heuristic_empty = (
            metrics["white_ratio"] <= float(state_cfg["empty_white_ratio_max"])
            and metrics["std"] <= float(state_cfg["empty_std_max"])
        )

        is_bright = (
            metrics["white_ratio"] >= float(state_cfg["bright_white_ratio_min"])
            and metrics["std"] >= float(state_cfg["bright_std_min"])
        )

        if is_heuristic_empty:
            return "EMPTY", metrics, empty_score

        if is_bright:
            return "BRIGHT", metrics, empty_score

        return "DIMMED", metrics, empty_score

    def hybrid_predict(self, roi):
        whole = self.template_matcher.predict(roi) if self.template_matcher.is_ready() else None

        digit = None
        if self.digit_recognizer and self.digit_recognizer.is_ready():
            try:
                digit = self.digit_recognizer.predict(roi)
            except Exception:
                digit = None

        score_min = float(self.cfg["matching"]["score_min"])
        margin_min = float(self.cfg["matching"]["margin_min"])

        decision = {
            "label": None,
            "score": 0.0,
            "margin": 0.0,
            "usable": False,
            "reason": "no_prediction",
            "whole": whole,
            "digit": digit,
        }

        whole_ok = (
            whole is not None
            and whole.get("score", -1) >= score_min
            and whole.get("margin", -1) >= margin_min
        )

        digit_ok = (
            digit is not None
            and digit.get("score", -1) >= score_min
            and digit.get("margin", -1) >= margin_min
        )

        if whole and digit and whole["label"] == digit["label"]:
            # Strongest case: both recognizers agree.
            label = whole["label"]
            score = max(whole["score"], digit["score"])
            margin = max(whole["margin"], digit["margin"])

            # Agreement lets us accept slightly weaker margin, but not very weak scores.
            usable = (
                whole["score"] >= 0.72
                and digit["score"] >= 0.72
                and max(whole["margin"], digit["margin"]) >= 0.06
            )

            decision.update({
                "label": label,
                "score": score,
                "margin": margin,
                "usable": usable,
                "reason": "whole_digit_agree" if usable else "agree_but_weak",
            })
            return decision

        if whole_ok:
            decision.update({
                "label": whole["label"],
                "score": whole["score"],
                "margin": whole["margin"],
                "usable": True,
                "reason": "whole_strong",
            })
            return decision

        if digit_ok:
            decision.update({
                "label": digit["label"],
                "score": digit["score"],
                "margin": digit["margin"],
                "usable": True,
                "reason": "digit_strong",
            })
            return decision

        # Preview only: choose the higher-confidence visible candidate.
        candidates = []
        if whole:
            candidates.append(("whole_preview", whole))
        if digit:
            candidates.append(("digit_preview", digit))

        if candidates:
            source, pred = max(candidates, key=lambda x: x[1].get("score", -1))
            decision.update({
                "label": pred["label"],
                "score": pred.get("score", 0.0),
                "margin": pred.get("margin", 0.0),
                "usable": False,
                "reason": source + "_weak",
            })

        return decision

    def try_confirm(self, frame, roi):
        if not self.votes:
            return None

        labels = [v["label"] for v in self.votes]
        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]

        total = len(self.votes)
        agreement = top_count / max(1, total)

        min_votes = int(self.cfg["decision"]["min_votes"])
        min_agreement = float(self.cfg["decision"]["min_agreement"])

        if top_count < min_votes or agreement < min_agreement:
            return None

        selected = [v for v in self.votes if v["label"] == top_label]

        avg_score = sum(v["score"] for v in selected) / len(selected)
        avg_margin = sum(v["margin"] for v in selected) / len(selected)

        self.round_index += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        frame_path = Path("data/evidence") / f"round_{self.round_index:06d}_{top_label}_frame_{timestamp}.png"
        roi_path = Path("data/evidence") / f"round_{self.round_index:06d}_{top_label}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(roi_path), roi)

        last = selected[-1]
        whole = last.get("whole") or {}
        digit = last.get("digit") or {}

        event = {
            "round_index": self.round_index,
            "timestamp": timestamp,
            "type": "confirmed",
            "number": top_label,
            "score": avg_score,
            "margin": avg_margin,
            "agreement": agreement,
            "reason": last.get("reason"),
            "frame_path": str(frame_path),
            "roi_path": str(roi_path),
            "whole": whole,
            "digit": digit,
        }

        self.events.appendleft(event)

        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.round_index, timestamp, "confirmed", top_label,
                round(avg_score, 4), round(avg_margin, 4), round(agreement, 4),
                whole.get("label"), whole.get("score"), whole.get("margin"),
                digit.get("label"), digit.get("score"), digit.get("margin"),
                str(frame_path), str(roi_path)
            ])

        self.armed = False
        self.votes.clear()
        self.bright_frames_since_empty = 0

        return event

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.03)
                continue

            roi = crop_relative_roi(frame, self.cfg["roi"])
            state, metrics, empty_score = self.classify_state(roi)

            decision = None
            event = None

            if state == "EMPTY":
                self.empty_counter += 1
                self.votes.clear()
                self.bright_frames_since_empty = 0

                if self.empty_counter >= int(self.cfg["state"]["empty_confirm_frames"]):
                    self.armed = True

            else:
                self.empty_counter = 0

            if state == "BRIGHT":
                decision = self.hybrid_predict(roi)

                if self.armed:
                    self.bright_frames_since_empty += 1

                    if decision and decision.get("usable"):
                        self.votes.append(decision)

                    event = self.try_confirm(frame, roi)

            status = {
                "ready": True,
                "state": state,
                "armed": self.armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "decision": decision,
                "votes": list(self.votes),
                "vote_count": len(self.votes),
                "events": list(self.events),
                "review_items": list(self.review_items),
                "event": event,
                "thresholds": {
                    "score_min": float(self.cfg["matching"]["score_min"]),
                    "margin_min": float(self.cfg["matching"]["margin_min"]),
                    "min_votes": int(self.cfg["decision"]["min_votes"]),
                    "min_agreement": float(self.cfg["decision"]["min_agreement"]),
                }
            }

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_roi = roi.copy()
                self.last_status = status

            time.sleep(delay)

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_roi(self):
        with self.lock:
            return None if self.latest_roi is None else self.latest_roi.copy()

    def get_status(self):
        with self.lock:
            return json.loads(json.dumps(self.last_status, default=str))

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.camera.close()


def encode_jpeg(frame, quality=85):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


def draw_overlay(frame, cfg, status):
    x, y, rw, rh = roi_rect_from_shape(frame.shape, cfg["roi"])
    cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

    txt = f"STATE: {status.get('state')} | ARMED: {status.get('armed')}"

    d = status.get("decision")
    if d and d.get("label") is not None:
        txt += f" | HYBRID: {d.get('label')} s={d.get('score',0):.3f} m={d.get('margin',0):.3f} {d.get('reason')}"

    cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 1100), 55), (0, 0, 0), -1)
    cv2.putText(frame, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def make_handler(hub, cfg):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path

            if path == "/":
                self.index()
            elif path == "/stream.mjpg":
                self.stream("full")
            elif path == "/roi.mjpg":
                self.stream("roi")
            elif path == "/api/status":
                self.json(hub.get_status())
            else:
                self.send_error(404)

        def json(self, data):
            body = json.dumps(data, indent=2, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def index(self):
            html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Roulette Vision - Hybrid Test</title>
<style>
body { margin:0; background:#101418; color:#f2f2f2; font-family:Arial,sans-serif; }
header { padding:14px 20px; background:#1b232b; border-bottom:1px solid #34404a; }
h1 { margin:0; font-size:22px; }
.grid { display:grid; grid-template-columns:minmax(680px,1fr) 480px; gap:16px; padding:16px; }
.card { background:#1b232b; border:1px solid #34404a; border-radius:10px; padding:12px; }
img { max-width:100%; border-radius:8px; border:1px solid #34404a; background:#000; }
.roi { width:100%; }
.statusBox { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; }
.pill { background:#0b0e11; border:1px solid #34404a; border-radius:8px; padding:10px; font-size:15px; }
.big { font-size:27px; font-weight:bold; }
.ok { color:#9ef09e; }
.warn { color:#ffd27a; }
.bad { color:#ff9999; }
pre { background:#0b0e11; padding:10px; border-radius:8px; overflow:auto; font-size:12px; max-height:360px; }
table { width:100%; border-collapse:collapse; font-size:14px; }
td, th { border-bottom:1px solid #34404a; padding:6px; text-align:left; }
</style>
</head>
<body>
<header><h1>Roulette Vision - Hybrid Real Test</h1></header>

<div class="grid">
  <div class="card">
    <h2>Full Camera Frame</h2>
    <img src="/stream.mjpg">
  </div>

  <div class="card">
    <h2>Result ROI</h2>
    <img class="roi" src="/roi.mjpg">

    <h2>Live Status</h2>
    <div class="statusBox">
      <div class="pill">State<br><span id="state" class="big warn">---</span></div>
      <div class="pill">Armed<br><span id="armed" class="big">---</span></div>
      <div class="pill">Hybrid<br><span id="hybrid" class="big">---</span></div>
      <div class="pill">Votes<br><span id="votes" class="big">---</span></div>
    </div>

    <h2>Recent Confirmed Results</h2>
    <table>
      <thead><tr><th>Round</th><th>Number</th><th>Score</th><th>Margin</th><th>Reason</th></tr></thead>
      <tbody id="events"></tbody>
    </table>

    <h2>Debug</h2>
    <pre id="debug">loading...</pre>
  </div>
</div>

<script>
function fmt(x,d=3){ if(x===null||x===undefined)return '---'; if(typeof x==='number')return x.toFixed(d); return x; }

async function refresh(){
  const res = await fetch('/api/status');
  const data = await res.json();

  document.getElementById('state').textContent = data.state || '---';
  document.getElementById('armed').textContent = data.armed ? 'YES' : 'NO';
  document.getElementById('votes').textContent = (data.vote_count || 0) + ' / ' + data.thresholds.min_votes;

  if(data.decision && data.decision.label !== null){
    let prefix = data.decision.usable ? '' : 'REJECT ';
    document.getElementById('hybrid').textContent =
      prefix + data.decision.label + ' / ' +
      fmt(data.decision.score,3) + ' / m=' +
      fmt(data.decision.margin,3);
  } else {
    document.getElementById('hybrid').textContent = '---';
  }

  const tbody = document.getElementById('events');
  tbody.innerHTML = '';
  if(data.events){
    for(const e of data.events.slice(0,12)){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${e.round_index}</td><td><b>${e.number}</b></td><td>${fmt(e.score)}</td><td>${fmt(e.margin)}</td><td>${e.reason || ''}</td>`;
      tbody.appendChild(tr);
    }
  }

  const debug = {
    state: data.state,
    armed: data.armed,
    decision: data.decision,
    vote_count: data.vote_count,
    thresholds: data.thresholds,
    event: data.event
  };
  document.getElementById('debug').textContent = JSON.stringify(debug,null,2);
}

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
                if jpg is None:
                    continue

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

    hub = HybridHub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("Hybrid real-test web server running:")
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
