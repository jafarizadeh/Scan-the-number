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
from .classifier import NumberClassifier


class ClassifierHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)

        self.template_matcher = TemplateMatcher("data/templates", cfg["matching"])
        self.empty_matcher = EmptyMatcher("data/empty", cfg["matching"])
        self.classifier = NumberClassifier("data/templates", top_k=40)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.status = {"ready": False, "state": "STARTING"}

        self.armed = False
        self.empty_counter = 0
        self.result_frames = 0

        self.vote_window = deque(maxlen=int(cfg["decision"].get("vote_window", 12)))
        self.round_index = 0

        self.events = deque(maxlen=80)
        self.review_items = deque(maxlen=80)

        self.running = False
        self.thread = None

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)
        Path("data/review").mkdir(parents=True, exist_ok=True)

        self.event_log = Path("data/logs/classifier_events.csv")
        self.review_log = Path("data/logs/classifier_review.csv")

        if not self.event_log.exists():
            with self.event_log.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round_index", "timestamp", "number", "score", "margin",
                    "agreement", "reason", "roi_path", "frame_path"
                ])

        if not self.review_log.exists():
            with self.review_log.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "review_index", "timestamp", "suggested_number", "score",
                    "margin", "reason", "roi_path", "frame_path"
                ])

    def start(self):
        print("Template labels:", self.template_matcher.labels())
        print("Classifier counts:", self.classifier.counts())
        print("Classifier ready:", self.classifier.is_ready())

        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def classify_state(self, roi):
        metrics = roi_metrics(roi)
        state_cfg = self.cfg["state"]

        empty_score = self.empty_matcher.score(roi)

        if empty_score is not None and empty_score >= float(state_cfg["empty_template_score_min"]):
            return "EMPTY", metrics, empty_score

        heuristic_empty = (
            metrics["white_ratio"] <= float(state_cfg["empty_white_ratio_max"])
            and metrics["std"] <= float(state_cfg["empty_std_max"])
        )

        if heuristic_empty:
            return "EMPTY", metrics, empty_score

        visible_result = (
            metrics["white_ratio"] >= float(state_cfg["bright_white_ratio_min"])
            and metrics["std"] >= float(state_cfg["bright_std_min"])
        )

        if visible_result:
            return "RESULT_VISIBLE", metrics, empty_score

        return "TRANSITION", metrics, empty_score

    def make_decision(self, roi):
        template_pred = self.template_matcher.predict(roi) if self.template_matcher.is_ready() else None
        classifier_pred = self.classifier.predict(roi) if self.classifier.is_ready() else None

        score_min = float(self.cfg.get("classifier", {}).get("score_min", 0.72))
        margin_min = float(self.cfg.get("classifier", {}).get("margin_min", 0.05))
        topk_min = float(self.cfg.get("classifier", {}).get("topk_min_agreement", 0.55))

        decision = {
            "label": None,
            "score": 0.0,
            "margin": 0.0,
            "usable": False,
            "reason": "no_prediction",
            "template": template_pred,
            "classifier": classifier_pred
        }

        template_label = template_pred.get("label") if template_pred else None
        classifier_label = classifier_pred.get("label") if classifier_pred else None

        classifier_ok = (
            classifier_pred is not None
            and classifier_pred["score"] >= score_min
            and classifier_pred["margin"] >= margin_min
            and classifier_pred["topk_agreement"] >= topk_min
        )

        template_ok = (
            template_pred is not None
            and template_pred["score"] >= float(self.cfg["matching"]["score_min"])
            and template_pred["margin"] >= float(self.cfg["matching"]["margin_min"])
        )

        if template_label is not None and classifier_label is not None and template_label == classifier_label:
            score = max(template_pred["score"], classifier_pred["score"])
            margin = max(template_pred["margin"], classifier_pred["margin"])

            usable = (
                score >= 0.72
                and margin >= 0.045
                and classifier_pred["topk_agreement"] >= 0.45
            )

            decision.update({
                "label": int(classifier_label),
                "score": float(score),
                "margin": float(margin),
                "usable": bool(usable),
                "reason": "template_classifier_agree" if usable else "agree_but_weak"
            })
            return decision

        if classifier_ok:
            decision.update({
                "label": int(classifier_label),
                "score": float(classifier_pred["score"]),
                "margin": float(classifier_pred["margin"]),
                "usable": True,
                "reason": "classifier_strong"
            })
            return decision

        if template_ok:
            decision.update({
                "label": int(template_label),
                "score": float(template_pred["score"]),
                "margin": float(template_pred["margin"]),
                "usable": True,
                "reason": "template_strong"
            })
            return decision

        candidates = []

        if classifier_pred:
            candidates.append(("classifier_preview", classifier_pred))

        if template_pred:
            candidates.append(("template_preview", template_pred))

        if candidates:
            reason, pred = max(candidates, key=lambda x: x[1].get("score", -1))
            decision.update({
                "label": int(pred["label"]),
                "score": float(pred.get("score", 0.0)),
                "margin": float(pred.get("margin", 0.0)),
                "usable": False,
                "reason": reason
            })

        return decision

    def save_images(self, folder, prefix, frame, roi, number):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = Path(folder) / f"{prefix}_{number}_frame_{timestamp}.png"
        roi_path = Path(folder) / f"{prefix}_{number}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(roi_path), roi)

        return timestamp, frame_path, roi_path

    def confirm_if_ready(self, frame, roi):
        if not self.vote_window:
            return None

        labels = [v["label"] for v in self.vote_window]
        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]

        min_votes = int(self.cfg["decision"].get("min_votes", 8))
        min_agreement = float(self.cfg["decision"].get("min_agreement", 0.75))

        total = len(self.vote_window)
        agreement = top_count / max(1, total)

        if top_count < min_votes or agreement < min_agreement:
            return None

        selected = [v for v in self.vote_window if v["label"] == top_label]

        avg_score = sum(v["score"] for v in selected) / len(selected)
        avg_margin = sum(v["margin"] for v in selected) / len(selected)

        self.round_index += 1
        timestamp, frame_path, roi_path = self.save_images(
            "data/evidence",
            f"round_{self.round_index:06d}",
            frame,
            roi,
            top_label
        )

        reason = selected[-1].get("reason")

        event = {
            "round_index": self.round_index,
            "timestamp": timestamp,
            "number": int(top_label),
            "score": float(avg_score),
            "margin": float(avg_margin),
            "agreement": float(agreement),
            "reason": reason,
            "frame_path": str(frame_path),
            "roi_path": str(roi_path)
        }

        self.events.appendleft(event)

        with self.event_log.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                event["round_index"],
                event["timestamp"],
                event["number"],
                round(event["score"], 4),
                round(event["margin"], 4),
                round(event["agreement"], 4),
                event["reason"],
                event["roi_path"],
                event["frame_path"]
            ])

        self.armed = False
        self.vote_window.clear()
        self.result_frames = 0

        return event

    def create_review_if_needed(self, frame, roi, decision):
        max_frames = int(self.cfg.get("review", {}).get("max_result_frames_without_confirm", 45))

        if self.result_frames < max_frames:
            return None

        suggested = decision.get("label") if decision else None
        score = decision.get("score", 0.0) if decision else 0.0
        margin = decision.get("margin", 0.0) if decision else 0.0
        reason = decision.get("reason", "no_decision") if decision else "no_decision"

        review_index = len(self.review_items) + 1
        timestamp, frame_path, roi_path = self.save_images(
            "data/review",
            f"review_{review_index:06d}",
            frame,
            roi,
            suggested if suggested is not None else "unknown"
        )

        item = {
            "review_index": review_index,
            "timestamp": timestamp,
            "suggested_number": suggested,
            "score": float(score),
            "margin": float(margin),
            "reason": reason,
            "frame_path": str(frame_path),
            "roi_path": str(roi_path)
        }

        self.review_items.appendleft(item)

        with self.review_log.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                review_index,
                timestamp,
                suggested,
                round(float(score), 4),
                round(float(margin), 4),
                reason,
                str(roi_path),
                str(frame_path)
            ])

        self.armed = False
        self.vote_window.clear()
        self.result_frames = 0

        return item

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
            review = None

            if state == "EMPTY":
                self.empty_counter += 1
                self.result_frames = 0
                self.vote_window.clear()

                if self.empty_counter >= int(self.cfg["state"].get("empty_confirm_frames", 2)):
                    self.armed = True

            else:
                self.empty_counter = 0

            if state == "RESULT_VISIBLE":
                decision = self.make_decision(roi)

                if self.armed:
                    self.result_frames += 1

                    if decision and decision.get("usable"):
                        self.vote_window.append(decision)

                    event = self.confirm_if_ready(frame, roi)

                    if event is None:
                        review = self.create_review_if_needed(frame, roi, decision)

            status = {
                "ready": True,
                "state": state,
                "armed": self.armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "decision": decision,
                "votes": list(self.vote_window),
                "vote_count": len(self.vote_window),
                "events": list(self.events),
                "review_items": list(self.review_items),
                "event": event,
                "review": review,
                "thresholds": {
                    "classifier": self.cfg.get("classifier", {}),
                    "matching": self.cfg.get("matching", {}),
                    "decision": self.cfg.get("decision", {}),
                    "review": self.cfg.get("review", {})
                }
            }

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_roi = roi.copy()
                self.status = status

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
    x, y, rw, rh = roi_rect_from_shape(frame.shape, cfg["roi"])
    cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

    text = f"STATE: {status.get('state')} | ARMED: {status.get('armed')}"

    decision = status.get("decision")
    if decision and decision.get("label") is not None:
        prefix = "" if decision.get("usable") else "REVIEW "
        text += f" | {prefix}{decision.get('label')} s={decision.get('score', 0):.3f} m={decision.get('margin', 0):.3f}"

    cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 1100), 55), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

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
                self.send_json(hub.get_status())
            else:
                self.send_error(404)

        def send_json(self, data):
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
<title>Roulette Vision - Classifier Test</title>
<style>
body { margin:0; background:#101418; color:#f2f2f2; font-family:Arial,sans-serif; }
header { padding:14px 20px; background:#1b232b; border-bottom:1px solid #34404a; }
h1 { margin:0; font-size:22px; }
.grid { display:grid; grid-template-columns:minmax(680px,1fr) 510px; gap:16px; padding:16px; }
.card { background:#1b232b; border:1px solid #34404a; border-radius:10px; padding:12px; }
img { max-width:100%; border-radius:8px; border:1px solid #34404a; background:#000; }
.roi { width:100%; }
.statusBox { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px; }
.pill { background:#0b0e11; border:1px solid #34404a; border-radius:8px; padding:10px; font-size:15px; }
.big { font-size:26px; font-weight:bold; }
.ok { color:#9ef09e; }
.warn { color:#ffd27a; }
.bad { color:#ff9999; }
pre { background:#0b0e11; padding:10px; border-radius:8px; overflow:auto; font-size:12px; max-height:340px; }
table { width:100%; border-collapse:collapse; font-size:14px; }
td, th { border-bottom:1px solid #34404a; padding:6px; text-align:left; }
</style>
</head>
<body>
<header><h1>Roulette Vision - Classifier Real Test</h1></header>

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
      <div class="pill">Decision<br><span id="decision" class="big">---</span></div>
      <div class="pill">Votes<br><span id="votes" class="big">---</span></div>
    </div>

    <h2>Confirmed Results</h2>
    <table>
      <thead><tr><th>Round</th><th>Number</th><th>Score</th><th>Margin</th><th>Reason</th></tr></thead>
      <tbody id="events"></tbody>
    </table>

    <h2>Needs Review</h2>
    <table>
      <thead><tr><th>ID</th><th>Suggested</th><th>Score</th><th>Margin</th><th>Reason</th></tr></thead>
      <tbody id="reviews"></tbody>
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
  document.getElementById('votes').textContent = (data.vote_count || 0) + ' / ' + data.thresholds.decision.min_votes;

  if(data.decision && data.decision.label !== null){
    const prefix = data.decision.usable ? '' : 'REVIEW ';
    document.getElementById('decision').textContent =
      prefix + data.decision.label + ' / ' +
      fmt(data.decision.score,3) + ' / m=' + fmt(data.decision.margin,3);
  } else {
    document.getElementById('decision').textContent = '---';
  }

  const events = document.getElementById('events');
  events.innerHTML = '';
  if(data.events){
    for(const e of data.events.slice(0,10)){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${e.round_index}</td><td><b>${e.number}</b></td><td>${fmt(e.score)}</td><td>${fmt(e.margin)}</td><td>${e.reason || ''}</td>`;
      events.appendChild(tr);
    }
  }

  const reviews = document.getElementById('reviews');
  reviews.innerHTML = '';
  if(data.review_items){
    for(const r of data.review_items.slice(0,10)){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${r.review_index}</td><td><b>${r.suggested_number}</b></td><td>${fmt(r.score)}</td><td>${fmt(r.margin)}</td><td>${r.reason || ''}</td>`;
      reviews.appendChild(tr);
    }
  }

  const debug = {
    state: data.state,
    armed: data.armed,
    decision: data.decision,
    vote_count: data.vote_count,
    thresholds: data.thresholds,
    last_event: data.event,
    last_review: data.review
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

    hub = ClassifierHub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("Classifier real-test web server running:")
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
