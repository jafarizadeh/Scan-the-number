import argparse
import csv
import json
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from http.server import ThreadingHTTPServer

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_metrics, EmptyMatcher
from .web_classifier import make_handler
from .cnn_number_model import CNNNumberModel


class CNNFastNoDupHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)
        self.empty_matcher = EmptyMatcher("data/empty", cfg["matching"])

        model_cfg = cfg.get("cnn_model", {})
        self.model = CNNNumberModel(
            model_cfg.get("model_path", "data/models/roulette_mobilenet_best_single.onnx"),
            img_size=int(model_cfg.get("img_size", 160))
        )

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.status = {"ready": False, "state": "STARTING"}

        self.mode = "WAIT_RESULT"
        self.capture_frames = 0
        self.capture_votes = []
        self.last_output_number = None

        self.round_index = 0
        self.review_index = 0

        self.events = deque(maxlen=120)
        self.review_items = deque(maxlen=120)

        self.running = False
        self.thread = None

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)
        Path("data/review").mkdir(parents=True, exist_ok=True)

        self.event_log = Path("data/logs/cnn_realtest_events.csv")
        self.review_log = Path("data/logs/cnn_realtest_review.csv")

        if not self.event_log.exists():
            with self.event_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "round_index", "timestamp", "number", "score", "margin",
                    "agreement", "votes", "total_frames", "reason",
                    "roi_path", "frame_path"
                ])

        if not self.review_log.exists():
            with self.review_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "review_index", "timestamp", "suggested_number", "score", "margin",
                    "agreement", "votes", "total_frames", "reason",
                    "roi_path", "frame_path"
                ])

    def start(self):
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

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        white_digit_ratio = float(((v >= 145) & (s <= 130)).mean())
        green_ratio = float(((h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)).mean())
        red_ratio = float(((((h <= 12) | (h >= 165)) & (s >= 45) & (v >= 45))).mean())

        visible_by_gray = (
            metrics["white_ratio"] >= float(state_cfg["bright_white_ratio_min"])
            and metrics["std"] >= float(state_cfg["bright_std_min"])
        )

        visible_by_color = (
            white_digit_ratio >= 0.004
            and (
                green_ratio >= 0.035
                or red_ratio >= 0.035
                or metrics["std"] >= 20.0
            )
        )

        if visible_by_gray or visible_by_color:
            return "RESULT_VISIBLE", metrics, empty_score

        return "TRANSITION", metrics, empty_score

    def reset_capture(self):
        self.capture_frames = 0
        self.capture_votes = []

    def save_pair(self, folder, prefix, number, frame, roi):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        frame_path = Path(folder) / f"{prefix}_{number}_frame_{timestamp}.png"
        roi_path = Path(folder) / f"{prefix}_{number}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(roi_path), roi)

        return timestamp, frame_path, roi_path

    def finalize_capture(self, frame, roi, reason):
        cfg = self.cfg.get("cnn_fast_nodup_test", {})

        if not self.capture_votes:
            self.mode = "WAIT_RESULT"
            self.reset_capture()
            return None, None

        labels = [v["label"] for v in self.capture_votes]
        counts = Counter(labels)
        suggested, vote_count = counts.most_common(1)[0]
        agreement = vote_count / max(1, len(labels))

        selected = [v for v in self.capture_votes if v["label"] == suggested]
        avg_score = sum(float(v["score"]) for v in selected) / len(selected)
        avg_margin = sum(float(v["margin"]) for v in selected) / len(selected)
        total_frames = len(self.capture_votes)

        # فعلاً عدد تکراری پشت سر هم را ثبت نمی‌کنیم.
        if self.last_output_number is not None and int(suggested) == int(self.last_output_number):
            self.mode = "WAIT_RESULT"
            self.reset_capture()
            return None, None

        min_votes = int(cfg.get("min_confirm_votes", 3))
        min_agreement = float(cfg.get("min_confirm_agreement", 0.70))
        min_score = float(cfg.get("min_confirm_score", 0.85))
        min_margin = float(cfg.get("min_confirm_margin", 0.20))

        confirmed = (
            vote_count >= min_votes
            and agreement >= min_agreement
            and avg_score >= min_score
            and avg_margin >= min_margin
        )

        if confirmed:
            self.round_index += 1
            timestamp, frame_path, roi_path = self.save_pair(
                "data/evidence",
                f"cnn_fast_round_{self.round_index:06d}",
                suggested,
                frame,
                roi
            )

            event = {
                "round_index": self.round_index,
                "timestamp": timestamp,
                "number": int(suggested),
                "score": float(avg_score),
                "margin": float(avg_margin),
                "agreement": float(agreement),
                "votes": int(vote_count),
                "total_frames": int(total_frames),
                "reason": reason,
                "frame_path": str(frame_path),
                "roi_path": str(roi_path),
            }

            self.events.appendleft(event)
            self.last_output_number = int(suggested)

            with self.event_log.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    event["round_index"],
                    event["timestamp"],
                    event["number"],
                    round(event["score"], 4),
                    round(event["margin"], 4),
                    round(event["agreement"], 4),
                    event["votes"],
                    event["total_frames"],
                    event["reason"],
                    event["roi_path"],
                    event["frame_path"],
                ])

            self.mode = "WAIT_RESULT"
            self.reset_capture()
            return event, None

        self.review_index += 1
        timestamp, frame_path, roi_path = self.save_pair(
            "data/review",
            f"cnn_fast_review_{self.review_index:06d}",
            suggested,
            frame,
            roi
        )

        item = {
            "review_index": self.review_index,
            "timestamp": timestamp,
            "suggested_number": int(suggested),
            "score": float(avg_score),
            "margin": float(avg_margin),
            "agreement": float(agreement),
            "votes": int(vote_count),
            "total_frames": int(total_frames),
            "reason": reason,
            "frame_path": str(frame_path),
            "roi_path": str(roi_path),
        }

        self.review_items.appendleft(item)
        self.last_output_number = int(suggested)

        with self.review_log.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                item["review_index"],
                item["timestamp"],
                item["suggested_number"],
                round(item["score"], 4),
                round(item["margin"], 4),
                round(item["agreement"], 4),
                item["votes"],
                item["total_frames"],
                item["reason"],
                item["roi_path"],
                item["frame_path"],
            ])

        self.mode = "WAIT_RESULT"
        self.reset_capture()
        return None, item

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        cfg = self.cfg.get("cnn_fast_nodup_test", {})
        capture_max_frames = int(cfg.get("capture_max_frames", 4))

        last_decision = None

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

            if self.mode == "WAIT_RESULT":
                if state == "RESULT_VISIBLE":
                    decision = self.model.predict(roi)
                    last_decision = decision

                    label = int(decision["label"])

                    # اگر همان عدد قبلی است، فعلاً کلاً نادیده بگیر.
                    if self.last_output_number is not None and label == int(self.last_output_number):
                        pass
                    else:
                        self.mode = "CAPTURE"
                        self.capture_frames = 1
                        self.capture_votes = [decision]

            elif self.mode == "CAPTURE":
                if state == "RESULT_VISIBLE":
                    decision = self.model.predict(roi)
                    last_decision = decision

                    self.capture_frames += 1
                    self.capture_votes.append(decision)

                    if self.capture_frames >= capture_max_frames:
                        event, review = self.finalize_capture(frame, roi, "fast_capture")
                else:
                    event, review = self.finalize_capture(frame, roi, "result_ended")

            armed = self.mode in ("WAIT_RESULT", "CAPTURE")

            status = {
                "ready": True,
                "state": state,
                "mode": self.mode,
                "armed": armed,
                "last_output_number": self.last_output_number,
                "metrics": metrics,
                "empty_score": empty_score,
                "decision": decision or last_decision,
                "votes": list(self.capture_votes),
                "vote_count": len(self.capture_votes),
                "soft_vote_count": len(self.capture_votes),
                "events": list(self.events),
                "review_items": list(self.review_items),
                "event": event,
                "review": review,
                "thresholds": {
                    "cnn_fast_nodup_test": self.cfg.get("cnn_fast_nodup_test", {})
                }
            }

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_roi = roi.copy()
                self.status = status

            time.sleep(delay)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    hub = CNNFastNoDupHub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("CNN Fast No-Duplicate web server running:")
    print(f"  http://192.168.1.251:{args.port}")
    print("")
    print("Logic:")
    print("  Scan as soon as a clear result is visible")
    print("  Different number: accepted immediately")
    print("  Same consecutive number: ignored for this version")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
