import argparse
import csv
import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from http.server import ThreadingHTTPServer

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_metrics
from .web_classifier import make_handler
from .cnn_number_model import CNNNumberModel


class CNNFastV2Hub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)

        model_cfg = cfg.get("cnn_model", {})
        self.model = CNNNumberModel(
            model_cfg.get("model_path", "data/models/roulette_mobilenet_best_single.onnx"),
            img_size=int(model_cfg.get("img_size", 160))
        )

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.status = {"ready": False, "state": "STARTING"}

        self.last_output_number = None
        self.candidate_label = None
        self.candidate_predictions = []
        self.candidate_streak = 0

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

        if (
            white_digit_ratio >= 0.003
            or green_ratio >= 0.03
            or red_ratio >= 0.03
            or metrics["std"] >= 18.0
        ):
            return "RESULT_VISIBLE"

        return "LOW_SIGNAL"

    def save_pair(self, folder, prefix, number, frame, roi):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        frame_path = Path(folder) / f"{prefix}_{number}_frame_{timestamp}.png"
        roi_path = Path(folder) / f"{prefix}_{number}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(roi_path), roi)

        return timestamp, frame_path, roi_path

    def finalize_candidate(self, frame, roi):
        cfg = self.cfg.get("cnn_fast_v2_test", {})

        preds = list(self.candidate_predictions)

        if not preds:
            self.reset_candidate()
            return None, None

        label = int(self.candidate_label)

        avg_score = sum(float(p["score"]) for p in preds) / len(preds)
        avg_margin = sum(float(p["margin"]) for p in preds) / len(preds)

        min_score = float(cfg.get("min_confirm_score", 0.85))
        min_margin = float(cfg.get("min_confirm_margin", 0.20))

        confirmed = avg_score >= min_score and avg_margin >= min_margin

        if confirmed:
            self.round_index += 1

            timestamp, frame_path, roi_path = self.save_pair(
                "data/evidence",
                f"cnn_fast_v2_round_{self.round_index:06d}",
                label,
                frame,
                roi
            )

            event = {
                "round_index": self.round_index,
                "timestamp": timestamp,
                "number": label,
                "score": float(avg_score),
                "margin": float(avg_margin),
                "agreement": 1.0,
                "votes": len(preds),
                "total_frames": len(preds),
                "reason": "fast_stable_cnn",
                "frame_path": str(frame_path),
                "roi_path": str(roi_path),
            }

            self.events.appendleft(event)
            self.last_output_number = label

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

            self.reset_candidate()
            return event, None

        self.review_index += 1

        timestamp, frame_path, roi_path = self.save_pair(
            "data/review",
            f"cnn_fast_v2_review_{self.review_index:06d}",
            label,
            frame,
            roi
        )

        item = {
            "review_index": self.review_index,
            "timestamp": timestamp,
            "suggested_number": label,
            "score": float(avg_score),
            "margin": float(avg_margin),
            "agreement": 1.0,
            "votes": len(preds),
            "total_frames": len(preds),
            "reason": "fast_stable_but_low_confidence",
            "frame_path": str(frame_path),
            "roi_path": str(roi_path),
        }

        self.review_items.appendleft(item)
        self.last_output_number = label

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

        self.reset_candidate()
        return None, item

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        cfg = self.cfg.get("cnn_fast_v2_test", {})
        min_stable_frames = int(cfg.get("min_stable_frames", 2))
        min_live_score = float(cfg.get("min_live_score", 0.70))
        min_live_margin = float(cfg.get("min_live_margin", 0.10))

        last_decision = None

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
            review = None

            try:
                decision = self.model.predict(roi)
                last_decision = decision

                label = int(decision["label"])
                score = float(decision["score"])
                margin = float(decision["margin"])

                usable = (
                    state == "RESULT_VISIBLE"
                    and score >= min_live_score
                    and margin >= min_live_margin
                )

                if usable:
                    # نسخه فعلی: عدد تکراری پشت سر هم را ثبت نمی‌کنیم.
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
                            event, review = self.finalize_candidate(frame, roi)
                else:
                    self.reset_candidate()

            except Exception as e:
                state = f"ERROR: {e}"
                self.reset_candidate()

            status = {
                "ready": True,
                "state": state,
                "mode": "FAST_V2",
                "armed": True,
                "last_output_number": self.last_output_number,
                "candidate_label": self.candidate_label,
                "candidate_streak": self.candidate_streak,
                "metrics": metrics,
                "empty_score": None,
                "decision": decision or last_decision,
                "votes": list(self.candidate_predictions),
                "vote_count": self.candidate_streak,
                "soft_vote_count": self.candidate_streak,
                "events": list(self.events),
                "review_items": list(self.review_items),
                "event": event,
                "review": review,
                "thresholds": {
                    "decision": {
                        "min_votes": min_stable_frames
                    },
                    "cnn_fast_v2_test": self.cfg.get("cnn_fast_v2_test", {})
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

    hub = CNNFastV2Hub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("CNN Fast V2 web server running:")
    print(f"  http://192.168.1.251:{args.port}")
    print("")
    print("Logic:")
    print("  Predict every frame")
    print("  Confirm fast after stable CNN frames")
    print("  Consecutive duplicate labels are ignored in this version")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
