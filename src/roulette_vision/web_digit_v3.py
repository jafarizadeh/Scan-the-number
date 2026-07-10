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
from .vision import crop_relative_roi, roi_metrics
from .vision import TemplateMatcher, EmptyMatcher
from .web_classifier import make_handler
from .digit_number_classifier import DigitNumberClassifier


class DigitV3Hub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)

        self.template_matcher = TemplateMatcher("data/templates", cfg["matching"])
        self.empty_matcher = EmptyMatcher("data/empty", cfg["matching"])
        self.digit_classifier = DigitNumberClassifier("data/templates")

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.status = {"ready": False, "state": "STARTING"}

        self.armed = False
        self.empty_counter = 0
        self.result_frames = 0
        self.non_result_counter = 0

        self.vote_window = deque(maxlen=int(cfg["decision"].get("vote_window", 12)))
        self.soft_window = deque(maxlen=int(cfg.get("review", {}).get("soft_vote_window", 24)))

        self.pending_frame = None
        self.pending_roi = None
        self.pending_decision = None

        self.round_index = 0
        self.review_index = 0

        self.events = deque(maxlen=80)
        self.review_items = deque(maxlen=80)

        self.running = False
        self.thread = None

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)
        Path("data/review").mkdir(parents=True, exist_ok=True)

        self.event_log = Path("data/logs/digit_v3_events.csv")
        self.review_log = Path("data/logs/digit_v3_review.csv")

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
        print("Digit counts:", self.digit_classifier.counts())
        print("Digit classifier ready:", self.digit_classifier.is_ready())

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

        visible = (
            metrics["white_ratio"] >= float(state_cfg["bright_white_ratio_min"])
            and metrics["std"] >= float(state_cfg["bright_std_min"])
        )

        if visible:
            return "RESULT_VISIBLE", metrics, empty_score

        return "TRANSITION", metrics, empty_score

    def make_decision(self, roi):
        template = self.template_matcher.predict(roi) if self.template_matcher.is_ready() else None
        digit = self.digit_classifier.predict(roi) if self.digit_classifier.is_ready() else None

        digit_cfg = self.cfg.get("digit", {})
        digit_score_min = float(digit_cfg.get("score_min", 0.72))
        digit_margin_min = float(digit_cfg.get("margin_min", 0.060))

        template_score_min = float(self.cfg["matching"].get("score_min", 0.76))
        template_margin_min = float(self.cfg["matching"].get("margin_min", 0.08))
        template_only_margin_min = float(digit_cfg.get("template_only_margin_min", 0.14))

        decision = {
            "label": None,
            "score": 0.0,
            "margin": 0.0,
            "usable": False,
            "reason": "no_prediction",
            "digit": digit,
            "template": template
        }

        digit_ok = (
            digit is not None
            and digit["score"] >= digit_score_min
            and digit["margin"] >= digit_margin_min
        )

        template_ok = (
            template is not None
            and template["score"] >= template_score_min
            and template["margin"] >= template_margin_min
        )

        if digit and template:
            if digit["label"] == template["label"]:
                score = max(float(digit["score"]), float(template["score"]))
                margin = max(float(digit["margin"]), float(template["margin"]))

                usable = (
                    score >= 0.72
                    and margin >= 0.050
                )

                decision.update({
                    "label": int(digit["label"]),
                    "score": float(score),
                    "margin": float(margin),
                    "usable": bool(usable),
                    "reason": "digit_template_agree" if usable else "agree_but_weak"
                })
                return decision

            # Critical: never auto-confirm template if digit disagrees.
            # This prevents cases like 29 -> 20.
            chosen = digit if digit["score"] >= 0.62 else template

            decision.update({
                "label": int(chosen["label"]),
                "score": float(chosen["score"]),
                "margin": float(chosen["margin"]),
                "usable": False,
                "reason": "digit_template_disagree"
            })
            return decision

        if digit_ok:
            decision.update({
                "label": int(digit["label"]),
                "score": float(digit["score"]),
                "margin": float(digit["margin"]),
                "usable": True,
                "reason": "digit_strong"
            })
            return decision

        if template_ok and template["margin"] >= template_only_margin_min:
            decision.update({
                "label": int(template["label"]),
                "score": float(template["score"]),
                "margin": float(template["margin"]),
                "usable": True,
                "reason": "template_very_strong_no_digit"
            })
            return decision

        candidates = []

        if digit:
            candidates.append(("digit_preview", digit))

        if template:
            candidates.append(("template_preview", template))

        if candidates:
            reason, pred = max(candidates, key=lambda x: x[1].get("score", -1.0))
            decision.update({
                "label": int(pred["label"]),
                "score": float(pred["score"]),
                "margin": float(pred["margin"]),
                "usable": False,
                "reason": reason
            })

        return decision

    def reset_round_buffers(self):
        self.vote_window.clear()
        self.soft_window.clear()
        self.result_frames = 0
        self.non_result_counter = 0
        self.pending_frame = None
        self.pending_roi = None
        self.pending_decision = None

    def save_pair(self, folder, prefix, number, frame, roi):
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

        min_votes = int(self.cfg["decision"].get("min_votes", 7))
        min_agreement = float(self.cfg["decision"].get("min_agreement", 0.70))

        total = len(self.vote_window)
        agreement = top_count / max(1, total)

        if top_count < min_votes or agreement < min_agreement:
            return None

        selected = [v for v in self.vote_window if v["label"] == top_label]

        score = sum(float(v["score"]) for v in selected) / len(selected)
        margin = sum(float(v["margin"]) for v in selected) / len(selected)
        reason = selected[-1].get("reason")

        self.round_index += 1

        timestamp, frame_path, roi_path = self.save_pair(
            "data/evidence",
            f"round_{self.round_index:06d}",
            top_label,
            frame,
            roi
        )

        event = {
            "round_index": self.round_index,
            "timestamp": timestamp,
            "number": int(top_label),
            "score": float(score),
            "margin": float(margin),
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
        self.reset_round_buffers()

        return event

    def review_if_needed(self, reason):
        if not self.armed:
            return None

        if self.result_frames <= 0:
            return None

        if self.pending_frame is None or self.pending_roi is None:
            return None

        labels = [v["label"] for v in self.soft_window if v.get("label") is not None]

        suggested = None
        score = 0.0
        margin = 0.0

        if labels:
            counts = Counter(labels)
            suggested, count = counts.most_common(1)[0]

            selected = [v for v in self.soft_window if v.get("label") == suggested]
            score = sum(float(v.get("score", 0.0)) for v in selected) / len(selected)
            margin = sum(float(v.get("margin", 0.0)) for v in selected) / len(selected)

            agreement = count / max(1, len(labels))
            reason = f"{reason}_suggested_{count}_{agreement:.2f}"

        elif self.pending_decision:
            suggested = self.pending_decision.get("label")
            score = float(self.pending_decision.get("score", 0.0))
            margin = float(self.pending_decision.get("margin", 0.0))
            reason = self.pending_decision.get("reason", reason)

        self.review_index += 1

        timestamp, frame_path, roi_path = self.save_pair(
            "data/review",
            f"review_{self.review_index:06d}",
            suggested if suggested is not None else "unknown",
            self.pending_frame,
            self.pending_roi
        )

        item = {
            "review_index": self.review_index,
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
                item["review_index"],
                item["timestamp"],
                item["suggested_number"],
                round(item["score"], 4),
                round(item["margin"], 4),
                item["reason"],
                item["roi_path"],
                item["frame_path"]
            ])

        self.armed = False
        self.reset_round_buffers()

        return item

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        review_cfg = self.cfg.get("review", {})
        max_frames = int(review_cfg.get("max_result_frames_without_confirm", 18))
        transition_finalize_frames = int(review_cfg.get("transition_finalize_frames", 2))

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
                if self.armed and self.result_frames > 0:
                    review = self.review_if_needed("ended_on_empty_without_confirm")

                self.empty_counter += 1

                if self.empty_counter >= int(self.cfg["state"].get("empty_confirm_frames", 2)):
                    self.armed = True
                    self.reset_round_buffers()

            elif state == "RESULT_VISIBLE":
                self.empty_counter = 0
                self.non_result_counter = 0

                decision = self.make_decision(roi)

                if self.armed:
                    self.result_frames += 1

                    if decision and decision.get("label") is not None:
                        self.soft_window.append(decision)
                        self.pending_frame = frame.copy()
                        self.pending_roi = roi.copy()
                        self.pending_decision = decision

                    if decision and decision.get("usable"):
                        self.vote_window.append(decision)

                    event = self.confirm_if_ready(frame, roi)

                    if event is None and self.result_frames >= max_frames:
                        review = self.review_if_needed("timeout_without_confirm")

            else:
                self.empty_counter = 0

                if self.armed and self.result_frames > 0:
                    self.non_result_counter += 1

                    if self.non_result_counter >= transition_finalize_frames:
                        review = self.review_if_needed("transition_after_result_without_confirm")

            status = {
                "ready": True,
                "state": state,
                "armed": self.armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "decision": decision,
                "votes": list(self.vote_window),
                "soft_votes": list(self.soft_window),
                "vote_count": len(self.vote_window),
                "soft_vote_count": len(self.soft_window),
                "events": list(self.events),
                "review_items": list(self.review_items),
                "event": event,
                "review": review,
                "thresholds": {
                    "digit": self.cfg.get("digit", {}),
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

    hub = DigitV3Hub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("Digit V3 web server running:")
    print(f"  http://192.168.1.251:{args.port}")
    print("")
    print("Rules:")
    print("  digit/template disagree => no auto-confirm")
    print("  weak visible result => Needs Review")
    print("  no visible result should be silently lost")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
