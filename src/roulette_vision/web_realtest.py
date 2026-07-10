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
from .vision import crop_relative_roi, roi_metrics, TemplateMatcher, EmptyMatcher
from .digit_number_classifier import DigitNumberClassifier
from .web_classifier import make_handler


class RealTestHub:
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

        # FSM states:
        # WAIT_EMPTY -> WAIT_RESULT -> CAPTURE -> LOCKED
        self.mode = "WAIT_EMPTY"

        self.empty_streak = 0
        self.visible_streak = 0
        self.capture_frames = 0

        self.capture_votes = []
        self.round_index = 0
        self.review_index = 0

        # Initial safety rule:
        # Do not output the same detected/suggested number twice in a row.
        # This intentionally ignores true consecutive duplicates for now.
        self.last_output_number = None
        self.suppressed_items = deque(maxlen=80)

        self.events = deque(maxlen=80)
        self.review_items = deque(maxlen=80)

        self.running = False
        self.thread = None

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)
        Path("data/review").mkdir(parents=True, exist_ok=True)

        self.event_log = Path("data/logs/realtest_events.csv")
        self.review_log = Path("data/logs/realtest_review.csv")
        self.suppressed_log = Path("data/logs/realtest_suppressed_duplicates.csv")

        if not self.event_log.exists():
            with self.event_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "round_index", "timestamp", "number", "score", "margin",
                    "agreement", "usable_votes", "total_votes", "reason",
                    "roi_path", "frame_path"
                ])

        if not self.review_log.exists():
            with self.review_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "review_index", "timestamp", "suggested_number", "score", "margin",
                    "agreement", "total_votes", "reason",
                    "roi_path", "frame_path"
                ])

        if not self.suppressed_log.exists():
            with self.suppressed_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "output_type", "number", "score", "margin",
                    "reason", "roi_path", "frame_path"
                ])

    def start(self):
        print("Template labels:", self.template_matcher.labels())
        print("Digit counts:", self.digit_classifier.counts())
        print("Digit classifier ready:", self.digit_classifier.is_ready())

        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def is_empty(self, roi, metrics):
        state_cfg = self.cfg["state"]
        empty_score = self.empty_matcher.score(roi)

        if empty_score is not None and empty_score >= float(state_cfg["empty_template_score_min"]):
            return True, empty_score

        heuristic_empty = (
            metrics["white_ratio"] <= float(state_cfg["empty_white_ratio_max"])
            and metrics["std"] <= float(state_cfg["empty_std_max"])
        )

        return heuristic_empty, empty_score

    def is_result_visible(self, metrics):
        state_cfg = self.cfg["state"]
        return (
            metrics["white_ratio"] >= float(state_cfg["bright_white_ratio_min"])
            and metrics["std"] >= float(state_cfg["bright_std_min"])
        )

    def is_green_result_visible(self, roi, metrics):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        green = (
            (h >= 35) & (h <= 95) &
            (s >= 45) &
            (v >= 45)
        )

        white_digit = (
            (v >= 145) &
            (s <= 125)
        )

        green_ratio = float(green.mean())
        white_digit_ratio = float(white_digit.mean())

        return (
            green_ratio >= 0.08
            and white_digit_ratio >= 0.006
            and metrics["std"] >= 20.0
        )

    def classify_state(self, roi):
        metrics = roi_metrics(roi)

        empty, empty_score = self.is_empty(roi, metrics)

        if empty:
            return "EMPTY", metrics, empty_score

        if self.is_result_visible(metrics) or self.is_green_result_visible(roi, metrics):
            return "RESULT_VISIBLE", metrics, empty_score

        return "TRANSITION", metrics, empty_score

    def predict_frame(self, roi):
        template = self.template_matcher.predict(roi) if self.template_matcher.is_ready() else None
        digit = self.digit_classifier.predict(roi) if self.digit_classifier.is_ready() else None

        digit_cfg = self.cfg.get("digit", {})

        digit_score_min = float(digit_cfg.get("score_min", 0.70))
        digit_margin_min = float(digit_cfg.get("margin_min", 0.055))

        template_score_min = float(self.cfg["matching"].get("score_min", 0.76))
        template_margin_min = float(self.cfg["matching"].get("margin_min", 0.08))
        template_only_margin_min = float(digit_cfg.get("template_only_margin_min", 0.16))

        decision = {
            "label": None,
            "score": 0.0,
            "margin": 0.0,
            "usable": False,
            "reason": "no_prediction",
            "digit": digit,
            "template": template,
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

        # Safest case: both agree.
        if digit and template and digit["label"] == template["label"]:
            score = max(float(digit["score"]), float(template["score"]))
            margin = max(float(digit["margin"]), float(template["margin"]))

            usable = score >= 0.72 and margin >= 0.050

            decision.update({
                "label": int(digit["label"]),
                "score": score,
                "margin": margin,
                "usable": bool(usable),
                "reason": "digit_template_agree" if usable else "agree_but_weak",
            })
            return decision

        # Important safety rule:
        # If digit and template disagree, do not auto-confirm.
        if digit and template and digit["label"] != template["label"]:
            chosen = digit if digit["score"] >= 0.62 else template

            decision.update({
                "label": int(chosen["label"]),
                "score": float(chosen["score"]),
                "margin": float(chosen["margin"]),
                "usable": False,
                "reason": "digit_template_disagree",
            })
            return decision

        if digit_ok:
            decision.update({
                "label": int(digit["label"]),
                "score": float(digit["score"]),
                "margin": float(digit["margin"]),
                "usable": True,
                "reason": "digit_strong",
            })
            return decision

        if template_ok and template["margin"] >= template_only_margin_min:
            decision.update({
                "label": int(template["label"]),
                "score": float(template["score"]),
                "margin": float(template["margin"]),
                "usable": True,
                "reason": "template_very_strong_no_digit",
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
                "reason": reason,
            })

        return decision

    def reset_capture(self):
        self.capture_votes = []
        self.capture_frames = 0
        self.visible_streak = 0

    def suppress_duplicate_if_needed(self, output_type, number, score, margin, reason, frame, roi):
        rt_cfg = self.cfg.get("real_test", {})

        if not bool(rt_cfg.get("suppress_consecutive_duplicates", True)):
            return False

        if number is None:
            return False

        number = int(number)

        if self.last_output_number is None:
            return False

        if number != int(self.last_output_number):
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        folder = Path("data/suppressed")
        folder.mkdir(parents=True, exist_ok=True)

        frame_path = folder / f"suppressed_{output_type}_{number}_frame_{timestamp}.png"
        roi_path = folder / f"suppressed_{output_type}_{number}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(roi_path), roi)

        item = {
            "timestamp": timestamp,
            "output_type": output_type,
            "number": number,
            "score": float(score),
            "margin": float(margin),
            "reason": f"suppressed_consecutive_duplicate:{reason}",
            "frame_path": str(frame_path),
            "roi_path": str(roi_path),
        }

        self.suppressed_items.appendleft(item)

        with self.suppressed_log.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                item["timestamp"],
                item["output_type"],
                item["number"],
                round(item["score"], 4),
                round(item["margin"], 4),
                item["reason"],
                item["roi_path"],
                item["frame_path"],
            ])

        self.mode = "LOCKED"
        self.reset_capture()

        return True


    def save_pair(self, folder, prefix, number, frame, roi):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        frame_path = Path(folder) / f"{prefix}_{number}_frame_{timestamp}.png"
        roi_path = Path(folder) / f"{prefix}_{number}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(roi_path), roi)

        return timestamp, frame_path, roi_path

    def finalize_capture(self, frame, roi, reason):
        votes = [v for v in self.capture_votes if v.get("label") is not None]

        if not votes:
            self.mode = "LOCKED"
            self.reset_capture()
            return None, None

        all_labels = [v["label"] for v in votes]
        all_counts = Counter(all_labels)
        suggested, suggested_count = all_counts.most_common(1)[0]
        all_agreement = suggested_count / max(1, len(all_labels))

        rt_cfg = self.cfg.get("real_test", {})

        strict_agree_only = bool(rt_cfg.get("strict_confirm_agree_only", True))
        max_disagree_votes = int(rt_cfg.get("max_disagree_votes_for_confirm", 0))

        min_confirm_votes = int(rt_cfg.get("min_confirm_votes", 5))
        min_confirm_agreement = float(rt_cfg.get("min_confirm_agreement", 0.80))
        min_confirm_score = float(rt_cfg.get("min_confirm_score", 0.78))
        min_confirm_margin = float(rt_cfg.get("min_confirm_margin", 0.08))

        disagree_votes = [
            v for v in votes
            if v.get("reason") == "digit_template_disagree"
        ]

        if strict_agree_only:
            usable_votes = [
                v for v in votes
                if v.get("usable") and v.get("reason") == "digit_template_agree"
            ]
        else:
            usable_votes = [v for v in votes if v.get("usable")]

        usable_labels = [v["label"] for v in usable_votes]
        usable_counts = Counter(usable_labels)

        confirmed = False
        confirm_label = None
        confirm_count = 0
        confirm_agreement = 0.0
        selected = []
        candidate_score = 0.0
        candidate_margin = 0.0

        if usable_counts:
            confirm_label, confirm_count = usable_counts.most_common(1)[0]
            confirm_agreement = confirm_count / max(1, len(usable_votes))
            selected = [v for v in usable_votes if v["label"] == confirm_label]

            candidate_score = sum(float(v["score"]) for v in selected) / len(selected)
            candidate_margin = sum(float(v["margin"]) for v in selected) / len(selected)

            confirmed = (
                confirm_count >= min_confirm_votes
                and confirm_agreement >= min_confirm_agreement
                and candidate_score >= min_confirm_score
                and candidate_margin >= min_confirm_margin
                and len(disagree_votes) <= max_disagree_votes
            )

        if confirmed:
            avg_score = sum(float(v["score"]) for v in selected) / len(selected)
            avg_margin = sum(float(v["margin"]) for v in selected) / len(selected)

            if self.suppress_duplicate_if_needed(
                "confirmed",
                confirm_label,
                avg_score,
                avg_margin,
                selected[-1].get("reason", reason),
                frame,
                roi
            ):
                return None, None

            self.round_index += 1
            timestamp, frame_path, roi_path = self.save_pair(
                "data/evidence",
                f"round_{self.round_index:06d}",
                confirm_label,
                frame,
                roi
            )

            event = {
                "round_index": self.round_index,
                "timestamp": timestamp,
                "number": int(confirm_label),
                "score": float(avg_score),
                "margin": float(avg_margin),
                "agreement": float(confirm_agreement),
                "usable_votes": int(confirm_count),
                "total_votes": int(len(votes)),
                "reason": selected[-1].get("reason", reason),
                "frame_path": str(frame_path),
                "roi_path": str(roi_path),
            }

            self.events.appendleft(event)

            with self.event_log.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    event["round_index"],
                    event["timestamp"],
                    event["number"],
                    round(event["score"], 4),
                    round(event["margin"], 4),
                    round(event["agreement"], 4),
                    event["usable_votes"],
                    event["total_votes"],
                    event["reason"],
                    event["roi_path"],
                    event["frame_path"],
                ])

            self.last_output_number = int(confirm_label)

            self.mode = "LOCKED"
            self.reset_capture()
            return event, None

        # Not confirmed: save exactly one review item for this visible result.
        selected_review = [v for v in votes if v["label"] == suggested]

        avg_score = sum(float(v["score"]) for v in selected_review) / len(selected_review)
        avg_margin = sum(float(v["margin"]) for v in selected_review) / len(selected_review)

        if self.suppress_duplicate_if_needed(
            "review",
            suggested,
            avg_score,
            avg_margin,
            reason,
            frame,
            roi
        ):
            return None, None

        self.review_index += 1
        timestamp, frame_path, roi_path = self.save_pair(
            "data/review",
            f"review_{self.review_index:06d}",
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
            "agreement": float(all_agreement),
            "total_votes": int(len(votes)),
            "reason": reason,
            "frame_path": str(frame_path),
            "roi_path": str(roi_path),
        }

        self.review_items.appendleft(item)

        with self.review_log.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                item["review_index"],
                item["timestamp"],
                item["suggested_number"],
                round(item["score"], 4),
                round(item["margin"], 4),
                round(item["agreement"], 4),
                item["total_votes"],
                item["reason"],
                item["roi_path"],
                item["frame_path"],
            ])

        if suggested is not None:
            self.last_output_number = int(suggested)

        self.mode = "LOCKED"
        self.reset_capture()
        return None, item

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        rt_cfg = self.cfg.get("real_test", {})

        empty_frames_to_arm = int(rt_cfg.get("empty_frames_to_arm", 5))
        visible_frames_to_start = int(rt_cfg.get("visible_frames_to_start", 3))
        capture_min_frames = int(rt_cfg.get("capture_min_frames", 8))
        capture_max_frames = int(rt_cfg.get("capture_max_frames", 18))
        unlock_empty_frames = int(rt_cfg.get("unlock_empty_frames", 5))

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

            if self.mode == "WAIT_EMPTY":
                if state == "EMPTY":
                    self.empty_streak += 1

                    if self.empty_streak >= empty_frames_to_arm:
                        self.mode = "WAIT_RESULT"
                        self.empty_streak = 0
                        self.reset_capture()
                else:
                    self.empty_streak = 0

            elif self.mode == "WAIT_RESULT":
                if state == "RESULT_VISIBLE":
                    self.visible_streak += 1

                    if self.visible_streak >= visible_frames_to_start:
                        self.mode = "CAPTURE"
                        self.capture_frames = 0
                        self.capture_votes = []
                elif state == "EMPTY":
                    self.visible_streak = 0
                else:
                    self.visible_streak = 0

            elif self.mode == "CAPTURE":
                if state == "RESULT_VISIBLE":
                    decision = self.predict_frame(roi)
                    self.capture_frames += 1

                    if decision and decision.get("label") is not None:
                        self.capture_votes.append(decision)

                    if self.capture_frames >= capture_max_frames:
                        event, review = self.finalize_capture(
                            frame,
                            roi,
                            "max_capture_frames"
                        )

                else:
                    if self.capture_frames >= capture_min_frames:
                        event, review = self.finalize_capture(
                            frame,
                            roi,
                            "result_ended"
                        )
                    else:
                        # Too short; probably transition, discard and wait again.
                        self.mode = "WAIT_RESULT"
                        self.reset_capture()

            elif self.mode == "LOCKED":
                if state == "EMPTY":
                    self.empty_streak += 1

                    if self.empty_streak >= unlock_empty_frames:
                        self.mode = "WAIT_RESULT"
                        self.empty_streak = 0
                        self.reset_capture()
                else:
                    self.empty_streak = 0

            armed = self.mode in ("WAIT_RESULT", "CAPTURE")

            status = {
                "ready": True,
                "state": state,
                "mode": self.mode,
                "armed": armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "decision": decision,
                "votes": list(self.capture_votes),
                "vote_count": len([v for v in self.capture_votes if v.get("usable")]),
                "soft_vote_count": len(self.capture_votes),
                "events": list(self.events),
                "review_items": list(self.review_items),
                "suppressed_items": list(self.suppressed_items),
                "last_output_number": self.last_output_number,
                "event": event,
                "review": review,
                "thresholds": {
                    "digit": self.cfg.get("digit", {}),
                    "matching": self.cfg.get("matching", {}),
                    "decision": self.cfg.get("decision", {}),
                    "real_test": self.cfg.get("real_test", {}),
                    "review": self.cfg.get("review", {}),
                },
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

    hub = RealTestHub(cfg, source=args.source)
    hub.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("Real Test web server running:")
    print(f"  http://192.168.1.251:{args.port}")
    print("")
    print("Important:")
    print("  One visible result => exactly one output")
    print("  Output is either CONFIRMED or NEEDS_REVIEW")
    print("  System does not read expected-number files")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
