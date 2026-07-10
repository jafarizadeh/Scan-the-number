from collections import Counter, deque
import csv
from datetime import datetime
from pathlib import Path

import cv2

from .vision import roi_metrics, TemplateMatcher, EmptyMatcher


class RouletteDetectionEngine:
    def __init__(self, cfg):
        self.cfg = cfg

        self.matcher = TemplateMatcher("data/templates", cfg["matching"])
        self.empty_matcher = EmptyMatcher("data/empty", cfg["matching"])

        self.armed = False
        self.empty_counter = 0
        self.votes = deque(maxlen=int(cfg["decision"]["vote_window"]))
        self.round_index = 0

        self.log_path = Path("data/logs/events.csv")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round_index",
                    "timestamp",
                    "number",
                    "confidence",
                    "agreement",
                    "score",
                    "margin",
                    "evidence_frame",
                    "evidence_roi"
                ])

    def classify_roi_state(self, roi_bgr):
        metrics = roi_metrics(roi_bgr)
        state_cfg = self.cfg["state"]

        empty_template_score = self.empty_matcher.score(roi_bgr)

        if empty_template_score is not None:
            if empty_template_score >= float(state_cfg["empty_template_score_min"]):
                return "EMPTY", metrics, empty_template_score

        is_heuristic_empty = (
            metrics["white_ratio"] <= float(state_cfg["empty_white_ratio_max"])
            and metrics["std"] <= float(state_cfg["empty_std_max"])
        )

        is_bright = (
            metrics["white_ratio"] >= float(state_cfg["bright_white_ratio_min"])
            and metrics["std"] >= float(state_cfg["bright_std_min"])
        )

        if is_heuristic_empty:
            return "EMPTY", metrics, empty_template_score

        if is_bright:
            return "BRIGHT", metrics, empty_template_score

        return "DIMMED", metrics, empty_template_score

    def process(self, frame_bgr, roi_bgr):
        state, metrics, empty_score = self.classify_roi_state(roi_bgr)

        if state == "EMPTY":
            self.empty_counter += 1
            self.votes.clear()

            if self.empty_counter >= int(self.cfg["state"]["empty_confirm_frames"]):
                self.armed = True

            return {
                "state": state,
                "armed": self.armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "event": None
            }

        self.empty_counter = 0

        if state != "BRIGHT":
            return {
                "state": state,
                "armed": self.armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "event": None
            }

        if not self.armed:
            return {
                "state": state,
                "armed": self.armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "event": None
            }

        prediction = self.matcher.predict(roi_bgr)

        if prediction is None:
            return {
                "state": state,
                "armed": self.armed,
                "metrics": metrics,
                "empty_score": empty_score,
                "event": None,
                "prediction": None
            }

        score_min = float(self.cfg["matching"]["score_min"])
        margin_min = float(self.cfg["matching"]["margin_min"])

        # Safe Mode:
        # Do NOT allow top-k alone to confirm a result. A result must have
        # both enough absolute score and enough separation from the second label.
        usable_by_margin = (
            prediction["score"] >= score_min
            and prediction["margin"] >= margin_min
        )

        usable_prediction = usable_by_margin

        prediction["usable"] = usable_prediction
        prediction["usable_by_margin"] = usable_by_margin
        prediction["usable_by_topk"] = False

        if usable_prediction:
            self.votes.append(prediction)

        event = self.try_confirm(frame_bgr, roi_bgr)

        return {
            "state": state,
            "armed": self.armed,
            "metrics": metrics,
            "empty_score": empty_score,
            "prediction": prediction,
            "event": event
        }

    def try_confirm(self, frame_bgr, roi_bgr):
        if not self.votes:
            return None

        labels = [v["label"] for v in self.votes]
        counts = Counter(labels)

        top_label, top_count = counts.most_common(1)[0]
        total_votes = len(self.votes)

        min_votes = int(self.cfg["decision"]["min_votes"])
        min_agreement = float(self.cfg["decision"]["min_agreement"])

        agreement = top_count / max(1, total_votes)

        if top_count < min_votes:
            return None

        if agreement < min_agreement:
            return None

        selected = [v for v in self.votes if v["label"] == top_label]

        avg_score = sum(v["score"] for v in selected) / len(selected)
        avg_margin = sum(v["margin"] for v in selected) / len(selected)

        self.round_index += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        evidence_dir = Path("data/evidence")
        evidence_dir.mkdir(parents=True, exist_ok=True)

        frame_path = evidence_dir / f"round_{self.round_index:06d}_{top_label}_frame_{timestamp}.png"
        roi_path = evidence_dir / f"round_{self.round_index:06d}_{top_label}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), frame_bgr)
        cv2.imwrite(str(roi_path), roi_bgr)

        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.round_index,
                timestamp,
                top_label,
                round(agreement, 4),
                round(agreement, 4),
                round(avg_score, 4),
                round(avg_margin, 4),
                str(frame_path),
                str(roi_path)
            ])

        self.armed = False
        self.votes.clear()

        return {
            "round_index": self.round_index,
            "timestamp": timestamp,
            "number": top_label,
            "agreement": agreement,
            "score": avg_score,
            "margin": avg_margin,
            "frame_path": str(frame_path),
            "roi_path": str(roi_path)
        }
