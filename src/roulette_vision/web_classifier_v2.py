import argparse
import csv
import json
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2

from .config import load_config, ensure_project_dirs
from .web_classifier import ClassifierHub, make_handler


class ClassifierHubV2(ClassifierHub):
    def __init__(self, cfg, source=None):
        super().__init__(cfg, source=source)

        review_cfg = cfg.get("review", {})

        self.soft_window = deque(
            maxlen=int(review_cfg.get("soft_vote_window", 24))
        )

        self.non_result_counter = 0
        self.pending_frame = None
        self.pending_roi = None
        self.pending_decision = None

    def reset_round_buffers(self):
        self.vote_window.clear()
        self.soft_window.clear()
        self.result_frames = 0
        self.non_result_counter = 0
        self.pending_frame = None
        self.pending_roi = None
        self.pending_decision = None

    def soft_review_if_needed(self, reason="soft_review"):
        if not self.armed:
            return None

        if self.result_frames <= 0:
            return None

        if self.pending_frame is None or self.pending_roi is None:
            return None

        review_cfg = self.cfg.get("review", {})

        min_soft_votes = int(review_cfg.get("min_soft_votes", 5))
        min_soft_agreement = float(review_cfg.get("min_soft_agreement", 0.55))

        suggested = None
        score = 0.0
        margin = 0.0
        final_reason = reason

        if self.soft_window:
            labels = [v["label"] for v in self.soft_window if v.get("label") is not None]
            counts = Counter(labels)

            if counts:
                top_label, top_count = counts.most_common(1)[0]
                agreement = top_count / max(1, len(labels))

                selected = [v for v in self.soft_window if v.get("label") == top_label]

                avg_score = sum(float(v.get("score", 0.0)) for v in selected) / len(selected)
                avg_margin = sum(float(v.get("margin", 0.0)) for v in selected) / len(selected)

                suggested = int(top_label)
                score = float(avg_score)
                margin = float(avg_margin)

                if top_count >= min_soft_votes and agreement >= min_soft_agreement:
                    final_reason = f"soft_stable_review_{top_count}_{agreement:.2f}"
                else:
                    final_reason = f"low_agreement_review_{top_count}_{agreement:.2f}"

        if suggested is None and self.pending_decision:
            suggested = self.pending_decision.get("label")
            score = float(self.pending_decision.get("score", 0.0))
            margin = float(self.pending_decision.get("margin", 0.0))
            final_reason = self.pending_decision.get("reason", reason)

        review_index = len(self.review_items) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        frame_path = Path("data/review") / f"review_{review_index:06d}_{suggested}_frame_{timestamp}.png"
        roi_path = Path("data/review") / f"review_{review_index:06d}_{suggested}_roi_{timestamp}.png"

        cv2.imwrite(str(frame_path), self.pending_frame)
        cv2.imwrite(str(roi_path), self.pending_roi)

        item = {
            "review_index": review_index,
            "timestamp": timestamp,
            "suggested_number": suggested,
            "score": score,
            "margin": margin,
            "reason": final_reason,
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
                round(score, 4),
                round(margin, 4),
                final_reason,
                str(roi_path),
                str(frame_path)
            ])

        self.armed = False
        self.reset_round_buffers()

        return item

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)

        review_cfg = self.cfg.get("review", {})
        max_result_frames = int(review_cfg.get("max_result_frames_without_confirm", 24))
        transition_finalize_frames = int(review_cfg.get("transition_finalize_frames", 2))

        while self.running:
            frame = self.camera.read()

            if frame is None:
                time.sleep(0.03)
                continue

            roi = self.__class__.__mro__[1].__dict__["classify_state"]
            state, metrics, empty_score = super().classify_state(
                cv2.resize(frame, (frame.shape[1], frame.shape[0]))[
                    self.cfg["roi"].get("dummy", 0): self.cfg["roi"].get("dummy", 0) + frame.shape[0],
                    :
                ] if False else __import__("roulette_vision.vision", fromlist=["crop_relative_roi"]).crop_relative_roi(frame, self.cfg["roi"])
            )

            roi_img = __import__("roulette_vision.vision", fromlist=["crop_relative_roi"]).crop_relative_roi(frame, self.cfg["roi"])

            decision = None
            event = None
            review = None

            if state == "EMPTY":
                if self.armed and self.result_frames > 0:
                    review = self.soft_review_if_needed("ended_on_empty_without_confirm")

                self.empty_counter += 1

                if self.empty_counter >= int(self.cfg["state"].get("empty_confirm_frames", 2)):
                    self.armed = True
                    self.reset_round_buffers()

            elif state == "RESULT_VISIBLE":
                self.empty_counter = 0
                self.non_result_counter = 0

                decision = self.make_decision(roi_img)

                if self.armed:
                    self.result_frames += 1

                    if decision and decision.get("label") is not None:
                        self.soft_window.append(decision)
                        self.pending_frame = frame.copy()
                        self.pending_roi = roi_img.copy()
                        self.pending_decision = decision

                    if decision and decision.get("usable"):
                        self.vote_window.append(decision)

                    event = self.confirm_if_ready(frame, roi_img)

                    if event is not None:
                        self.reset_round_buffers()

                    elif self.result_frames >= max_result_frames:
                        review = self.soft_review_if_needed("timeout_without_confirm")

            else:
                self.empty_counter = 0

                if self.armed and self.result_frames > 0:
                    self.non_result_counter += 1

                    if self.non_result_counter >= transition_finalize_frames:
                        review = self.soft_review_if_needed("transition_after_result_without_confirm")

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
                    "classifier": self.cfg.get("classifier", {}),
                    "matching": self.cfg.get("matching", {}),
                    "decision": self.cfg.get("decision", {}),
                    "review": self.cfg.get("review", {})
                }
            }

            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_roi = roi_img.copy()
                self.status = status

            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    hub = ClassifierHubV2(cfg, source=args.source)
    hub.start()

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer((args.host, args.port), make_handler(hub, cfg))

    print("Classifier V2 real-test web server running:")
    print(f"  http://192.168.1.251:{args.port}")
    print("")
    print("V2 behavior:")
    print("  CONFIRMED if strong enough")
    print("  NEEDS_REVIEW if visible result was not confirmed")
    print("  Nothing should be silently lost")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close()
        hub.close()


if __name__ == "__main__":
    main()
