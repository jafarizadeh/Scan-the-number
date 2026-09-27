import argparse
import csv
import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_rect_from_shape, roi_metrics
from .cnn_number_model import CNNNumberModel
from .hardware_actions import HardwareActionController, DEFAULT_HARDWARE_CONFIG, clean_hardware_config

APP_SETTINGS_PATH = Path("data/app_settings.json")

DEFAULT_TARGET_LISTS = {
    "selectedListId": "list-1",
    "lists": [
        {"id": "list-1", "name": "List 1", "numbers": [0, 8, 10, 11, 13, 20, 29, 35, 36]},
    ],
    "doubleMatchEnabled": False,
    "spinAutoEnabled": True,
    "hardware": dict(DEFAULT_HARDWARE_CONFIG),
}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def clean_numbers(values):
    out = []
    seen = set()
    for value in values or []:
        try:
            n = int(value)
        except Exception:
            continue
        if 0 <= n <= 36 and n not in seen:
            out.append(n)
            seen.add(n)
    return sorted(out)


def clean_lists(payload):
    raw_lists = payload.get("lists", []) if isinstance(payload, dict) else []
    lists = []
    for idx, item in enumerate(raw_lists):
        if not isinstance(item, dict):
            continue
        lid = str(item.get("id") or f"list-{uuid.uuid4().hex[:8]}")
        name = str(item.get("name") or f"List {idx + 1}").strip()[:60]
        nums = clean_numbers(item.get("numbers", []))
        lists.append({"id": lid, "name": name, "numbers": nums})
    if not lists:
        lists = list(DEFAULT_TARGET_LISTS["lists"])
    selected = str(payload.get("selectedListId") or lists[0]["id"]) if isinstance(payload, dict) else lists[0]["id"]
    if selected not in {x["id"] for x in lists}:
        selected = lists[0]["id"]
    return {"selectedListId": selected, "lists": lists}


class AppStore:
    def __init__(self, path=APP_SETTINGS_PATH):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data = self.load()

    def load(self):
        data = dict(DEFAULT_TARGET_LISTS)
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text())
                data.update(raw)
        except Exception as exc:
            print(f"[APP] settings load failed: {exc}")
        clean = clean_lists(data)
        clean["doubleMatchEnabled"] = bool(data.get("doubleMatchEnabled", False))
        clean["spinAutoEnabled"] = bool(data.get("spinAutoEnabled", True))
        clean["hardware"] = clean_hardware_config(data.get("hardware", {}))
        return clean

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get(self):
        with self.lock:
            return json.loads(json.dumps(self.data))

    def update_lists(self, payload):
        clean = clean_lists(payload)
        with self.lock:
            self.data["lists"] = clean["lists"]
            self.data["selectedListId"] = clean["selectedListId"]
            self.save()
            return json.loads(json.dumps(self.data))

    def update_double_match(self, enabled):
        with self.lock:
            self.data["doubleMatchEnabled"] = bool(enabled)
            self.save()
            return bool(self.data["doubleMatchEnabled"])

    def update_spin_auto(self, enabled):
        with self.lock:
            self.data["spinAutoEnabled"] = bool(enabled)
            self.save()
            return bool(self.data["spinAutoEnabled"])

    def update_hardware(self, payload):
        clean = clean_hardware_config(payload)
        with self.lock:
            self.data["hardware"] = clean
            self.save()
            return dict(clean)

    def selected_list(self):
        data = self.get()
        selected = data.get("selectedListId")
        for item in data.get("lists", []):
            if item.get("id") == selected:
                return item
        return data["lists"][0]


class FinalVisionHub:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.camera = FrameSource(cfg["camera"], source=source)
        model_cfg = cfg.get("cnn_model", {})
        self.model = CNNNumberModel(
            model_cfg.get("model_path", "data/models/roulette_mobilenet_best_single.onnx"),
            img_size=int(model_cfg.get("img_size", 160)),
        )
        self.store = AppStore()
        self.actions = HardwareActionController(self.store.get().get("hardware", {}))

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_roi = None
        self.status = {"ready": False, "state": "STARTING"}
        self.running = False
        self.monitoring = False
        self.camera_enabled = True
        self.thread = None

        self.last_output_number = None
        self.previous_pair_number = None

        # Scanner result lock.
        # After recording a result, keep the scanner locked until a
        # DIFFERENT number passes the full consecutive-frame validation.
        self.scan_locked = False
        self.unlock_gap_streak = 0
        self.unlock_change_label = None
        self.unlock_change_streak = 0

        self.candidate_label = None
        self.candidate_predictions = []
        self.candidate_streak = 0
        self.latest_decision = None
        self.latest_match = None
        self.session_id = None
        self.selected_list = self.store.selected_list()

        self.history = deque(maxlen=2000)
        self.events = deque(maxlen=100)

        Path("data/logs").mkdir(parents=True, exist_ok=True)
        Path("data/evidence").mkdir(parents=True, exist_ok=True)
        self.event_log = Path("data/logs/final_events.csv")
        if not self.event_log.exists():
            with self.event_log.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["id", "timestamp", "number", "score", "margin", "list_id", "list_name", "is_match", "action_accepted", "roi_path"])

    def start(self):
        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.camera.close()
        self.actions.close()


    def set_camera_enabled(self, enabled):
        with self.lock:
            self.camera_enabled = bool(enabled)
            if not self.camera_enabled:
                self.monitoring = False
                try:
                    self.reset_candidate()
                except Exception:
                    pass
        return self.public_status()

    def start_monitoring(self, selected_list_id=None):
        data = self.store.get()
        if selected_list_id:
            data["selectedListId"] = selected_list_id
            self.store.update_lists(data)
        self.selected_list = self.store.selected_list()
        with self.lock:
            self.monitoring = True
            self.session_id = f"session-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.last_output_number = None
            self.previous_pair_number = None

            self.scan_locked = False
            self.unlock_gap_streak = 0
            self.unlock_change_label = None
            self.unlock_change_streak = 0

            self.reset_candidate()
            self.latest_match = None
        return self.public_status()

    def stop_monitoring(self):
        with self.lock:
            self.monitoring = False
            self.reset_candidate()
        return self.public_status()

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
        if white_digit_ratio >= 0.003 or green_ratio >= 0.03 or red_ratio >= 0.03 or metrics["std"] >= 18.0:
            return "RESULT_VISIBLE"
        return "LOW_SIGNAL"

    def save_roi(self, number):
        with self.lock:
            roi = None if self.latest_roi is None else self.latest_roi.copy()
        if roi is None:
            return ""
        folder = Path("data/evidence")
        folder.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = folder / f"final_{number}_{ts}.png"
        cv2.imwrite(str(path), roi)
        return str(path)

    def record_number(self, number, score, margin):
        selected = self.selected_list or self.store.selected_list()
        target_set = set(int(x) for x in selected.get("numbers", []))
        is_match = int(number) in target_set
        roi_path = self.save_roi(number)
        event_id = f"ev-{uuid.uuid4().hex[:10]}"
        app_settings = self.store.get()
        double_match_enabled = bool(
            app_settings.get("doubleMatchEnabled", False)
        )
        spin_auto_enabled = bool(
            app_settings.get("spinAutoEnabled", True)
        )

        previous_number = self.previous_pair_number
        previous_is_match = (
            previous_number is not None and
            int(previous_number) in target_set
        )

        def trigger_components(*, sound, tapper, reason):
            base_hardware = app_settings.get("hardware", {})

            buzzer_enabled = bool(
                base_hardware.get("buzzerEnabled", False)
                and sound
            )

            actuator_enabled = bool(
                base_hardware.get("actuatorEnabled", False)
                and tapper
            )

            try:
                tapper_delay_ms = int(
                    base_hardware.get(
                        "actuatorDelayMs",
                        0,
                    )
                )
            except Exception:
                tapper_delay_ms = 0

            tapper_delay_ms = max(
                0,
                min(10000, tapper_delay_ms),
            )

            if not buzzer_enabled and not actuator_enabled:
                return {
                    "accepted": False,
                    "reason": "requested_actions_disabled",
                }

            # ------------------------------------------------
            # No Tapper:
            # Sound executes immediately.
            # ------------------------------------------------
            if not actuator_enabled:
                return self.actions.trigger_async(
                    hardware={
                        "buzzerEnabled": buzzer_enabled,
                        "actuatorEnabled": False,
                        "actuatorDelayMs": 0,
                    },
                    number=number,
                    list_name=selected.get("name", ""),
                    reason=reason,
                )

            # ------------------------------------------------
            # Tapper with no delay:
            # preserve normal immediate behavior.
            # ------------------------------------------------
            if tapper_delay_ms <= 0:
                return self.actions.trigger_async(
                    hardware={
                        "buzzerEnabled": buzzer_enabled,
                        "actuatorEnabled": True,
                        "actuatorDelayMs": 0,
                    },
                    number=number,
                    list_name=selected.get("name", ""),
                    reason=reason,
                )

            # ------------------------------------------------
            # Delayed Tapper:
            #
            # Sound MUST NOT wait for Tapper.
            # ------------------------------------------------
            if buzzer_enabled:
                sound_result = self.actions.trigger_async(
                    hardware={
                        "buzzerEnabled": True,
                        "actuatorEnabled": False,
                        "actuatorDelayMs": 0,
                    },
                    number=number,
                    list_name=selected.get("name", ""),
                    reason=reason + "_sound",
                )
            else:
                sound_result = {
                    "accepted": False,
                    "reason": "sound_not_requested",
                }

            def delayed_tapper():
                time.sleep(
                    tapper_delay_ms / 1000.0
                )

                # Check the CURRENT Tapper switch.
                # If the user switched it OFF during the delay,
                # the pending Tapper action is cancelled.
                current_hardware = (
                    self.store
                    .get()
                    .get("hardware", {})
                )

                if not bool(
                    current_hardware.get(
                        "actuatorEnabled",
                        False,
                    )
                ):
                    return

                tap_hardware = {
                    "buzzerEnabled": False,
                    "actuatorEnabled": True,

                    # Delay was already performed above.
                    "actuatorDelayMs": 0,

                    # Scheduled Tapper should not be rejected
                    # because of the earlier Sound cooldown.
                    "actionCooldownMs": 0,
                }

                # If hardware is briefly busy exactly at the
                # scheduled moment, retry for a short period.
                for _ in range(20):
                    result = self.actions.trigger_async(
                        hardware=tap_hardware,
                        number=number,
                        list_name=selected.get(
                            "name",
                            "",
                        ),
                        reason=(
                            reason
                            + "_delayed_tapper"
                        ),
                    )

                    if result.get("accepted"):
                        return

                    if result.get("reason") not in (
                        "busy",
                        "cooldown",
                    ):
                        return

                    time.sleep(0.1)

            threading.Thread(
                target=delayed_tapper,
                daemon=True,
                name="roulette-delayed-tapper",
            ).start()

            return {
                "accepted": True,
                "reason": "tapper_scheduled",
                "delayMs": tapper_delay_ms,
                "sound": sound_result,
            }

        tapper_auto_disabled = False
        action_result = {"accepted": False, "reason": "not_match"}

        # ----------------------------------------------------
        # DOUBLE MATCH GATE
        #
        # When enabled, NO hardware action may execute until
        # the current number AND the immediately previous
        # scanned number are both members of the active list.
        #
        # Once this condition is satisfied, the existing
        # Spin Auto behavior runs unchanged.
        # ----------------------------------------------------
        double_match_blocked = (
            double_match_enabled
            and not (
                is_match
                and previous_is_match
            )
        )

        if double_match_blocked:
            if is_match:
                action_result = {
                    "accepted": False,
                    "reason": "waiting_for_second_match",
                }
            else:
                action_result = {
                    "accepted": False,
                    "reason": "double_match_sequence_broken",
                }

        else:
            tapper_auto_disabled = False

            # Spin Auto OFF = sound on match, tapper on non-match.
            #
            # MATCH:
            #   Sound only.
            #   Tapper does NOT run for this match.
            #   The persistent Tapper toggle remains unchanged.
            #
            # NON-MATCH:
            #   Tapper runs only if the user has manually enabled it.
            if not spin_auto_enabled:
                if is_match:
                    action_result = trigger_components(
                        sound=True,
                        tapper=False,
                        reason="number_in_selected_list_spin_auto_off",
                    )

                else:
                    action_result = trigger_components(
                        sound=False,
                        tapper=True,
                        reason="number_outside_selected_list_spin_auto_off",
                    )

            # Spin Auto ON has priority for action execution.
            #
            # MATCH:
            #   Sound + Tapper are requested immediately.
            #   Double Match must not suppress the action.
            #
            # NON-MATCH:
            #   No action.
            elif is_match:
                action_result = trigger_components(
                    sound=True,
                    tapper=True,
                    reason="number_in_selected_list_spin_auto_on",
                )
        event = {
            "id": event_id,
            "timestamp": now_iso(),
            "number": int(number),
            "score": float(score),
            "margin": float(margin),
            "listId": selected.get("id"),
            "listName": selected.get("name"),
            "targetNumbers": list(selected.get("numbers", [])),
            "isMatch": bool(is_match),
            "spinAutoEnabled": spin_auto_enabled,
            "tapperAutoDisabled": bool(tapper_auto_disabled),
            "action": action_result,
            "roiPath": roi_path,
        }
        with self.lock:
            self.history.appendleft(event)
            self.events.appendleft(event)
            self.previous_pair_number = int(number)
            self.last_output_number = int(number)

            # Prevent the same visible roulette result from being recorded
            # repeatedly on consecutive camera frames.
            self.scan_locked = True
            self.unlock_gap_streak = 0
            self.unlock_change_label = None
            self.unlock_change_streak = 0

            if is_match:
                self.latest_match = event
        with self.event_log.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([event_id, event["timestamp"], number, round(score, 4), round(margin, 4), selected.get("id"), selected.get("name"), is_match, action_result.get("accepted"), roi_path])
        return event

    def finalize_candidate(self):
        preds = list(self.candidate_predictions)

        if not preds or self.candidate_label is None:
            self.reset_candidate()
            return None

        label = int(self.candidate_label)

        avg_score = (
            sum(float(p["score"]) for p in preds) / len(preds)
        )
        avg_margin = (
            sum(float(p["margin"]) for p in preds) / len(preds)
        )

        # Every prediction stored in candidate_predictions has already
        # passed the per-frame score/margin gate. Reaching this function
        # therefore means that the same label was observed for the
        # required number of consecutive valid frames.
        event = self.record_number(
            label,
            avg_score,
            avg_margin,
        )

        self.reset_candidate()
        return event

    def _loop(self):
        fps = int(self.cfg["camera"].get("fps", 15))
        delay = 1.0 / max(1, fps)
        cfg = self.cfg.get("final_app", {})
        required_consecutive_frames = int(
            cfg.get(
                "required_consecutive_frames",
                cfg.get("min_stable_frames", 5),
            )
        )

        min_frame_score = float(
            cfg.get(
                "min_frame_score",
                cfg.get("min_live_score", 0.30),
            )
        )

        min_frame_margin = float(
            cfg.get(
                "min_frame_margin",
                cfg.get("min_live_margin", 0.05),
            )
        )

        # --------------------------------------------------
        # Roulette background color hard-filter.
        #
        # GREEN -> only 0
        # RED   -> only standard red roulette numbers
        # BLACK -> only standard black roulette numbers
        #
        # Color limits the allowed CNN labels. The number
        # itself is still selected from CNN Top-K results.
        # --------------------------------------------------

        color_filter_enabled = bool(
            cfg.get("roulette_color_filter_enabled", True)
        )

        green_h_min = int(cfg.get("green_h_min", 35))
        green_h_max = int(cfg.get("green_h_max", 95))
        green_s_min = int(cfg.get("green_s_min", 55))
        green_v_min = int(cfg.get("green_v_min", 35))
        green_ratio_min = float(
            cfg.get("green_ratio_min", 0.20)
        )

        red_h1_min = int(cfg.get("red_h1_min", 0))
        red_h1_max = int(cfg.get("red_h1_max", 12))
        red_h2_min = int(cfg.get("red_h2_min", 165))
        red_h2_max = int(cfg.get("red_h2_max", 179))
        red_s_min = int(cfg.get("red_s_min", 60))
        red_v_min = int(cfg.get("red_v_min", 35))
        red_ratio_min = float(
            cfg.get("red_ratio_min", 0.20)
        )

        black_v_max = int(
            cfg.get("black_v_max", 85)
        )
        black_ratio_min = float(
            cfg.get("black_ratio_min", 0.20)
        )

        GREEN_NUMBERS = {0}

        RED_NUMBERS = {
            1, 3, 5, 7, 9,
            12, 14, 16, 18, 19,
            21, 23, 25, 27,
            30, 32, 34, 36,
        }

        BLACK_NUMBERS = {
            2, 4, 6, 8, 10, 11,
            13, 15, 17, 20, 22,
            24, 26, 28, 29, 31,
            33, 35,
        }

        # Re-arm protection after a confirmed roulette result.
        #
        # At 15 FPS, 3 low-signal frames are roughly 0.2 seconds.
        # A different label must also remain stable for several frames
        # before it is allowed to release the previous-result lock.
        unlock_low_signal_frames = int(
            cfg.get("unlock_low_signal_frames", 3)
        )
        change_stable_frames = int(
            cfg.get("change_stable_frames", 3)
        )
        change_min_score = float(
            cfg.get("change_min_score", min_frame_score)
        )
        change_min_margin = float(
            cfg.get("change_min_margin", min_frame_margin)
        )

        # A LOW_SIGNAL frame may still contain the dimmed previous result.
        # Do not interpret that as a real inter-result gap while the CNN
        # still recognizes the previous number with strong confidence.
        same_number_hold_score = float(
            cfg.get("same_number_hold_score", 0.80)
        )
        same_number_hold_margin = float(
            cfg.get("same_number_hold_margin", 0.50)
        )
        same_number_hold_std_min = float(
            cfg.get("same_number_hold_std_min", 10.0)
        )

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
            if self.monitoring:
                try:
                    # ==================================================
                    # Five-frame parallel fusion
                    #
                    # Number recognition and background-color evidence
                    # are collected from the SAME five frames.
                    #
                    # Color NEVER converts one CNN number into another.
                    # It only validates the final CNN winner.
                    #
                    # GREEN is special because roulette zero is the only
                    # green number.
                    # ==================================================

                    decision = dict(
                        self.model.predict(roi)
                    )

                    raw_label = int(
                        decision["label"]
                    )
                    raw_score = float(
                        decision["score"]
                    )
                    raw_margin = float(
                        decision["margin"]
                    )

                    # --------------------------------------------------
                    # Color evidence
                    #
                    # Keep CNN input untouched. Calibration showed that
                    # physical red is correctly interpreted when this ROI
                    # is treated as RGB for HSV conversion.
                    # --------------------------------------------------

                    hsv = cv2.cvtColor(
                        roi,
                        cv2.COLOR_RGB2HSV,
                    )

                    green_mask = cv2.inRange(
                        hsv,
                        (
                            green_h_min,
                            green_s_min,
                            green_v_min,
                        ),
                        (
                            green_h_max,
                            255,
                            255,
                        ),
                    )

                    red_mask_1 = cv2.inRange(
                        hsv,
                        (
                            red_h1_min,
                            red_s_min,
                            red_v_min,
                        ),
                        (
                            red_h1_max,
                            255,
                            255,
                        ),
                    )

                    red_mask_2 = cv2.inRange(
                        hsv,
                        (
                            red_h2_min,
                            red_s_min,
                            red_v_min,
                        ),
                        (
                            red_h2_max,
                            255,
                            255,
                        ),
                    )

                    red_mask = cv2.bitwise_or(
                        red_mask_1,
                        red_mask_2,
                    )

                    black_mask = cv2.inRange(
                        hsv,
                        (0, 0, 0),
                        (179, 255, black_v_max),
                    )

                    green_ratio = (
                        float(
                            cv2.countNonZero(
                                green_mask
                            )
                        )
                        / float(green_mask.size)
                    )

                    red_ratio = (
                        float(
                            cv2.countNonZero(
                                red_mask
                            )
                        )
                        / float(red_mask.size)
                    )

                    black_ratio = (
                        float(
                            cv2.countNonZero(
                                black_mask
                            )
                        )
                        / float(black_mask.size)
                    )

                    fusion_frames = int(
                        cfg.get(
                            "fusion_window_frames",
                            5,
                        )
                    )

                    fusion_min_votes = int(
                        cfg.get(
                            "fusion_min_votes",
                            3,
                        )
                    )

                    fusion_min_weight_share = float(
                        cfg.get(
                            "fusion_min_weight_share",
                            0.55,
                        )
                    )

                    fusion_color_margin = float(
                        cfg.get(
                            "fusion_color_dominance_margin",
                            0.04,
                        )
                    )

                    uncertain_color_min_votes = int(
                        cfg.get(
                            "fusion_uncertain_color_min_votes",
                            4,
                        )
                    )

                    uncertain_color_min_score = float(
                        cfg.get(
                            "fusion_uncertain_color_min_score",
                            0.45,
                        )
                    )

                    # Per-frame semantic color is diagnostic only.
                    # It does NOT change raw_label.
                    background_color = "UNKNOWN"

                    if (
                        green_ratio >= green_ratio_min
                        and green_ratio
                            > red_ratio + fusion_color_margin
                        and green_ratio
                            > black_ratio + fusion_color_margin
                    ):
                        background_color = "GREEN"

                    elif (
                        red_ratio >= red_ratio_min
                        and (
                            red_ratio - black_ratio
                        ) >= fusion_color_margin
                    ):
                        background_color = "RED"

                    elif (
                        black_ratio >= black_ratio_min
                        and (
                            black_ratio - red_ratio
                        ) >= fusion_color_margin
                    ):
                        background_color = "BLACK"

                    decision["raw_label"] = raw_label
                    decision["raw_score"] = raw_score
                    decision["raw_margin"] = raw_margin

                    # Do NOT replace CNN label based on RED/BLACK.
                    decision["label"] = raw_label
                    decision["score"] = raw_score
                    decision["margin"] = raw_margin

                    decision["background_color"] = (
                        background_color
                    )
                    decision["green_ratio"] = float(
                        green_ratio
                    )
                    decision["red_ratio"] = float(
                        red_ratio
                    )
                    decision["black_ratio"] = float(
                        black_ratio
                    )

                    decision["zero_color_override"] = False
                    decision["color_filter_blocked"] = False
                    decision["color_compatible_top5"] = []

                    decision["fusion_window_size"] = 0
                    decision["fusion_complete"] = False
                    decision["fusion_winner"] = None
                    decision["fusion_vote_count"] = 0
                    decision["fusion_weight_share"] = 0.0
                    decision["fused_color"] = None
                    decision["fusion_color_valid"] = None
                    decision["fusion_reason"] = "collecting"

                    # --------------------------------------------------
                    # Zero -> fake 10 transition guard
                    #
                    # After a confirmed zero, the display sometimes dims
                    # before the next roulette result. During that fade the
                    # CNN may strongly predict 10 even though substantial
                    # green evidence from the previous zero still remains.
                    #
                    # A real 10 has very little green evidence, so only the
                    # specific combination:
                    #
                    #   previous confirmed result == 0
                    #   raw CNN label == 10
                    #   green ratio >= configured threshold
                    #
                    # is treated as the fading previous zero.
                    #
                    # The frame is ignored completely and does not consume
                    # any position in the five-frame fusion window.
                    # --------------------------------------------------

                    # --------------------------------------------------
                    # Round Boundary Detector
                    #
                    # A new roulette round is identified by the short,
                    # almost completely black transition observed between
                    # results.
                    #
                    # IMPORTANT:
                    #   The next number is allowed to be IDENTICAL to the
                    #   previous number.
                    #
                    # Example:
                    #
                    #       25 -> blackout -> 25
                    #
                    # becomes two independent roulette results.
                    #
                    # Time is only a safety guard. The visual blackout is
                    # the primary round-boundary evidence.
                    # --------------------------------------------------

                    round_boundary_enabled = bool(
                        cfg.get(
                            "round_boundary_enabled",
                            True,
                        )
                    )

                    round_black_min = float(
                        cfg.get(
                            "round_boundary_black_min",
                            0.98,
                        )
                    )

                    round_red_max = float(
                        cfg.get(
                            "round_boundary_red_max",
                            0.10,
                        )
                    )

                    round_green_max = float(
                        cfg.get(
                            "round_boundary_green_max",
                            0.10,
                        )
                    )

                    round_raw_score_max = float(
                        cfg.get(
                            "round_boundary_raw_score_max",
                            0.20,
                        )
                    )

                    round_required_frames = int(
                        cfg.get(
                            "round_boundary_required_frames",
                            3,
                        )
                    )

                    round_min_seconds = float(
                        cfg.get(
                            "round_boundary_min_seconds_after_output",
                            6.0,
                        )
                    )

                    now_mono = time.monotonic()

                    # Initialize persistent detector state lazily so this
                    # patch does not require constructor changes.
                    if not hasattr(
                        self,
                        "_round_boundary_streak",
                    ):
                        self._round_boundary_streak = 0

                    if not hasattr(
                        self,
                        "_round_boundary_open",
                    ):
                        self._round_boundary_open = False

                    if not hasattr(
                        self,
                        "_round_guard_last_registration_mono",
                    ):
                        self._round_guard_last_registration_mono = (
                            now_mono
                        )

                    seconds_since_output = (
                        now_mono
                        - self._round_guard_last_registration_mono
                    )

                    strong_round_blackout = (
                        round_boundary_enabled
                        and self.last_output_number is not None
                        and seconds_since_output
                            >= round_min_seconds
                        and black_ratio
                            >= round_black_min
                        and red_ratio
                            <= round_red_max
                        and green_ratio
                            <= round_green_max
                        and raw_score
                            <= round_raw_score_max
                    )

                    # Every strong-blackout frame is held outside the
                    # 5-frame number fusion window.
                    round_boundary_hold = bool(
                        strong_round_blackout
                    )

                    round_boundary_detected = False

                    if (
                        strong_round_blackout
                        and not self._round_boundary_open
                    ):
                        self._round_boundary_streak += 1

                    elif not strong_round_blackout:
                        self._round_boundary_streak = 0

                    if (
                        not self._round_boundary_open
                        and self._round_boundary_streak
                            >= round_required_frames
                    ):
                        # A genuine new roulette round now exists.
                        #
                        # Unlock BEFORE the next visible result so the
                        # next result may equal last_output_number.
                        self.scan_locked = False

                        self._round_boundary_open = True
                        self._round_boundary_streak = 0

                        self.reset_candidate()

                        round_boundary_detected = True

                    decision[
                        "round_boundary_strong_blackout"
                    ] = bool(
                        strong_round_blackout
                    )

                    decision[
                        "round_boundary_detected"
                    ] = bool(
                        round_boundary_detected
                    )

                    decision[
                        "round_boundary_open"
                    ] = bool(
                        self._round_boundary_open
                    )

                    decision[
                        "round_boundary_streak"
                    ] = int(
                        self._round_boundary_streak
                    )

                    decision[
                        "round_seconds_since_output"
                    ] = float(
                        seconds_since_output
                    )

                    zero_transition_10_ignored = (
                        bool(
                            cfg.get(
                                "zero_transition_10_guard_enabled",
                                True,
                            )
                        )
                        and self.last_output_number is not None
                        and int(self.last_output_number) == 0
                        and raw_label == 10
                        and green_ratio
                            >= float(
                                cfg.get(
                                    "zero_transition_10_green_min",
                                    0.20,
                                )
                            )
                    )

                    decision[
                        "zero_transition_10_ignored"
                    ] = bool(
                        zero_transition_10_ignored
                    )

                    self.latest_decision = decision

                    # --------------------------------------------------
                    # Only RESULT_VISIBLE frames participate.
                    # A real non-result transition starts a new window.
                    # --------------------------------------------------

                    if round_boundary_hold:
                        self.reset_candidate()

                        decision["fusion_reason"] = (
                            "round_boundary_blackout"
                        )

                        self.latest_decision = decision

                    elif zero_transition_10_ignored:
                        self.reset_candidate()

                        decision["fusion_reason"] = (
                            "zero_transition_10_ignored"
                        )

                        self.latest_decision = decision

                    elif state != "RESULT_VISIBLE":
                        self.reset_candidate()

                    else:
                        # While locked on the previous output, frames that
                        # still clearly represent that same result do not
                        # start a new five-frame window.
                        #
                        # GREEN is checked first because physical zero can
                        # be predicted by CNN as 10 or 8.
                        green_hint = (
                            green_ratio >= green_ratio_min
                            and green_ratio
                                > red_ratio + fusion_color_margin
                            and green_ratio
                                > black_ratio + fusion_color_margin
                        )

                        effective_hint = (
                            0
                            if green_hint
                            else raw_label
                        )

                        if (
                            self.scan_locked
                            and self.last_output_number is not None
                            and int(effective_hint)
                                == int(self.last_output_number)
                        ):
                            self.reset_candidate()

                        else:
                            # A low-confidence/noisy CNN frame is still part
                            # of the five-frame window. Its small confidence
                            # simply contributes less weight.
                            self.candidate_predictions.append(
                                decision
                            )

                            window = list(
                                self.candidate_predictions
                            )

                            # --------------------------------------------------
                            # Current weighted leader for UI diagnostics.
                            # --------------------------------------------------

                            number_stats = {}

                            for item in window:
                                n = int(
                                    item["raw_label"]
                                )
                                s = max(
                                    float(
                                        item["raw_score"]
                                    ),
                                    1e-6,
                                )

                                stat = number_stats.setdefault(
                                    n,
                                    {
                                        "weight": 0.0,
                                        "count": 0,
                                        "scores": [],
                                        "margins": [],
                                    },
                                )

                                stat["weight"] += s
                                stat["count"] += 1
                                stat["scores"].append(
                                    float(
                                        item["raw_score"]
                                    )
                                )
                                stat["margins"].append(
                                    float(
                                        item["raw_margin"]
                                    )
                                )

                            leader = max(
                                number_stats,
                                key=lambda n: (
                                    number_stats[n]["weight"],
                                    number_stats[n]["count"],
                                ),
                            )

                            self.candidate_label = int(
                                leader
                            )
                            self.candidate_streak = len(
                                window
                            )

                            decision[
                                "fusion_window_size"
                            ] = len(window)

                            decision[
                                "fusion_winner"
                            ] = int(leader)

                            # --------------------------------------------------
                            # Exactly at frame 5: produce one final decision.
                            # --------------------------------------------------

                            if len(window) >= fusion_frames:
                                window = window[
                                    -fusion_frames:
                                ]

                                avg_green = (
                                    sum(
                                        float(
                                            p["green_ratio"]
                                        )
                                        for p in window
                                    )
                                    / len(window)
                                )

                                avg_red = (
                                    sum(
                                        float(
                                            p["red_ratio"]
                                        )
                                        for p in window
                                    )
                                    / len(window)
                                )

                                avg_black = (
                                    sum(
                                        float(
                                            p["black_ratio"]
                                        )
                                        for p in window
                                    )
                                    / len(window)
                                )

                                fused_color = "UNKNOWN"

                                if (
                                    avg_green
                                        >= green_ratio_min
                                    and avg_green
                                        > avg_red
                                            + fusion_color_margin
                                    and avg_green
                                        > avg_black
                                            + fusion_color_margin
                                ):
                                    fused_color = "GREEN"

                                elif (
                                    avg_red
                                        >= red_ratio_min
                                    and (
                                        avg_red
                                        - avg_black
                                    ) >= fusion_color_margin
                                ):
                                    fused_color = "RED"

                                elif (
                                    avg_black
                                        >= black_ratio_min
                                    and (
                                        avg_black
                                        - avg_red
                                    ) >= fusion_color_margin
                                ):
                                    fused_color = "BLACK"

                                # Rebuild number statistics for the exact
                                # five-frame final window.
                                number_stats = {}

                                for item in window:
                                    n = int(
                                        item["raw_label"]
                                    )

                                    s = max(
                                        float(
                                            item["raw_score"]
                                        ),
                                        1e-6,
                                    )

                                    stat = number_stats.setdefault(
                                        n,
                                        {
                                            "weight": 0.0,
                                            "count": 0,
                                            "scores": [],
                                            "margins": [],
                                        },
                                    )

                                    stat["weight"] += s
                                    stat["count"] += 1

                                    stat["scores"].append(
                                        float(
                                            item["raw_score"]
                                        )
                                    )

                                    stat["margins"].append(
                                        float(
                                            item["raw_margin"]
                                        )
                                    )

                                total_weight = sum(
                                    x["weight"]
                                    for x
                                    in number_stats.values()
                                )

                                cnn_winner = max(
                                    number_stats,
                                    key=lambda n: (
                                        number_stats[n]["weight"],
                                        number_stats[n]["count"],
                                    ),
                                )

                                winner_stat = (
                                    number_stats[
                                        cnn_winner
                                    ]
                                )

                                winner_votes = int(
                                    winner_stat[
                                        "count"
                                    ]
                                )

                                winner_weight_share = (
                                    float(
                                        winner_stat[
                                            "weight"
                                        ]
                                    )
                                    / max(
                                        total_weight,
                                        1e-9,
                                    )
                                )

                                winner_avg_score = (
                                    sum(
                                        winner_stat[
                                            "scores"
                                        ]
                                    )
                                    / len(
                                        winner_stat[
                                            "scores"
                                        ]
                                    )
                                )

                                winner_avg_margin = (
                                    sum(
                                        winner_stat[
                                            "margins"
                                        ]
                                    )
                                    / len(
                                        winner_stat[
                                            "margins"
                                        ]
                                    )
                                )

                                final_number = int(
                                    cnn_winner
                                )

                                fusion_reason = (
                                    "cnn_color_fusion"
                                )

                                # ------------------------------------------
                                # GREEN special case:
                                # zero is the only green roulette number.
                                # ------------------------------------------

                                if fused_color == "GREEN":
                                    final_number = 0

                                    winner_votes = len(
                                        window
                                    )

                                    winner_weight_share = 1.0

                                    winner_avg_score = float(
                                        avg_green
                                    )

                                    winner_avg_margin = max(
                                        0.0,
                                        float(
                                            avg_green
                                            - max(
                                                avg_red,
                                                avg_black,
                                            )
                                        ),
                                    )

                                    number_valid = True
                                    color_valid = True
                                    fusion_reason = (
                                        "green_zero"
                                    )

                                else:
                                    number_valid = (
                                        winner_votes
                                            >= fusion_min_votes
                                        and winner_weight_share
                                            >= fusion_min_weight_share
                                        and winner_avg_score
                                            >= min_frame_score
                                        and winner_avg_margin
                                            >= min_frame_margin
                                    )

                                    if (
                                        final_number
                                        in RED_NUMBERS
                                    ):
                                        expected_color = "RED"

                                    elif (
                                        final_number
                                        in BLACK_NUMBERS
                                    ):
                                        expected_color = "BLACK"

                                    elif final_number == 0:
                                        expected_color = "GREEN"

                                    else:
                                        expected_color = "UNKNOWN"

                                    if (
                                        fused_color
                                        == expected_color
                                    ):
                                        color_valid = True

                                    elif (
                                        fused_color
                                        == "UNKNOWN"
                                        and winner_votes
                                            >= uncertain_color_min_votes
                                        and winner_avg_score
                                            >= uncertain_color_min_score
                                    ):
                                        # Five frames are already finished.
                                        # Do not wait longer merely because
                                        # RED/BLACK remained close.
                                        color_valid = True
                                        fusion_reason = (
                                            "cnn_strong_color_uncertain"
                                        )

                                    else:
                                        color_valid = False
                                        fusion_reason = (
                                            "color_conflict"
                                        )

                                # ------------------------------------------
                                # Targeted number-6 color override
                                #
                                # On the real display, bright illumination can
                                # make BLACK evidence disappear even while CNN
                                # consistently recognizes number 6.
                                #
                                # This exception applies ONLY to number 6.
                                #
                                # Explicit RED evidence still rejects 6.
                                # Strong GREEN evidence is reserved for zero.
                                # ------------------------------------------
                                six_color_override = (
                                    number_valid
                                    and final_number == 6
                                    and avg_red < red_ratio_min
                                    and avg_green < green_ratio_min
                                )

                                if (
                                    six_color_override
                                    and not color_valid
                                ):
                                    color_valid = True
                                    fusion_reason = (
                                        "cnn_6_color_override"
                                    )

                                decision[
                                    "six_color_override"
                                ] = bool(
                                    six_color_override
                                )

                                accepted = (
                                    number_valid
                                    and color_valid
                                )

                                # Strict duplicate suppression remains:
                                # the same already-confirmed result cannot
                                # become a new event.
                                if (
                                    accepted
                                    and self.scan_locked
                                    and self.last_output_number
                                        is not None
                                    and final_number
                                        == int(
                                            self.last_output_number
                                        )
                                ):
                                    accepted = False
                                    fusion_reason = (
                                        "same_result_locked"
                                    )

                                decision[
                                    "fusion_complete"
                                ] = True

                                decision[
                                    "fusion_winner"
                                ] = int(
                                    final_number
                                )

                                decision[
                                    "fusion_vote_count"
                                ] = int(
                                    winner_votes
                                )

                                decision[
                                    "fusion_weight_share"
                                ] = float(
                                    winner_weight_share
                                )

                                decision[
                                    "fused_color"
                                ] = fused_color

                                decision[
                                    "fused_green_ratio"
                                ] = float(
                                    avg_green
                                )

                                decision[
                                    "fused_red_ratio"
                                ] = float(
                                    avg_red
                                )

                                decision[
                                    "fused_black_ratio"
                                ] = float(
                                    avg_black
                                )

                                decision[
                                    "fusion_color_valid"
                                ] = bool(
                                    color_valid
                                )

                                decision[
                                    "fusion_number_valid"
                                ] = bool(
                                    number_valid
                                )

                                decision[
                                    "fusion_reason"
                                ] = fusion_reason

                                # Display the FINAL fused number on the
                                # fifth frame.
                                decision["label"] = int(
                                    final_number
                                )

                                decision["score"] = float(
                                    winner_avg_score
                                )

                                decision["margin"] = float(
                                    winner_avg_margin
                                )

                                self.latest_decision = (
                                    decision
                                )

                                if accepted:
                                    event = self.record_number(
                                        final_number,
                                        winner_avg_score,
                                        winner_avg_margin,
                                    )

                                    # A result has now been registered for
                                    # the newly opened round. Lock the round
                                    # again until the NEXT blackout boundary.
                                    self._round_boundary_open = False
                                    self._round_boundary_streak = 0

                                    self._round_guard_last_registration_mono = (
                                        time.monotonic()
                                    )

                                    decision[
                                        "round_boundary_open"
                                    ] = False

                                # The five-frame window is finished,
                                # regardless of TRUE/FALSE.
                                self.reset_candidate()

                except Exception as exc:
                    state = f"ERROR: {exc}"
                    self.reset_candidate()
            status = {
                "ready": True,
                "monitoring": self.monitoring,
                "sessionId": self.session_id,
                "state": state,
                "selectedList": self.selected_list,
                "decision": decision or self.latest_decision,
                "candidateLabel": self.candidate_label,
                "candidateStreak": self.candidate_streak,
                "requiredConsecutiveFrames": required_consecutive_frames,
                "minFrameScore": min_frame_score,
                "minFrameMargin": min_frame_margin,
                "backgroundColor": (
                    (decision or self.latest_decision or {}).get(
                        "background_color",
                        "UNKNOWN",
                    )
                ),
                "greenRatio": float(
                    (decision or self.latest_decision or {}).get(
                        "green_ratio",
                        0.0,
                    )
                ),
                "redRatio": float(
                    (decision or self.latest_decision or {}).get(
                        "red_ratio",
                        0.0,
                    )
                ),
                "blackRatio": float(
                    (decision or self.latest_decision or {}).get(
                        "black_ratio",
                        0.0,
                    )
                ),
                "zeroColorOverride": bool(
                    (decision or self.latest_decision or {}).get(
                        "zero_color_override",
                        False,
                    )
                ),
                "colorFilterBlocked": bool(
                    (decision or self.latest_decision or {}).get(
                        "color_filter_blocked",
                        False,
                    )
                ),
                "rawDecisionLabel": (
                    (decision or self.latest_decision or {}).get(
                        "raw_label"
                    )
                ),
                "rawDecisionScore": (
                    (decision or self.latest_decision or {}).get(
                        "raw_score"
                    )
                ),
                "colorCompatibleTop5": (
                    (decision or self.latest_decision or {}).get(
                        "color_compatible_top5",
                        [],
                    )
                ),
                "lastOutputNumber": self.last_output_number,
                "scanLocked": self.scan_locked,
                "unlockGapStreak": self.unlock_gap_streak,
                "unlockChangeLabel": self.unlock_change_label,
                "unlockChangeStreak": self.unlock_change_streak,
                "latestMatch": self.latest_match,
                "history": list(self.history)[:200],
                "event": event,
                "metrics": metrics,
                "hardware": self.actions.status(),
            }
            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_roi = roi.copy()
                self.status = json.loads(json.dumps(status, default=str))
            time.sleep(delay)

    def public_status(self):
        with self.lock:
            return json.loads(json.dumps(self.status, default=str))

    def get_frame(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is not None:
            x, y, w, h = roi_rect_from_shape(frame.shape, self.cfg["roi"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame


    def get_focus_frame(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            camera_enabled = getattr(self, "camera_enabled", True)

        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        if not camera_enabled:
            black = np.zeros_like(frame)
            cv2.putText(
                black,
                "CAMERA OFF",
                (80, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            return black

        h_img, w_img = frame.shape[:2]
        x, y, w, h = roi_rect_from_shape(frame.shape, self.cfg["roi"])

        # Focus view: show a larger zone around ROI, not the full camera frame.
        # Increase this factor if you want more surroundings; decrease for closer zoom.
        factor = float(self.cfg.get("ui_focus_view", {}).get("roi_context_factor", 3.0))

        cx = x + w / 2.0
        cy = y + h / 2.0

        crop_w = max(int(w * factor), int(w_img * 0.20))
        crop_h = max(int(h * factor), int(h_img * 0.24))

        # Keep a 16:9-ish crop so it looks good inside the camera panel.
        target_ratio = 16 / 9
        current_ratio = crop_w / max(1, crop_h)

        if current_ratio < target_ratio:
            crop_w = int(crop_h * target_ratio)
        else:
            crop_h = int(crop_w / target_ratio)

        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w_img:
            x1 -= (x2 - w_img)
            x2 = w_img
        if y2 > h_img:
            y1 -= (y2 - h_img)
            y2 = h_img

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        crop = frame[y1:y2, x1:x2].copy()

        # Draw ROI rectangle in crop coordinates.
        rx1 = max(0, x - x1)
        ry1 = max(0, y - y1)
        rx2 = min(crop.shape[1] - 1, x + w - x1)
        ry2 = min(crop.shape[0] - 1, y + h - y1)

        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 230, 255), 2)

        # Small center cross for easier alignment.
        ccx = int((rx1 + rx2) / 2)
        ccy = int((ry1 + ry2) / 2)
        cv2.line(crop, (ccx - 8, ccy), (ccx + 8, ccy), (0, 230, 255), 1)
        cv2.line(crop, (ccx, ccy - 8), (ccx, ccy + 8), (0, 230, 255), 1)

        return crop

    def get_roi(self):
        with self.lock:
            return None if self.latest_roi is None else self.latest_roi.copy()

    def clear_history(self):
        with self.lock:
            self.history.clear()
            self.latest_match = None
        return {"ok": True}


def encode_jpeg(frame, quality=85):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else None


HTML = r'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=800, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>Roulette Vision</title>
  <link rel="stylesheet" href="/ui/styles.css">
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/ui/app.js"></script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    hub: FinalVisionHub = None

    def log_message(self, fmt, *args):
        return

    def _json_body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)




    def serve_static_file(self, file_path, head_only=False):
        import mimetypes

        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        data = file_path.read_bytes()
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

        if file_path.suffix == ".js":
            mime = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif file_path.suffix == ".html":
            mime = "text/html; charset=utf-8"
        elif file_path.suffix == ".png":
            mime = "image/png"

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()

        if not head_only:
            self.wfile.write(data)

    def do_HEAD(self):
        self.handle_get_request(head_only=True)

    def do_GET(self):
        self.handle_get_request(head_only=False)

    def handle_get_request(self, head_only=False):
        path = urlparse(self.path).path

        if path == "/":
            self.serve_static_file(Path("data/ui/index.html"), head_only=head_only)
            return

        if path.startswith("/ui/"):
            rel = path[len("/ui/"):].lstrip("/")
            if ".." in rel or rel.startswith("/"):
                self.send_error(404)
                return
            self.serve_static_file(Path("data/ui") / rel, head_only=head_only)
            return

        if path == "/assets/logo.png":
            self.serve_static_file(Path("data/assets/logo.png"), head_only=head_only)
            return

        if path == "/api/settings":
            if head_only:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                return
            data = self.hub.store.get()
            data["hardwareStatus"] = self.hub.actions.status()
            self.send_json(data)
            return

        if path == "/api/status":
            if head_only:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                return
            self.send_json(self.hub.public_status())
            return

        if path == "/stream.mjpg":
            if head_only:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                return
            self.stream("frame")
            return

        if path == "/focus.mjpg":
            if head_only:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                return
            self.stream("focus")
            return

        if path == "/roi.mjpg":
            if head_only:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                return
            self.stream("roi")
            return

        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/shutdown":
            import shutil
            import subprocess
            import time

            shutdown_cmd = shutil.which("shutdown") or "/usr/sbin/shutdown"

            try:
                proc = subprocess.Popen(
                    ["sudo", "-n", shutdown_cmd, "-h", "now"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                time.sleep(0.2)

                if proc.poll() is not None and proc.returncode != 0:
                    _out, err = proc.communicate(timeout=0.2)
                    self.send_json({
                        "ok": False,
                        "error": err.strip() or "Shutdown command failed."
                    })
                    return

                self.send_json({
                    "ok": True,
                    "message": "Safe shutdown started."
                })
                return

            except Exception as exc:
                self.send_json({
                    "ok": False,
                    "error": str(exc)
                })
                return

        try:
            if path == "/api/lists":
                self.send_json(self.hub.store.update_lists(self._json_body()))
                self.hub.selected_list = self.hub.store.selected_list()
            elif path == "/api/hardware":
                saved = self.hub.store.update_hardware(self._json_body())
                self.hub.actions.update_config(saved)
                self.send_json(saved)
            elif path == "/api/double-match":
                body = self._json_body()
                enabled = self.hub.store.update_double_match(
                    body.get("enabled", False)
                )
                self.send_json({
                    "ok": True,
                    "doubleMatchEnabled": enabled,
                })
            elif path == "/api/spin-auto":
                body = self._json_body()
                enabled = self.hub.store.update_spin_auto(
                    body.get("enabled", True)
                )
                self.send_json({
                    "ok": True,
                    "spinAutoEnabled": enabled,
                })
            elif path == "/api/start":
                body = self._json_body()
                self.send_json(self.hub.start_monitoring(body.get("selectedListId")))
            elif path == "/api/stop":
                self.send_json(self.hub.stop_monitoring())
            elif path == "/api/history/clear":
                self.send_json(self.hub.clear_history())
            elif path == "/api/test-action":
                self.send_json(self.hub.actions.trigger_async(number=None, list_name="manual", reason="manual_test"))
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)

    def stream(self, mode):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            if mode == "roi":
                frame = self.hub.get_roi()
            elif mode == "focus":
                frame = self.hub.get_focus_frame()
            else:
                frame = self.hub.get_frame()
            if frame is None:
                time.sleep(0.05); continue
            if mode == "roi":
                frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            jpg = encode_jpeg(frame)
            if not jpg:
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg + b"\r\n")
                time.sleep(0.06)
            except Exception:
                break


# Roulette Vision calibration API
try:
    from roulette_vision.calibration_api import install_calibration_api
    install_calibration_api(app)
    print("Calibration API installed: /calibrate")
except Exception as exc:
    print("Calibration API install failed:", exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--source", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs()
    cfg.setdefault("final_app", {"min_stable_frames": 2, "min_live_score": 0.70, "min_live_margin": 0.10, "min_confirm_score": 0.85, "min_confirm_margin": 0.20})
    hub = FinalVisionHub(cfg, source=args.source)
    Handler.hub = hub
    hub.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Roulette final app: http://192.168.1.251:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        server.server_close(); hub.close()

if __name__ == "__main__":
    main()
