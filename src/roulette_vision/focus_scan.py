import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_metrics


def set_manual_focus(camera, lens_position):
    try:
        from libcamera import controls
        camera.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": float(lens_position)
        })
        return True
    except Exception as e:
        print(f"WARNING: could not set LensPosition={lens_position}: {e}")
        return False


def update_config_focus(config_path, best_lens_position):
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("camera", {})
    cfg["camera"].setdefault("controls", {})
    cfg["camera"]["controls"]["af_mode"] = "manual"
    cfg["camera"]["controls"]["lens_position"] = float(best_lens_position)

    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("")
    print("Config updated with manual focus:")
    print(json.dumps(cfg["camera"]["controls"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--start", type=float, default=3.0)
    parser.add_argument("--end", type=float, default=14.0)
    parser.add_argument("--steps", type=int, default=23)
    parser.add_argument("--frames-per-step", type=int, default=8)
    parser.add_argument("--settle", type=float, default=0.45)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    out_dir = Path("data/debug/focus_scan")
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = FrameSource(cfg["camera"])
    camera.start()

    rows = []
    best = None

    try:
        positions = np.linspace(args.start, args.end, args.steps)

        for pos in positions:
            ok = set_manual_focus(camera, pos)
            if not ok:
                continue

            time.sleep(args.settle)

            sharpness_values = []
            overexposed_values = []
            white_values = []
            roi_samples = []
            frame_samples = []

            for _ in range(args.frames_per_step):
                frame = camera.read()
                if frame is None:
                    continue

                roi = crop_relative_roi(frame, cfg["roi"])
                m = roi_metrics(roi)

                sharpness_values.append(m["sharpness"])
                overexposed_values.append(m.get("overexposed_ratio", 0.0))
                white_values.append(m["white_ratio"])

                roi_samples.append(roi)
                frame_samples.append(frame)

                time.sleep(0.05)

            if not sharpness_values:
                continue

            sharp_med = float(np.median(sharpness_values))
            sharp_avg = float(np.mean(sharpness_values))
            over_avg = float(np.mean(overexposed_values))
            white_avg = float(np.mean(white_values))

            # Penalize strong overexposure, but keep sharpness as main score.
            quality_score = sharp_med * (1.0 - min(0.5, over_avg))

            row = {
                "lens_position": float(pos),
                "sharpness_median": sharp_med,
                "sharpness_average": sharp_avg,
                "overexposed_ratio": over_avg,
                "white_ratio": white_avg,
                "quality_score": quality_score
            }
            rows.append(row)

            mid = len(roi_samples) // 2
            roi_path = out_dir / f"focus_{pos:.2f}_roi.png"
            frame_path = out_dir / f"focus_{pos:.2f}_frame.png"

            cv2.imwrite(str(roi_path), roi_samples[mid])
            cv2.imwrite(str(frame_path), frame_samples[mid])

            print(
                f"LensPosition={pos:.2f} "
                f"sharp_med={sharp_med:.2f} "
                f"over={over_avg:.4f} "
                f"white={white_avg:.4f} "
                f"quality={quality_score:.2f}"
            )

            if best is None or quality_score > best["quality_score"]:
                best = row

    finally:
        camera.close()

    csv_path = out_dir / "focus_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lens_position",
                "sharpness_median",
                "sharpness_average",
                "overexposed_ratio",
                "white_ratio",
                "quality_score"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print(f"Saved focus scan results: {csv_path}")

    if best is None:
        print("No valid focus result.")
        return

    print("")
    print("BEST FOCUS:")
    print(json.dumps(best, indent=2))

    if args.apply:
        update_config_focus(args.config, best["lens_position"])


if __name__ == "__main__":
    main()
