import argparse
import json
import statistics
import time
from pathlib import Path
from copy import deepcopy

import numpy as np

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_config_roi, roi_metrics


def median(values):
    cleaned = []
    for v in values:
        try:
            cleaned.append(float(v))
        except Exception:
            pass
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def set_af_continuous(camera):
    try:
        from libcamera import controls
        camera.set_controls({
            "AfMode": controls.AfModeEnum.Continuous
        })
        return True
    except Exception as e:
        print("WARNING: could not set continuous AF:", e)
        return False


def set_manual_focus(camera, lens_position):
    try:
        from libcamera import controls
        camera.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": float(lens_position)
        })
        return True
    except Exception as e:
        print(f"WARNING: could not set manual focus {lens_position}: {e}")
        return False


def sample_quality(camera, cfg, seconds=1.2, interval=0.07):
    sharpness = []
    edge_sharpness = []
    over = []
    white = []

    start = time.time()

    while time.time() - start < seconds:
        frame = camera.read()
        if frame is None:
            continue

        focus_roi = crop_config_roi(frame, cfg, "focus_roi")
        m = roi_metrics(focus_roi)

        sharpness.append(m["sharpness"])
        edge_sharpness.append(m["digit_edge_sharpness"])
        over.append(m["overexposed_ratio"])
        white.append(m["white_ratio"])

        time.sleep(interval)

    sharp = median(sharpness) or 0.0
    edge = median(edge_sharpness) or 0.0
    over_m = median(over) or 0.0
    white_m = median(white) or 0.0

    # Penalize overexposure a little.
    quality = edge * (1.0 - min(0.40, over_m))

    return {
        "sharpness": sharp,
        "digit_edge_sharpness": edge,
        "overexposed_ratio": over_m,
        "white_ratio": white_m,
        "quality": quality
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--start", type=float, default=6.0)
    parser.add_argument("--end", type=float, default=18.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--settle", type=float, default=0.35)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-improvement", type=float, default=1.08)
    args = parser.parse_args()

    ensure_project_dirs()
    cfg = load_config(args.config)

    run_cfg = deepcopy(cfg)
    run_cfg.setdefault("camera", {})
    run_cfg["camera"].setdefault("controls", {})
    run_cfg["camera"]["controls"] = {
        "af_mode": "continuous",
        "awb_enable": True
    }

    camera = FrameSource(run_cfg["camera"])
    camera.start()

    try:
        print("")
        print("Measuring autofocus baseline...")
        set_af_continuous(camera)
        time.sleep(1.8)
        baseline = sample_quality(camera, run_cfg, seconds=args.sample_seconds)

        print("AUTOFOCUS BASELINE:")
        print(json.dumps(baseline, indent=2))

        best = None
        rows = []

        print("")
        print("Scanning manual focus positions...")

        for pos in np.linspace(args.start, args.end, args.steps):
            if not set_manual_focus(camera, pos):
                continue

            time.sleep(args.settle)

            q = sample_quality(camera, run_cfg, seconds=args.sample_seconds)
            q["lens_position"] = float(pos)
            rows.append(q)

            print(
                f"Lens={pos:.2f} "
                f"sharp={q['sharpness']:.1f} "
                f"edge={q['digit_edge_sharpness']:.1f} "
                f"over={q['overexposed_ratio']:.4f} "
                f"quality={q['quality']:.1f}"
            )

            if best is None or q["quality"] > best["quality"]:
                best = q

    finally:
        camera.close()

    out_dir = Path("data/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "camera_optimize_report.json"

    report = {
        "baseline_autofocus": baseline,
        "best_manual": best,
        "manual_scan": rows
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print(f"Saved report: {report_path}")

    if best is None:
        print("No manual focus result. Keeping autofocus.")
        return

    print("")
    print("BEST MANUAL:")
    print(json.dumps(best, indent=2))

    should_use_manual = best["quality"] >= baseline["quality"] * float(args.min_improvement)

    print("")
    print(f"Use manual focus? {should_use_manual}")

    if not args.apply:
        print("Config not changed. Add --apply to save the decision.")
        return

    path = Path(args.config)
    real_cfg = json.loads(path.read_text())

    real_cfg.setdefault("camera", {})
    real_cfg["camera"].setdefault("controls", {})

    if should_use_manual:
        real_cfg["camera"]["controls"] = {
            "af_mode": "manual",
            "lens_position": float(best["lens_position"]),
            "awb_enable": True
        }
    else:
        real_cfg["camera"]["controls"] = {
            "af_mode": "continuous",
            "awb_enable": True
        }

    path.write_text(json.dumps(real_cfg, indent=2))

    print("")
    print("Updated camera controls:")
    print(json.dumps(real_cfg["camera"]["controls"], indent=2))


if __name__ == "__main__":
    main()
