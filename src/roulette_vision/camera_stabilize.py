import argparse
import json
import statistics
import time
from pathlib import Path
from copy import deepcopy

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_relative_roi, roi_metrics


def median_number(values):
    cleaned = []
    for v in values:
        try:
            cleaned.append(float(v))
        except Exception:
            pass
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def median_pair(values):
    left = []
    right = []

    for v in values:
        try:
            if isinstance(v, (list, tuple)) and len(v) == 2:
                left.append(float(v[0]))
                right.append(float(v[1]))
        except Exception:
            pass

    if not left or not right:
        return None

    return [float(statistics.median(left)), float(statistics.median(right))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-continuous-focus", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()

    original_cfg = load_config(args.config)
    run_cfg = deepcopy(original_cfg)

    # During stabilization, let camera auto-adjust exposure and white balance.
    run_cfg.setdefault("camera", {})
    run_cfg["camera"].setdefault("controls", {})
    controls = run_cfg["camera"]["controls"]

    controls["af_mode"] = "continuous"
    controls["awb_enable"] = True
    controls.pop("exposure_time", None)
    controls.pop("analogue_gain", None)
    controls.pop("colour_gains", None)

    camera = FrameSource(run_cfg["camera"])
    camera.start()

    exposure_values = []
    gain_values = []
    colour_gain_values = []
    lens_values = []

    sharpness_values = []
    overexposed_values = []
    white_values = []

    print("Stabilizing camera. Keep the image steady...")
    print("Collecting samples...")

    start = time.time()

    try:
        while time.time() - start < args.seconds:
            frame = camera.read()
            if frame is None:
                continue

            roi = crop_relative_roi(frame, run_cfg["roi"])
            metrics = roi_metrics(roi)
            meta = camera.metadata()

            sharpness_values.append(metrics.get("sharpness", 0.0))
            overexposed_values.append(metrics.get("overexposed_ratio", 0.0))
            white_values.append(metrics.get("white_ratio", 0.0))

            if "ExposureTime" in meta:
                exposure_values.append(meta["ExposureTime"])

            if "AnalogueGain" in meta:
                gain_values.append(meta["AnalogueGain"])

            if "ColourGains" in meta:
                colour_gain_values.append(meta["ColourGains"])

            if "LensPosition" in meta:
                lens_values.append(meta["LensPosition"])

            time.sleep(0.08)

    finally:
        camera.close()

    exposure = median_number(exposure_values)
    gain = median_number(gain_values)
    colour_gains = median_pair(colour_gain_values)
    lens_position = median_number(lens_values)

    sharpness = median_number(sharpness_values)
    overexposed = median_number(overexposed_values)
    white_ratio = median_number(white_values)

    result = {
        "exposure_time": exposure,
        "analogue_gain": gain,
        "colour_gains": colour_gains,
        "lens_position": lens_position,
        "roi_sharpness_median": sharpness,
        "roi_overexposed_ratio_median": overexposed,
        "roi_white_ratio_median": white_ratio
    }

    print("")
    print("Camera stabilization result:")
    print(json.dumps(result, indent=2))

    if not args.apply:
        print("")
        print("Config not changed. Run again with --apply to save these settings.")
        return

    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("camera", {})
    cfg["camera"].setdefault("controls", {})
    new_controls = cfg["camera"]["controls"]

    if args.keep_continuous_focus:
        new_controls["af_mode"] = "continuous"
        new_controls.pop("lens_position", None)
    else:
        if lens_position is not None:
            new_controls["af_mode"] = "manual"
            new_controls["lens_position"] = float(lens_position)
        else:
            new_controls["af_mode"] = "continuous"

    if exposure is not None and gain is not None:
        new_controls["exposure_time"] = int(round(exposure))
        new_controls["analogue_gain"] = float(gain)

    if colour_gains is not None:
        new_controls["awb_enable"] = False
        new_controls["colour_gains"] = colour_gains
    else:
        new_controls["awb_enable"] = True

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("")
    print("Updated camera controls in data/config.json:")
    print(json.dumps(cfg["camera"]["controls"], indent=2))


if __name__ == "__main__":
    main()
