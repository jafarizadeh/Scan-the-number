import argparse
import json
import statistics
import time
from pathlib import Path

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .vision import crop_config_roi, roi_metrics


def median_number(values):
    cleaned = []
    for value in values:
        try:
            cleaned.append(float(value))
        except Exception:
            pass
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def median_pair(values):
    left = []
    right = []

    for value in values:
        try:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                left.append(float(value[0]))
                right.append(float(value[1]))
        except Exception:
            pass

    if not left or not right:
        return None

    return [float(statistics.median(left)), float(statistics.median(right))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--keep-awb", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    cfg = load_config(args.config)

    camera = FrameSource(cfg["camera"])
    camera.start()

    exposure_values = []
    gain_values = []
    colour_gain_values = []
    lens_values = []

    roi_sharpness = []
    roi_edge_sharpness = []
    roi_overexposed = []

    print("Freezing camera parameters.")
    print("Keep the camera and screen stable. The result number should be visible and bright.")

    start = time.time()

    try:
        while time.time() - start < args.seconds:
            frame = camera.read()
            if frame is None:
                continue

            roi = crop_config_roi(frame, cfg, "roi")
            metrics = roi_metrics(roi)
            metadata = camera.metadata()

            roi_sharpness.append(metrics.get("sharpness", 0.0))
            roi_edge_sharpness.append(metrics.get("digit_edge_sharpness", 0.0))
            roi_overexposed.append(metrics.get("overexposed_ratio", 0.0))

            if "ExposureTime" in metadata:
                exposure_values.append(metadata["ExposureTime"])

            if "AnalogueGain" in metadata:
                gain_values.append(metadata["AnalogueGain"])

            if "ColourGains" in metadata:
                colour_gain_values.append(metadata["ColourGains"])

            if "LensPosition" in metadata:
                lens_values.append(metadata["LensPosition"])

            time.sleep(0.08)

    finally:
        camera.close()

    exposure = median_number(exposure_values)
    gain = median_number(gain_values)
    colour_gains = median_pair(colour_gain_values)
    lens_position_from_meta = median_number(lens_values)

    sharpness = median_number(roi_sharpness)
    edge_sharpness = median_number(roi_edge_sharpness)
    overexposed = median_number(roi_overexposed)

    path = Path(args.config)
    real_cfg = json.loads(path.read_text())

    real_cfg.setdefault("camera", {})
    real_cfg["camera"].setdefault("controls", {})

    old_controls = real_cfg["camera"].get("controls", {})
    lens_position = old_controls.get("lens_position", lens_position_from_meta)

    new_controls = {}

    if old_controls.get("af_mode") == "manual" and lens_position is not None:
        new_controls["af_mode"] = "manual"
        new_controls["lens_position"] = float(lens_position)
    else:
        new_controls["af_mode"] = "continuous"

    if exposure is not None and gain is not None:
        new_controls["exposure_time"] = int(round(exposure))
        new_controls["analogue_gain"] = float(gain)

    if args.keep_awb:
        new_controls["awb_enable"] = True
    else:
        if colour_gains is not None:
            new_controls["awb_enable"] = False
            new_controls["colour_gains"] = colour_gains
        else:
            new_controls["awb_enable"] = True

    real_cfg["camera"]["controls"] = new_controls

    path.write_text(json.dumps(real_cfg, indent=2))

    result = {
        "saved_controls": new_controls,
        "roi_sharpness_median": sharpness,
        "roi_digit_edge_sharpness_median": edge_sharpness,
        "roi_overexposed_ratio_median": overexposed
    }

    print("")
    print("Camera frozen:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
