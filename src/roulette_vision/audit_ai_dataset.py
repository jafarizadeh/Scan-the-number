import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


def image_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    white_ratio = float(((v >= 145) & (s <= 130)).mean())
    green_ratio = float(((h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)).mean())
    red_ratio = float(((((h <= 12) | (h >= 165)) & (s >= 45) & (v >= 45))).mean())

    return {
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "sharpness": sharpness,
        "white_ratio": white_ratio,
        "green_ratio": green_ratio,
        "red_ratio": red_ratio,
    }


def make_sheet(items, output_path, tile_size=(120, 120), cols=5):
    if not items:
        return

    thumbs = []

    for path in items:
        img = cv2.imread(str(path))
        if img is None:
            continue

        thumb = cv2.resize(img, tile_size, interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)

    if not thumbs:
        return

    rows = int(np.ceil(len(thumbs) / cols))
    h, w = tile_size[1], tile_size[0]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def percentile(values, p):
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ai_dataset")
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-sheet", type=int, default=25)
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    logs = Path("data/logs")
    debug = Path("data/debug/ai_dataset_sheets")
    logs.mkdir(parents=True, exist_ok=True)
    debug.mkdir(parents=True, exist_ok=True)

    image_csv = logs / "ai_dataset_audit_images.csv"
    summary_csv = logs / "ai_dataset_audit_summary.csv"
    summary_json = logs / "ai_dataset_audit_summary.json"

    all_summary = []
    total = 0
    total_invalid = 0
    low_count = []

    with image_csv.open("w", newline="", encoding="utf-8") as f_img:
        img_writer = csv.writer(f_img)
        img_writer.writerow([
            "label", "path", "valid", "width", "height",
            "mean", "std", "sharpness",
            "white_ratio", "green_ratio", "red_ratio"
        ])

        for label in range(37):
            folder = root / str(label)
            paths = sorted(folder.glob("*.png")) if folder.exists() else []
            total += len(paths)

            metrics_list = []
            invalid = 0

            for path in paths:
                img = cv2.imread(str(path))

                if img is None:
                    invalid += 1
                    total_invalid += 1
                    img_writer.writerow([label, str(path), False, "", "", "", "", "", "", "", ""])
                    continue

                m = image_metrics(img)
                metrics_list.append(m)

                img_writer.writerow([
                    label,
                    str(path),
                    True,
                    m["width"],
                    m["height"],
                    round(m["mean"], 3),
                    round(m["std"], 3),
                    round(m["sharpness"], 3),
                    round(m["white_ratio"], 5),
                    round(m["green_ratio"], 5),
                    round(m["red_ratio"], 5),
                ])

            sharpness_values = [m["sharpness"] for m in metrics_list]
            mean_values = [m["mean"] for m in metrics_list]
            std_values = [m["std"] for m in metrics_list]
            white_values = [m["white_ratio"] for m in metrics_list]
            green_values = [m["green_ratio"] for m in metrics_list]

            count = len(paths)
            valid = len(metrics_list)

            if count < args.min_count:
                low_count.append((label, count))

            summary = {
                "label": label,
                "count": count,
                "valid": valid,
                "invalid": invalid,
                "sharpness_p10": percentile(sharpness_values, 10),
                "sharpness_median": percentile(sharpness_values, 50),
                "sharpness_p90": percentile(sharpness_values, 90),
                "mean_median": percentile(mean_values, 50),
                "std_median": percentile(std_values, 50),
                "white_ratio_median": percentile(white_values, 50),
                "green_ratio_median": percentile(green_values, 50),
            }

            all_summary.append(summary)

            sample_paths = paths
            if len(sample_paths) > args.samples_per_sheet:
                sample_paths = random.sample(sample_paths, args.samples_per_sheet)

            make_sheet(
                sample_paths,
                debug / f"label_{label:02d}_sheet.png",
                tile_size=(120, 120),
                cols=5,
            )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "count", "valid", "invalid",
            "sharpness_p10", "sharpness_median", "sharpness_p90",
            "mean_median", "std_median",
            "white_ratio_median", "green_ratio_median"
        ])
        writer.writeheader()
        writer.writerows(all_summary)

    summary_json.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

    print("")
    print("AI DATASET AUDIT")
    print("-" * 40)

    for s in all_summary:
        label = s["label"]
        count = s["count"]
        invalid = s["invalid"]
        sharp = s["sharpness_median"]

        status = "OK"

        if count < args.min_count:
            status = "LOW_COUNT"

        if invalid > 0:
            status = "INVALID_IMAGES"

        print(
            f"{label:2d}: count={count:4d} valid={s['valid']:4d} "
            f"invalid={invalid:2d} sharp_med={sharp if sharp is not None else 0:.1f} "
            f"{status}"
        )

    print("-" * 40)
    print("TOTAL:", total)
    print("INVALID:", total_invalid)
    print("")
    print(f"Image-level CSV: {image_csv}")
    print(f"Summary CSV:     {summary_csv}")
    print(f"Summary JSON:    {summary_json}")
    print(f"Contact sheets:  {debug}/label_XX_sheet.png")

    if low_count:
        print("")
        print("Labels with fewer than required samples:")
        for label, count in low_count:
            print(f"  {label}: {count}/{args.min_count}")

    if total_invalid > 0:
        print("")
        print("WARNING: Some images are unreadable. Remove or recollect them.")

    print("")
    print("Manual check required:")
    print("Open the contact sheets and verify that each label folder really contains that number.")


if __name__ == "__main__":
    main()
