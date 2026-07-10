import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2

from .config import load_config
from .vision import (
    normalize_for_match,
    normalize_digit_mask,
    ncc_score,
    roi_metrics,
)


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


def load_samples(template_root, size):
    samples = []

    for label in range(37):
        label_dir = Path(template_root) / str(label)
        if not label_dir.exists():
            continue

        for img_path in sorted(label_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            samples.append({
                "label": label,
                "path": str(img_path),
                "gray": normalize_for_match(img, size),
                "digit": normalize_digit_mask(img, size),
                "metrics": roi_metrics(img),
            })

    return samples


def predict_against_others(sample, samples):
    best_by_label = {}

    for other in samples:
        if other["path"] == sample["path"]:
            continue

        gray_score = ncc_score(sample["gray"], other["gray"])
        digit_score = ncc_score(sample["digit"], other["digit"])
        score = 0.35 * gray_score + 0.65 * digit_score

        label = other["label"]

        item = {
            "label": label,
            "score": float(score),
            "gray_score": float(gray_score),
            "digit_score": float(digit_score),
            "path": other["path"],
        }

        if label not in best_by_label or item["score"] > best_by_label[label]["score"]:
            best_by_label[label] = item

    ranked = sorted(best_by_label.values(), key=lambda x: x["score"], reverse=True)

    if not ranked:
        return None

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None

    return {
        "predicted_label": best["label"],
        "score": best["score"],
        "gray_score": best["gray_score"],
        "digit_score": best["digit_score"],
        "matched_template": best["path"],
        "second_label": second["label"] if second else None,
        "second_score": second["score"] if second else -1.0,
        "margin": best["score"] - (second["score"] if second else -1.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--template-root", default="data/templates")
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)

    size = (
        int(cfg["matching"]["match_size_w"]),
        int(cfg["matching"]["match_size_h"]),
    )

    score_min = float(cfg["matching"]["score_min"])
    margin_min = float(cfg["matching"]["margin_min"])

    samples = load_samples(args.template_root, size)

    counts = {str(n): 0 for n in range(37)}
    sharpness_by_label = {str(n): [] for n in range(37)}
    edge_by_label = {str(n): [] for n in range(37)}

    for s in samples:
        key = str(s["label"])
        counts[key] += 1
        sharpness_by_label[key].append(s["metrics"].get("sharpness", 0.0))
        edge_by_label[key].append(s["metrics"].get("digit_edge_sharpness", 0.0))

    suspicious = []

    for s in samples:
        pred = predict_against_others(s, samples)

        if pred is None:
            suspicious.append({
                "reason": "no_prediction",
                "label": s["label"],
                "path": s["path"],
            })
            continue

        reasons = []

        if pred["predicted_label"] != s["label"]:
            reasons.append("wrong_label_match")

        if pred["score"] < score_min:
            reasons.append("low_score")

        if pred["margin"] < margin_min:
            reasons.append("low_margin")

        if reasons:
            suspicious.append({
                "reason": ",".join(reasons),
                "label": s["label"],
                "predicted_label": pred["predicted_label"],
                "score": round(pred["score"], 4),
                "margin": round(pred["margin"], 4),
                "second_label": pred["second_label"],
                "second_score": round(pred["second_score"], 4),
                "path": s["path"],
                "matched_template": pred["matched_template"],
            })

    low_count = {
        label: count
        for label, count in counts.items()
        if count < args.min_count
    }

    summary = {
        "total_samples": len(samples),
        "counts": counts,
        "low_count_labels": low_count,
        "suspicious_count": len(suspicious),
        "median_sharpness_by_label": {
            k: median(v) for k, v in sharpness_by_label.items()
        },
        "median_digit_edge_sharpness_by_label": {
            k: median(v) for k, v in edge_by_label.items()
        },
    }

    out_dir = Path("data/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "template_audit_report.json"
    suspicious_path = out_dir / "template_audit_suspicious.csv"

    report_path.write_text(json.dumps({
        "summary": summary,
        "suspicious": suspicious,
    }, indent=2), encoding="utf-8")

    with suspicious_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "reason",
            "label",
            "predicted_label",
            "score",
            "margin",
            "second_label",
            "second_score",
            "path",
            "matched_template",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in suspicious:
            writer.writerow(row)

    print("")
    print("TEMPLATE AUDIT SUMMARY")
    print(json.dumps(summary, indent=2))
    print("")
    print(f"Report saved: {report_path}")
    print(f"Suspicious list saved: {suspicious_path}")

    if suspicious:
        print("")
        print("WARNING: suspicious templates found.")
        print("Check data/debug/template_audit_suspicious.csv")
    else:
        print("")
        print("OK: no suspicious template found.")


if __name__ == "__main__":
    main()
