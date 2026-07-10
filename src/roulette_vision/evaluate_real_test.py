import argparse
import csv
from pathlib import Path
from datetime import datetime


def parse_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def load_expected(path):
    values = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            values.append(int(line))
    return values


def load_outputs(events_path, review_path):
    rows = []

    events_path = Path(events_path)
    review_path = Path(review_path)

    if events_path.exists():
        with events_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "timestamp": row.get("timestamp", ""),
                    "type": "CONFIRMED",
                    "detected": parse_int(row.get("number")),
                    "score": row.get("score", ""),
                    "margin": row.get("margin", ""),
                    "reason": row.get("reason", ""),
                    "roi_path": row.get("roi_path", ""),
                    "frame_path": row.get("frame_path", ""),
                })

    if review_path.exists():
        with review_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "timestamp": row.get("timestamp", ""),
                    "type": "NEEDS_REVIEW",
                    "detected": parse_int(row.get("suggested_number")),
                    "score": row.get("score", ""),
                    "margin": row.get("margin", ""),
                    "reason": row.get("reason", ""),
                    "roi_path": row.get("roi_path", ""),
                    "frame_path": row.get("frame_path", ""),
                })

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", default="data/test_expected_40.txt")
    parser.add_argument("--events", default="data/logs/digit_v3_events.csv")
    parser.add_argument("--review", default="data/logs/digit_v3_review.csv")
    args = parser.parse_args()

    expected = load_expected(args.expected)
    outputs = load_outputs(args.events, args.review)

    out_dir = Path("data/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"real_test_eval_{ts}.csv"

    max_len = max(len(expected), len(outputs))

    confirmed_correct = 0
    confirmed_wrong = 0
    review_correct_suggestion = 0
    review_wrong_suggestion = 0
    missing = 0
    extra = 0

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "expected",
            "output_type",
            "detected_or_suggested",
            "correct",
            "score",
            "margin",
            "reason",
            "roi_path",
            "frame_path"
        ])

        for i in range(max_len):
            exp = expected[i] if i < len(expected) else None
            out = outputs[i] if i < len(outputs) else None

            if exp is None and out is not None:
                extra += 1
                writer.writerow([
                    i + 1,
                    "",
                    out["type"],
                    out["detected"],
                    "EXTRA",
                    out["score"],
                    out["margin"],
                    out["reason"],
                    out["roi_path"],
                    out["frame_path"],
                ])
                continue

            if exp is not None and out is None:
                missing += 1
                writer.writerow([
                    i + 1,
                    exp,
                    "MISSING",
                    "",
                    "NO_OUTPUT",
                    "",
                    "",
                    "",
                    "",
                    "",
                ])
                continue

            detected = out["detected"]
            correct = detected == exp

            if out["type"] == "CONFIRMED":
                if correct:
                    confirmed_correct += 1
                else:
                    confirmed_wrong += 1

            elif out["type"] == "NEEDS_REVIEW":
                if correct:
                    review_correct_suggestion += 1
                else:
                    review_wrong_suggestion += 1

            writer.writerow([
                i + 1,
                exp,
                out["type"],
                detected,
                "OK" if correct else "WRONG",
                out["score"],
                out["margin"],
                out["reason"],
                out["roi_path"],
                out["frame_path"],
            ])

    print("")
    print("REAL TEST EVALUATION")
    print(f"Expected count: {len(expected)}")
    print(f"System outputs:  {len(outputs)}")
    print("")
    print(f"Confirmed correct:        {confirmed_correct}")
    print(f"Confirmed wrong:          {confirmed_wrong}")
    print(f"Review correct suggestion:{review_correct_suggestion}")
    print(f"Review wrong suggestion:  {review_wrong_suggestion}")
    print(f"Missing outputs:          {missing}")
    print(f"Extra outputs:            {extra}")
    print("")
    print(f"Report saved: {report_path}")

    if confirmed_wrong > 0:
        print("")
        print("WARNING: There are wrong confirmed results. These are the most important errors.")

    if missing > 0:
        print("")
        print("WARNING: Some expected numbers produced no confirmed/review output.")

    if review_correct_suggestion or review_wrong_suggestion:
        print("")
        print("Needs Review items are not counted as automatic success. They are saved for manual correction.")


if __name__ == "__main__":
    main()
