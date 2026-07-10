import argparse
import csv
from pathlib import Path
from datetime import datetime


def parse_int(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def load_expected(path):
    return [
        int(x.strip())
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


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
                    "value": parse_int(row.get("number")),
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
                    "value": parse_int(row.get("suggested_number")),
                    "score": row.get("score", ""),
                    "margin": row.get("margin", ""),
                    "reason": row.get("reason", ""),
                    "roi_path": row.get("roi_path", ""),
                    "frame_path": row.get("frame_path", ""),
                })

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def align_sequences(expected, outputs):
    """
    Edit-distance alignment:
    - MATCH: expected[i] == output[j]
    - SUBSTITUTE: expected[i] != output[j]
    - DELETE_EXPECTED: expected item missing from system output
    - INSERT_OUTPUT: extra system output
    """
    n = len(expected)
    m = len(outputs)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "DELETE_EXPECTED"

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "INSERT_OUTPUT"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            exp = expected[i - 1]
            out = outputs[j - 1]["value"]

            if exp == out:
                cost_diag = dp[i - 1][j - 1]
                op_diag = "MATCH"
            else:
                cost_diag = dp[i - 1][j - 1] + 2
                op_diag = "SUBSTITUTE"

            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1

            best = min(cost_diag, cost_del, cost_ins)

            dp[i][j] = best

            if best == cost_diag:
                back[i][j] = op_diag
            elif best == cost_del:
                back[i][j] = "DELETE_EXPECTED"
            else:
                back[i][j] = "INSERT_OUTPUT"

    aligned = []
    i, j = n, m

    while i > 0 or j > 0:
        op = back[i][j]

        if op in ("MATCH", "SUBSTITUTE"):
            aligned.append({
                "op": op,
                "expected_index": i,
                "output_index": j,
                "expected": expected[i - 1],
                "output_type": outputs[j - 1]["type"],
                "output": outputs[j - 1]["value"],
                "score": outputs[j - 1]["score"],
                "margin": outputs[j - 1]["margin"],
                "reason": outputs[j - 1]["reason"],
                "roi_path": outputs[j - 1]["roi_path"],
                "frame_path": outputs[j - 1]["frame_path"],
            })
            i -= 1
            j -= 1

        elif op == "DELETE_EXPECTED":
            aligned.append({
                "op": "DELETE_EXPECTED",
                "expected_index": i,
                "output_index": "",
                "expected": expected[i - 1],
                "output_type": "MISSING",
                "output": "",
                "score": "",
                "margin": "",
                "reason": "",
                "roi_path": "",
                "frame_path": "",
            })
            i -= 1

        elif op == "INSERT_OUTPUT":
            aligned.append({
                "op": "INSERT_OUTPUT",
                "expected_index": "",
                "output_index": j,
                "expected": "",
                "output_type": outputs[j - 1]["type"],
                "output": outputs[j - 1]["value"],
                "score": outputs[j - 1]["score"],
                "margin": outputs[j - 1]["margin"],
                "reason": outputs[j - 1]["reason"],
                "roi_path": outputs[j - 1]["roi_path"],
                "frame_path": outputs[j - 1]["frame_path"],
            })
            j -= 1

        else:
            raise RuntimeError(f"Unexpected op at i={i}, j={j}: {op}")

    aligned.reverse()
    return aligned, dp[n][m]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()

    expected = load_expected(args.expected)
    outputs = load_outputs(args.events, args.review)

    aligned, distance = align_sequences(expected, outputs)

    counts = {
        "MATCH": 0,
        "SUBSTITUTE": 0,
        "DELETE_EXPECTED": 0,
        "INSERT_OUTPUT": 0,
    }

    confirmed_matches = 0
    confirmed_substitutions = 0
    review_matches = 0
    review_substitutions = 0

    for row in aligned:
        counts[row["op"]] += 1

        if row["op"] == "MATCH":
            if row["output_type"] == "CONFIRMED":
                confirmed_matches += 1
            elif row["output_type"] == "NEEDS_REVIEW":
                review_matches += 1

        elif row["op"] == "SUBSTITUTE":
            if row["output_type"] == "CONFIRMED":
                confirmed_substitutions += 1
            elif row["output_type"] == "NEEDS_REVIEW":
                review_substitutions += 1

    out_dir = Path("data/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = out_dir / f"real_test_aligned_eval_{ts}.csv"

    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "op",
            "expected_index",
            "output_index",
            "expected",
            "output_type",
            "output",
            "score",
            "margin",
            "reason",
            "roi_path",
            "frame_path",
        ])
        writer.writeheader()
        writer.writerows(aligned)

    print("")
    print("ALIGNED REAL TEST EVALUATION")
    print(f"Expected count: {len(expected)}")
    print(f"System outputs:  {len(outputs)}")
    print(f"Edit distance:    {distance}")
    print("")
    print(f"Matches:          {counts['MATCH']}")
    print(f"Substitutions:    {counts['SUBSTITUTE']}")
    print(f"Missing expected: {counts['DELETE_EXPECTED']}")
    print(f"Extra outputs:    {counts['INSERT_OUTPUT']}")
    print("")
    print(f"Confirmed correct after alignment: {confirmed_matches}")
    print(f"Confirmed wrong after alignment:   {confirmed_substitutions}")
    print(f"Review correct after alignment:    {review_matches}")
    print(f"Review wrong after alignment:      {review_substitutions}")
    print("")
    print(f"Report saved: {report}")

    print("")
    print("First non-match rows:")
    shown = 0
    for row in aligned:
        if row["op"] != "MATCH":
            print(
                f"  op={row['op']} "
                f"expected_index={row['expected_index']} expected={row['expected']} "
                f"output_index={row['output_index']} output={row['output']} "
                f"type={row['output_type']} reason={row['reason']}"
            )
            shown += 1
            if shown >= 20:
                break

    if shown == 0:
        print("  None")


if __name__ == "__main__":
    main()
