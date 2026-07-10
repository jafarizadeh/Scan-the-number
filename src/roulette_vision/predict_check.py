import argparse
import time
from collections import Counter

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .engine import RouletteDetectionEngine
from .vision import crop_relative_roi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--print-interval", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_project_dirs()

    engine = RouletteDetectionEngine(cfg)

    if not engine.matcher.is_ready():
        print("ERROR: no templates found.")
        return

    print("Loaded labels:", engine.matcher.labels())
    print("Starting prediction check. This will NOT log results.")
    print("Keep the current number visible in the Result ROI.")
    print("")

    camera = FrameSource(cfg["camera"])
    camera.start()

    votes = []
    state_votes = Counter()
    last_print = 0.0
    start = time.time()

    try:
        while time.time() - start < args.duration:
            frame = camera.read()
            if frame is None:
                continue

            roi = crop_relative_roi(frame, cfg["roi"])
            state, metrics, empty_score = engine.classify_roi_state(roi)
            pred = engine.matcher.predict(roi) if state == "BRIGHT" else None

            state_votes[state] += 1

            if pred is not None and state == "BRIGHT":
                votes.append(pred["label"])

            now = time.time()
            if now - last_print >= args.print_interval:
                if pred:
                    print(
                        f"state={state:<6} "
                        f"pred={pred['label']:>2} "
                        f"score={pred['score']:.3f} "
                        f"margin={pred['margin']:.3f} "
                        f"second={str(pred.get('second_label')):>2} "
                        f"white={metrics['white_ratio']:.4f} "
                        f"sharp={metrics.get('sharpness', 0):.1f}"
                    )
                else:
                    print(
                        f"state={state:<6} "
                        f"pred=None "
                        f"white={metrics['white_ratio']:.4f} "
                        f"sharp={metrics.get('sharpness', 0):.1f}"
                    )
                last_print = now

    finally:
        camera.close()

    print("")
    print("State summary:")
    print(state_votes)

    print("")
    print("Prediction summary for BRIGHT frames:")
    print(Counter(votes))

    if votes:
        top_label, top_count = Counter(votes).most_common(1)[0]
        agreement = top_count / len(votes)
        print("")
        print(f"Top prediction: {top_label}")
        print(f"Agreement: {agreement:.3f}")
        print(f"Usable BRIGHT votes: {len(votes)}")


if __name__ == "__main__":
    main()
