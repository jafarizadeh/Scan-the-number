import argparse
import time
from collections import Counter

from .camera import FrameSource
from .config import load_config
from .vision import crop_relative_roi
from .digit_recognizer import DigitRecognizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--print-interval", type=float, default=0.25)
    args = parser.parse_args()

    cfg = load_config(args.config)

    recognizer = DigitRecognizer("data/templates")

    print("Digit template counts:")
    print(recognizer.counts())

    if not recognizer.is_ready():
        print("ERROR: digit recognizer is not ready.")
        return

    camera = FrameSource(cfg["camera"])
    camera.start()

    votes = []
    last_print = 0.0
    start = time.time()

    try:
        while time.time() - start < args.duration:
            frame = camera.read()
            if frame is None:
                continue

            roi = crop_relative_roi(frame, cfg["roi"])
            pred = recognizer.predict(roi)

            if pred:
                votes.append(pred["label"])

            now = time.time()
            if now - last_print >= args.print_interval:
                if pred:
                    print(
                        f"digit_pred={pred['label']:>2} "
                        f"digits={pred.get('digits')} "
                        f"score={pred['score']:.3f} "
                        f"margin={pred['margin']:.3f} "
                        f"second={pred.get('second_label')}"
                    )
                else:
                    print("digit_pred=None")

                last_print = now

    finally:
        camera.close()

    print("")
    print("Digit prediction summary:")
    print(Counter(votes))


if __name__ == "__main__":
    main()
