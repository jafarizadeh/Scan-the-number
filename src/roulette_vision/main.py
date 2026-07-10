import argparse
import time
from pathlib import Path

import cv2

from .camera import FrameSource
from .config import load_config, ensure_project_dirs
from .engine import RouletteDetectionEngine
from .vision import crop_relative_roi, roi_metrics


def open_source(cfg, args):
    source = args.source if hasattr(args, "source") else None
    camera = FrameSource(cfg["camera"], source=source)
    camera.start()
    return camera


def cmd_camera_test(args):
    cfg = load_config(args.config)
    ensure_project_dirs()

    camera = open_source(cfg, args)

    try:
        frame = camera.read()
        if frame is None:
            raise RuntimeError("No frame captured")

        roi = crop_relative_roi(frame, cfg["roi"])

        cv2.imwrite("data/debug/frame_test.png", frame)
        cv2.imwrite("data/debug/roi_test.png", roi)

        print("Saved:")
        print("  data/debug/frame_test.png")
        print("  data/debug/roi_test.png")
        print("ROI metrics:", roi_metrics(roi))

    finally:
        camera.close()


def cmd_save_roi(args):
    cfg = load_config(args.config)
    ensure_project_dirs()

    camera = open_source(cfg, args)

    try:
        for i in range(args.count):
            frame = camera.read()
            if frame is None:
                break

            roi = crop_relative_roi(frame, cfg["roi"])
            path = Path("data/debug") / f"roi_sample_{i:04d}.png"
            cv2.imwrite(str(path), roi)

            print(f"saved {path} metrics={roi_metrics(roi)}")
            time.sleep(args.interval)

    finally:
        camera.close()


def cmd_collect_template(args):
    cfg = load_config(args.config)
    ensure_project_dirs()

    label = str(args.label)
    out_dir = Path("data/templates") / label
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = open_source(cfg, args)

    deadline = time.time() + args.seconds
    saved = 0

    try:
        while time.time() < deadline and saved < args.max_images:
            frame = camera.read()
            if frame is None:
                continue

            roi = crop_relative_roi(frame, cfg["roi"])
            metrics = roi_metrics(roi)

            if metrics["white_ratio"] >= cfg["state"]["bright_white_ratio_min"]:
                path = out_dir / f"{label}_{int(time.time() * 1000)}_{saved:04d}.png"
                cv2.imwrite(str(path), roi)
                saved += 1
                print(f"saved template {path} metrics={metrics}")

            time.sleep(args.interval)

    finally:
        camera.close()

    print(f"Done. Saved {saved} templates for label {label}")


def cmd_collect_empty(args):
    cfg = load_config(args.config)
    ensure_project_dirs()

    out_dir = Path("data/empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = open_source(cfg, args)

    deadline = time.time() + args.seconds
    saved = 0

    try:
        while time.time() < deadline and saved < args.max_images:
            frame = camera.read()
            if frame is None:
                continue

            roi = crop_relative_roi(frame, cfg["roi"])
            path = out_dir / f"empty_{int(time.time() * 1000)}_{saved:04d}.png"
            cv2.imwrite(str(path), roi)
            saved += 1
            print(f"saved empty {path} metrics={roi_metrics(roi)}")
            time.sleep(args.interval)

    finally:
        camera.close()

    print(f"Done. Saved {saved} empty samples")


def cmd_run(args):
    cfg = load_config(args.config)
    ensure_project_dirs()

    engine = RouletteDetectionEngine(cfg)

    if not engine.matcher.is_ready():
        print("ERROR: No number templates found in data/templates/")
        print("First collect templates, for example:")
        print("  python -m roulette_vision.main collect-template --label 27 --seconds 5")
        return

    camera = open_source(cfg, args)

    print("Live detector started.")
    print("Important: it will log a number only after EMPTY -> BRIGHT -> stable votes.")
    print("Press Ctrl+C to stop.")

    start_time = time.time()
    last_print = 0.0

    try:
        while True:
            if args.duration > 0 and time.time() - start_time > args.duration:
                break

            frame = camera.read()
            if frame is None:
                continue

            roi = crop_relative_roi(frame, cfg["roi"])
            result = engine.process(frame, roi)

            now = time.time()

            if now - last_print >= args.print_interval:
                pred = result.get("prediction")
                pred_text = ""
                if pred:
                    pred_text = (
                        f" pred={pred['label']}"
                        f" score={pred['score']:.3f}"
                        f" margin={pred['margin']:.3f}"
                    )

                empty_score = result.get("empty_score")
                empty_text = ""
                if empty_score is not None:
                    empty_text = f" empty_score={empty_score:.3f}"

                m = result["metrics"]
                print(
                    f"state={result['state']}"
                    f" armed={result['armed']}"
                    f" white={m['white_ratio']:.4f}"
                    f" std={m['std']:.1f}"
                    f"{empty_text}"
                    f"{pred_text}"
                )
                last_print = now

            event = result.get("event")
            if event:
                print("")
                print("CONFIRMED RESULT")
                print(f"  round: {event['round_index']}")
                print(f"  number: {event['number']}")
                print(f"  agreement: {event['agreement']:.3f}")
                print(f"  score: {event['score']:.3f}")
                print(f"  margin: {event['margin']:.3f}")
                print(f"  frame: {event['frame_path']}")
                print(f"  roi: {event['roi_path']}")
                print("")

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera.close()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("camera-test")
    p.add_argument("--source", default=None)
    p.set_defaults(func=cmd_camera_test)

    p = sub.add_parser("save-roi")
    p.add_argument("--source", default=None)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--interval", type=float, default=0.2)
    p.set_defaults(func=cmd_save_roi)

    p = sub.add_parser("collect-template")
    p.add_argument("--source", default=None)
    p.add_argument("--label", type=int, required=True)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--interval", type=float, default=0.15)
    p.add_argument("--max-images", type=int, default=40)
    p.set_defaults(func=cmd_collect_template)

    p = sub.add_parser("collect-empty")
    p.add_argument("--source", default=None)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--interval", type=float, default=0.10)
    p.add_argument("--max-images", type=int, default=30)
    p.set_defaults(func=cmd_collect_empty)

    p = sub.add_parser("run")
    p.add_argument("--source", default=None)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--print-interval", type=float, default=0.5)
    p.set_defaults(func=cmd_run)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
