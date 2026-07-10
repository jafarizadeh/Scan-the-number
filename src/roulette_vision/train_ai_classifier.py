import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from .ai_features import extract_ai_feature


def collect_paths(roots):
    by_label = defaultdict(list)

    for root in roots:
        root = Path(root)

        if not root.exists():
            continue

        for label in range(37):
            folder = root / str(label)

            if not folder.exists():
                continue

            for path in sorted(folder.glob("*.png")):
                by_label[label].append(path)

    return by_label


def one_hot(labels, n_classes=37):
    y = -np.ones((len(labels), n_classes), dtype=np.float32)

    for i, label in enumerate(labels):
        y[i, int(label)] = 1.0

    return y


def build_split(by_label, val_ratio=0.20, seed=42):
    rng = np.random.default_rng(seed)
    train = []
    val = []

    for label in range(37):
        paths = list(by_label.get(label, []))
        rng.shuffle(paths)

        if len(paths) == 0:
            continue

        n_val = max(1, int(len(paths) * val_ratio)) if len(paths) >= 5 else 0

        val_paths = paths[:n_val]
        train_paths = paths[n_val:]

        for p in train_paths:
            train.append((label, p))

        for p in val_paths:
            val.append((label, p))

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


def load_features(items, augmentations=0, seed=42):
    rng = np.random.default_rng(seed)

    x = []
    y = []

    for label, path in items:
        img = cv2.imread(str(path))

        if img is None:
            continue

        x.append(extract_ai_feature(img, augment=False))
        y.append(label)

        for _ in range(augmentations):
            x.append(extract_ai_feature(img, augment=True, rng=rng))
            y.append(label)

    return np.stack(x).astype(np.float32), np.array(y, dtype=np.int32)


def evaluate(ann, x, y):
    if len(y) == 0:
        return None

    _, out = ann.predict(x)
    pred = np.argmax(out, axis=1).astype(np.int32)

    acc = float((pred == y).mean())

    cm_errors = []

    for i in range(len(y)):
        if pred[i] != y[i]:
            cm_errors.append((int(y[i]), int(pred[i])))

    return acc, cm_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ai_dataset")
    parser.add_argument("--include-templates", action="store_true")
    parser.add_argument("--templates", default="data/templates")
    parser.add_argument("--model", default="data/models/roulette_ann_mlp.xml")
    parser.add_argument("--stats", default="data/models/roulette_ann_stats.npz")
    parser.add_argument("--meta", default="data/models/roulette_ann_meta.json")
    parser.add_argument("--augmentations", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    roots = [args.dataset]

    if args.include_templates:
        roots.append(args.templates)

    by_label = collect_paths(roots)

    print("Sample counts:")
    missing = []

    for label in range(37):
        n = len(by_label.get(label, []))
        print(f"  {label:2d}: {n}")
        if n == 0:
            missing.append(label)

    if missing:
        raise SystemExit(f"Missing classes: {missing}")

    train_items, val_items = build_split(by_label, seed=args.seed)

    print("")
    print(f"Train originals: {len(train_items)}")
    print(f"Val originals:   {len(val_items)}")
    print(f"Augmentations per train image: {args.augmentations}")

    x_train, y_train = load_features(
        train_items,
        augmentations=args.augmentations,
        seed=args.seed
    )

    x_val, y_val = load_features(
        val_items,
        augmentations=0,
        seed=args.seed + 1
    )

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0

    x_train = ((x_train - mean) / std).astype(np.float32)
    x_val = ((x_val - mean) / std).astype(np.float32)

    y_train_oh = one_hot(y_train)

    input_dim = x_train.shape[1]

    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([input_dim, args.hidden, 37], dtype=np.int32))
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)
    ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, args.max_iter, 1e-5))

    print("")
    print("Training ANN/MLP...")
    ok = ann.train(x_train, cv2.ml.ROW_SAMPLE, y_train_oh)

    if not ok:
        raise SystemExit("Training failed.")

    train_eval = evaluate(ann, x_train[:min(len(x_train), 3000)], y_train[:min(len(y_train), 3000)])
    val_eval = evaluate(ann, x_val, y_val)

    print("")
    if train_eval:
        print(f"Train subset accuracy: {train_eval[0]:.4f}")

    if val_eval:
        acc, errors = val_eval
        print(f"Validation accuracy:   {acc:.4f}")

        if errors:
            print("Validation errors, first 30:")
            for exp, pred in errors[:30]:
                print(f"  expected={exp} predicted={pred}")

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    ann.save(args.model)

    np.savez(
        args.stats,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        labels=np.arange(37, dtype=np.int32)
    )

    meta = {
        "model": args.model,
        "stats": args.stats,
        "classes": list(range(37)),
        "feature_dim": int(input_dim),
        "hidden": int(args.hidden),
        "augmentations": int(args.augmentations),
        "include_templates": bool(args.include_templates),
    }

    Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("")
    print(f"Model saved: {args.model}")
    print(f"Stats saved: {args.stats}")
    print(f"Meta saved:  {args.meta}")


if __name__ == "__main__":
    main()
