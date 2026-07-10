from pathlib import Path

import cv2
import numpy as np

from .ai_features import extract_ai_feature


def _softmax(x):
    x = x.astype(np.float32)
    x = x - float(np.max(x))
    e = np.exp(x)
    s = float(e.sum())

    if s <= 1e-9:
        return np.ones_like(x) / len(x)

    return e / s


class AINumberModel:
    def __init__(
        self,
        model_path="data/models/roulette_ann_mlp.xml",
        stats_path="data/models/roulette_ann_stats.npz"
    ):
        model_path = Path(model_path)
        stats_path = Path(stats_path)

        if not model_path.exists():
            raise FileNotFoundError(f"AI model not found: {model_path}")

        if not stats_path.exists():
            raise FileNotFoundError(f"AI stats not found: {stats_path}")

        self.ann = cv2.ml.ANN_MLP_load(str(model_path))

        stats = np.load(str(stats_path))
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)

    def predict(self, roi_bgr):
        feat = extract_ai_feature(roi_bgr, augment=False)

        if feat.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"Feature size mismatch: got {feat.shape[0]}, expected {self.mean.shape[0]}"
            )

        x = ((feat - self.mean) / self.std).reshape(1, -1).astype(np.float32)

        _, out = self.ann.predict(x)

        raw = out.reshape(-1).astype(np.float32)
        probs = _softmax(raw)

        ranked = np.argsort(probs)[::-1]

        top = int(ranked[0])
        second = int(ranked[1])

        return {
            "label": top,
            "score": float(probs[top]),
            "margin": float(probs[top] - probs[second]),
            "second_label": second,
            "second_score": float(probs[second]),
            "top5": [
                {
                    "label": int(i),
                    "score": float(probs[i])
                }
                for i in ranked[:5]
            ],
            "source": "ai_ann_mlp"
        }
