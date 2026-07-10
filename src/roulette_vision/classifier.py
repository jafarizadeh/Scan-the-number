from pathlib import Path
from collections import Counter

import cv2
import numpy as np


def _l2_normalize(v):
    v = v.astype(np.float32).reshape(-1)
    v = v - float(v.mean())
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def _letterbox(gray, size):
    target_w, target_h = size
    h, w = gray.shape[:2]

    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    return canvas


def _extract_inner(gray):
    h, w = gray.shape[:2]

    x1 = int(0.08 * w)
    x2 = int(0.92 * w)
    y1 = int(0.10 * h)
    y2 = int(0.90 * h)

    inner = gray[y1:y2, x1:x2]

    if inner.size == 0:
        return gray

    return inner


def _digit_mask(inner):
    blur = cv2.GaussianBlur(inner, (3, 3), 0)

    # White digits are much brighter than the box background.
    threshold_value = max(125, int(np.percentile(blur, 82)))
    mask = (blur >= threshold_value).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _crop_digit_area(inner, mask):
    ys, xs = np.where(mask > 0)

    if len(xs) < 10 or len(ys) < 10:
        return inner, mask

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    pad_x = max(5, int(0.22 * bw))
    pad_y = max(5, int(0.22 * bh))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(inner.shape[1] - 1, x2 + pad_x)
    y2 = min(inner.shape[0] - 1, y2 + pad_y)

    return inner[y1:y2 + 1, x1:x2 + 1], mask[y1:y2 + 1, x1:x2 + 1]


def _projection_features(mask):
    m = (mask > 0).astype(np.float32)

    if m.size == 0:
        return np.zeros(1, dtype=np.float32)

    vertical = m.mean(axis=0)
    horizontal = m.mean(axis=1)

    return np.concatenate([vertical, horizontal]).astype(np.float32)


def extract_number_feature(roi_bgr, size=(72, 54)):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    inner = _extract_inner(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    inner_eq = clahe.apply(inner)

    mask = _digit_mask(inner_eq)
    digit_gray, digit_mask = _crop_digit_area(inner_eq, mask)

    norm_mask = _letterbox(digit_mask, size)
    norm_gray = _letterbox(digit_gray, size)

    # Clean binary shape.
    _, norm_binary = cv2.threshold(
        norm_mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Edge/shape signal.
    edges = cv2.Canny(norm_gray, 60, 140)

    proj = _projection_features(norm_binary)
    proj = cv2.resize(
        proj.reshape(1, -1),
        (size[0] + size[1], 1),
        interpolation=cv2.INTER_AREA
    ).reshape(-1)

    feature = np.concatenate([
        norm_binary.reshape(-1).astype(np.float32) / 255.0,
        edges.reshape(-1).astype(np.float32) / 255.0,
        proj.astype(np.float32)
    ])

    return _l2_normalize(feature)


class NumberClassifier:
    def __init__(self, template_root="data/templates", size=(72, 54), top_k=40):
        self.template_root = Path(template_root)
        self.size = size
        self.top_k = int(top_k)

        self.features = None
        self.labels = None
        self.paths = []
        self.class_centroids = {}

        self.load()

    def load(self):
        features = []
        labels = []
        paths = []

        for label in range(37):
            folder = self.template_root / str(label)

            if not folder.exists():
                continue

            for path in sorted(folder.glob("*.png")):
                img = cv2.imread(str(path))

                if img is None:
                    continue

                feat = extract_number_feature(img, self.size)

                features.append(feat)
                labels.append(label)
                paths.append(str(path))

        if not features:
            self.features = None
            self.labels = None
            self.paths = []
            return

        self.features = np.stack(features).astype(np.float32)
        self.labels = np.array(labels, dtype=np.int32)
        self.paths = paths

        self.class_centroids = {}

        for label in range(37):
            idx = np.where(self.labels == label)[0]

            if len(idx) == 0:
                continue

            centroid = self.features[idx].mean(axis=0)
            centroid = _l2_normalize(centroid)
            self.class_centroids[label] = centroid

    def is_ready(self):
        return self.features is not None and len(self.features) > 0

    def counts(self):
        result = {}

        if self.labels is None:
            return {str(i): 0 for i in range(37)}

        for i in range(37):
            result[str(i)] = int((self.labels == i).sum())

        return result

    def predict(self, roi_bgr):
        if not self.is_ready():
            return None

        feat = extract_number_feature(roi_bgr, self.size)

        sample_scores = self.features @ feat

        top_idx = np.argsort(sample_scores)[::-1][:self.top_k]
        top_labels = self.labels[top_idx]
        top_scores = sample_scores[top_idx]

        counts = Counter(int(x) for x in top_labels)

        best_sample_by_label = {}
        for idx in top_idx:
            label = int(self.labels[idx])
            score = float(sample_scores[idx])

            if label not in best_sample_by_label or score > best_sample_by_label[label]["score"]:
                best_sample_by_label[label] = {
                    "label": label,
                    "score": score,
                    "path": self.paths[idx]
                }

        centroid_scores = {}
        for label, centroid in self.class_centroids.items():
            centroid_scores[label] = float(np.dot(feat, centroid))

        combined = {}

        for label in range(37):
            best_score = best_sample_by_label.get(label, {}).get("score", -1.0)
            centroid_score = centroid_scores.get(label, -1.0)
            vote_ratio = counts.get(label, 0) / max(1, len(top_idx))

            combined[label] = (
                0.55 * best_score +
                0.30 * centroid_score +
                0.15 * vote_ratio
            )

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        best_label, best_score = ranked[0]
        second_label, second_score = ranked[1]

        topk_count = counts.get(best_label, 0)
        topk_agreement = topk_count / max(1, len(top_idx))

        return {
            "label": int(best_label),
            "score": float(best_score),
            "margin": float(best_score - second_score),
            "second_label": int(second_label),
            "second_score": float(second_score),
            "top_k": int(len(top_idx)),
            "topk_count": int(topk_count),
            "topk_agreement": float(topk_agreement),
            "topk_counts": dict(counts.most_common(5)),
            "best_template_path": best_sample_by_label.get(best_label, {}).get("path"),
            "centroid_score": float(centroid_scores.get(best_label, -1.0)),
            "sample_score": float(best_sample_by_label.get(best_label, {}).get("score", -1.0)),
            "source": "number_classifier"
        }
