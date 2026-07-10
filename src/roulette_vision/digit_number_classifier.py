from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def _normalize(v):
    v = v.reshape(-1).astype(np.float32)
    v = v - float(v.mean())
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


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


def _inner_bgr(roi_bgr):
    h, w = roi_bgr.shape[:2]

    x1 = int(0.08 * w)
    x2 = int(0.92 * w)
    y1 = int(0.10 * h)
    y2 = int(0.90 * h)

    inner = roi_bgr[y1:y2, x1:x2]

    if inner.size == 0:
        return roi_bgr

    return inner


def _prepare_inner(roi_bgr):
    """
    Color-aware digit extraction.

    Important for 0:
    The result background can be green. A grayscale threshold may include the
    green background as part of the digit. HSV lets us keep bright/low-saturation
    white digit pixels and suppress saturated green background.
    """
    inner_bgr = _inner_bgr(roi_bgr)

    gray = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)

    hsv = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # White digits: high value, relatively low saturation.
    white_hsv = (
        (v >= max(145, int(np.percentile(v, 78)))) &
        (s <= 115)
    )

    # Backup grayscale mask for non-green/gray backgrounds.
    gray_thr = max(150, int(np.percentile(eq, 86)))
    white_gray = eq >= gray_thr

    # Suppress saturated green/blue/red background.
    saturated_bg = (s >= 135) & (v >= 80)

    mask = ((white_hsv | white_gray) & (~saturated_bg)).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return eq, mask


def _bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) < 6 or len(ys) < 6:
        return None

    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _box_from_xrange(mask, x1, x2):
    sub = mask[:, x1:x2 + 1]
    ys, xs = np.where(sub > 0)

    if len(xs) < 4 or len(ys) < 4:
        return None

    return [
        x1 + int(xs.min()),
        int(ys.min()),
        x1 + int(xs.max()),
        int(ys.max())
    ]


def _single_box(mask):
    box = _bbox(mask)
    return [box] if box else []


def _split_two_boxes(mask):
    box = _bbox(mask)

    if box is None:
        return []

    x1, y1, x2, y2 = box
    bw = x2 - x1 + 1

    active = (mask[:, x1:x2 + 1] > 0).sum(axis=0).astype(np.float32)

    if len(active) < 10:
        mid = (x1 + x2) // 2
    else:
        smooth = cv2.GaussianBlur(active.reshape(1, -1), (1, 7), 0).reshape(-1)

        lo = int(0.34 * bw)
        hi = int(0.66 * bw)

        if hi <= lo:
            mid = (x1 + x2) // 2
        else:
            valley = lo + int(np.argmin(smooth[lo:hi]))
            mid = x1 + valley

    left = _box_from_xrange(mask, x1, mid)
    right = _box_from_xrange(mask, mid + 1, x2)

    boxes = []

    if left:
        boxes.append(left)

    if right:
        boxes.append(right)

    return boxes


def _pad_box(box, shape, px=0.22, py=0.22):
    x1, y1, x2, y2 = box
    h, w = shape[:2]

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    pad_x = max(3, int(px * bw))
    pad_y = max(3, int(py * bh))

    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w - 1, x2 + pad_x),
        min(h - 1, y2 + pad_y)
    ]


def _digit_feature(inner_eq, mask, box, size=(44, 64)):
    x1, y1, x2, y2 = _pad_box(box, inner_eq.shape)

    crop_gray = inner_eq[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

    if crop_gray.size == 0:
        return np.zeros(size[0] * size[1] * 2 + size[0] + size[1], dtype=np.float32)

    norm_mask = _letterbox(crop_mask, size)
    norm_gray = _letterbox(crop_gray, size)

    edges = cv2.Canny(norm_gray, 60, 140)

    m = (norm_mask > 0).astype(np.float32)
    vertical = m.mean(axis=0)
    horizontal = m.mean(axis=1)

    feature = np.concatenate([
        norm_mask.reshape(-1).astype(np.float32) / 255.0,
        edges.reshape(-1).astype(np.float32) / 255.0,
        vertical.astype(np.float32),
        horizontal.astype(np.float32)
    ])

    return _normalize(feature)


class DigitNumberClassifier:
    def __init__(self, template_root="data/templates", digit_size=(44, 64)):
        self.template_root = Path(template_root)
        self.digit_size = digit_size

        self.features = defaultdict(list)
        self.sources = defaultdict(list)
        self.centroids = {}

        self.load()

    def load(self):
        self.features.clear()
        self.sources.clear()
        self.centroids.clear()

        for number in range(37):
            folder = self.template_root / str(number)

            if not folder.exists():
                continue

            label = str(number)
            expected_digits = len(label)

            for path in sorted(folder.glob("*.png")):
                img = cv2.imread(str(path))

                if img is None:
                    continue

                inner_eq, mask = _prepare_inner(img)

                if expected_digits == 1:
                    boxes = _single_box(mask)
                else:
                    boxes = _split_two_boxes(mask)

                if len(boxes) != expected_digits:
                    continue

                for digit_char, box in zip(label, boxes):
                    digit = int(digit_char)
                    feat = _digit_feature(inner_eq, mask, box, self.digit_size)
                    self.features[digit].append(feat)
                    self.sources[digit].append(str(path))

        for digit, feats in self.features.items():
            if not feats:
                continue

            c = np.stack(feats).mean(axis=0)
            self.centroids[digit] = _normalize(c)

    def is_ready(self):
        return all(len(self.features[d]) > 0 for d in range(10))

    def counts(self):
        return {str(d): len(self.features[d]) for d in range(10)}

    def _classify_digit(self, feat, allowed=None):
        allowed = allowed or list(range(10))
        ranked = []

        for digit in allowed:
            feats = self.features.get(digit, [])

            if not feats:
                continue

            mat = np.stack(feats).astype(np.float32)
            sample_scores = mat @ feat
            best_idx = int(np.argmax(sample_scores))
            best_score = float(sample_scores[best_idx])

            centroid_score = float(np.dot(self.centroids[digit], feat)) if digit in self.centroids else -1.0
            score = 0.75 * best_score + 0.25 * centroid_score

            ranked.append({
                "digit": int(digit),
                "score": float(score),
                "sample_score": float(best_score),
                "centroid_score": float(centroid_score),
                "source": self.sources[digit][best_idx]
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)

        if not ranked:
            return None, []

        best = dict(ranked[0])
        second = ranked[1] if len(ranked) > 1 else None

        best["second_digit"] = second["digit"] if second else None
        best["second_score"] = second["score"] if second else -1.0
        best["margin"] = best["score"] - (second["score"] if second else -1.0)

        return best, ranked[:4]

    def _estimate_digit_count(self, mask):
        box = _bbox(mask)

        if box is None:
            return 0

        x1, y1, x2, y2 = box
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1

        ratio = bw / max(1, bh)

        # More conservative than before. This helps avoid reading 0 as two digits.
        return 2 if ratio >= 1.05 else 1

    def predict(self, roi_bgr):
        if not self.is_ready():
            return None

        inner_eq, mask = _prepare_inner(roi_bgr)
        estimated_count = self._estimate_digit_count(mask)

        if estimated_count == 0:
            return None

        candidates = []

        if estimated_count == 1:
            boxes = _single_box(mask)

            if len(boxes) == 1:
                feat = _digit_feature(inner_eq, mask, boxes[0], self.digit_size)
                best, top = self._classify_digit(feat, allowed=list(range(10)))

                if best:
                    candidates.append({
                        "label": int(best["digit"]),
                        "score": float(best["score"]),
                        "margin": float(best["margin"]),
                        "digits": [int(best["digit"])],
                        "digit_results": [best],
                        "source": "digit_single"
                    })

        if estimated_count == 2:
            boxes = _split_two_boxes(mask)

            if len(boxes) == 2:
                left_feat = _digit_feature(inner_eq, mask, boxes[0], self.digit_size)
                right_feat = _digit_feature(inner_eq, mask, boxes[1], self.digit_size)

                left_best, left_top = self._classify_digit(left_feat, allowed=[1, 2, 3])
                right_best, right_top = self._classify_digit(right_feat, allowed=list(range(10)))

                if left_top and right_top:
                    for a in left_top:
                        for b in right_top:
                            value = int(f"{a['digit']}{b['digit']}")

                            if 10 <= value <= 36:
                                score = (float(a["score"]) + float(b["score"])) / 2.0
                                margin = min(float(a.get("margin", 0.0)), float(b.get("margin", 0.0)))

                                candidates.append({
                                    "label": value,
                                    "score": float(score),
                                    "margin": float(margin),
                                    "digits": [int(a["digit"]), int(b["digit"])],
                                    "digit_results": [a, b],
                                    "source": "digit_double"
                                })

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x["score"], x["margin"]), reverse=True)

        best = dict(candidates[0])
        second = candidates[1] if len(candidates) > 1 else None

        best["second_label"] = second["label"] if second else None
        best["second_score"] = second["score"] if second else -1.0
        best["candidate_count"] = len(candidates)
        best["estimated_digit_count"] = estimated_count

        return best
