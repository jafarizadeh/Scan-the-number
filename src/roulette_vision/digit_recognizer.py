from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np


def _normalize_vector(img):
    v = img.reshape(-1).astype(np.float32)
    v = v - float(v.mean())
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def _ncc_vec(a, b):
    return float(np.dot(a, b))


def _letterbox(img, size):
    target_w, target_h = size
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    return canvas


def _inner_gray_and_mask(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Remove border of the result box. Keep central digit area.
    x1 = int(0.08 * w)
    x2 = int(0.92 * w)
    y1 = int(0.10 * h)
    y2 = int(0.90 * h)

    inner = gray[y1:y2, x1:x2]

    if inner.size == 0:
        return gray, np.zeros_like(gray)

    blur = cv2.GaussianBlur(inner, (3, 3), 0)

    threshold = max(135, int(np.percentile(blur, 82)))
    mask = (blur >= threshold).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return inner, mask


def _runs_from_projection(mask):
    h, w = mask.shape[:2]
    active = (mask > 0).sum(axis=0)

    threshold = max(2, int(0.04 * h))
    cols = np.where(active >= threshold)[0]

    if len(cols) == 0:
        return []

    runs = []
    start = int(cols[0])
    prev = int(cols[0])

    for c in cols[1:]:
        c = int(c)
        if c - prev <= 3:
            prev = c
        else:
            runs.append([start, prev])
            start = c
            prev = c

    runs.append([start, prev])

    # Merge very close runs.
    merged = []
    for r in runs:
        if not merged:
            merged.append(r)
            continue

        if r[0] - merged[-1][1] <= 5:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    # Filter tiny runs.
    filtered = []
    for x1, x2 in merged:
        if x2 - x1 + 1 >= 4:
            filtered.append([x1, x2])

    return filtered


def _box_for_xrange(mask, x1, x2):
    sub = mask[:, x1:x2 + 1]
    ys, xs = np.where(sub > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    bx1 = x1 + int(xs.min())
    bx2 = x1 + int(xs.max())
    by1 = int(ys.min())
    by2 = int(ys.max())

    return [bx1, by1, bx2, by2]


def _segment_digit_boxes(mask, expected_count=None):
    h, w = mask.shape[:2]
    runs = _runs_from_projection(mask)

    if not runs:
        return []

    if expected_count == 1:
        x1 = min(r[0] for r in runs)
        x2 = max(r[1] for r in runs)
        box = _box_for_xrange(mask, x1, x2)
        return [box] if box else []

    if expected_count == 2:
        if len(runs) >= 2:
            # Split by the largest gap between runs.
            gaps = []
            for i in range(len(runs) - 1):
                gaps.append((runs[i + 1][0] - runs[i][1], i))

            _, split_idx = max(gaps)

            left_runs = runs[:split_idx + 1]
            right_runs = runs[split_idx + 1:]

            lx1 = min(r[0] for r in left_runs)
            lx2 = max(r[1] for r in left_runs)
            rx1 = min(r[0] for r in right_runs)
            rx2 = max(r[1] for r in right_runs)

            left_box = _box_for_xrange(mask, lx1, lx2)
            right_box = _box_for_xrange(mask, rx1, rx2)

            boxes = []
            if left_box:
                boxes.append(left_box)
            if right_box:
                boxes.append(right_box)
            return boxes

        # Fallback: split the active region in half.
        x1 = min(r[0] for r in runs)
        x2 = max(r[1] for r in runs)
        mid = (x1 + x2) // 2

        left_box = _box_for_xrange(mask, x1, mid)
        right_box = _box_for_xrange(mask, mid + 1, x2)

        boxes = []
        if left_box:
            boxes.append(left_box)
        if right_box:
            boxes.append(right_box)
        return boxes

    # Auto mode.
    if len(runs) >= 2:
        # If there is a clear large gap, use two digits.
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        largest_gap = max(gaps) if gaps else 0

        if largest_gap >= max(6, int(0.05 * w)):
            return _segment_digit_boxes(mask, expected_count=2)

    return _segment_digit_boxes(mask, expected_count=1)


def _feature_from_box(inner_gray, box, size):
    x1, y1, x2, y2 = box

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    pad_x = max(3, int(0.20 * bw))
    pad_y = max(3, int(0.20 * bh))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(inner_gray.shape[1] - 1, x2 + pad_x)
    y2 = min(inner_gray.shape[0] - 1, y2 + pad_y)

    crop = inner_gray[y1:y2 + 1, x1:x2 + 1]

    if crop.size == 0:
        return np.zeros(size[0] * size[1], dtype=np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    crop = clahe.apply(crop)

    _, binary = cv2.threshold(
        crop,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    normalized = _letterbox(binary, size)
    return _normalize_vector(normalized)


class DigitRecognizer:
    def __init__(self, template_root="data/templates", digit_size=(48, 72)):
        self.template_root = Path(template_root)
        self.digit_size = digit_size
        self.templates = defaultdict(list)
        self.load_templates()

    def load_templates(self):
        self.templates.clear()

        for number in range(37):
            label = str(number)
            label_dir = self.template_root / label

            if not label_dir.exists():
                continue

            expected_count = len(label)

            for path in sorted(label_dir.glob("*.png")):
                img = cv2.imread(str(path))
                if img is None:
                    continue

                inner, mask = _inner_gray_and_mask(img)
                boxes = _segment_digit_boxes(mask, expected_count=expected_count)

                if len(boxes) != expected_count:
                    continue

                for digit_char, box in zip(label, boxes):
                    digit = int(digit_char)
                    feature = _feature_from_box(inner, box, self.digit_size)
                    self.templates[digit].append({
                        "feature": feature,
                        "source": str(path),
                        "number": number
                    })

    def is_ready(self):
        return all(len(self.templates[d]) > 0 for d in range(10))

    def counts(self):
        return {str(d): len(self.templates[d]) for d in range(10)}

    def _classify_digit(self, feature):
        best_by_digit = {}

        for digit, items in self.templates.items():
            best_score = -999.0
            best_source = None

            for item in items:
                score = _ncc_vec(feature, item["feature"])
                if score > best_score:
                    best_score = score
                    best_source = item["source"]

            best_by_digit[digit] = {
                "digit": digit,
                "score": float(best_score),
                "source": best_source
            }

        ranked = sorted(best_by_digit.values(), key=lambda x: x["score"], reverse=True)

        best = ranked[0]
        second = ranked[1]

        best["second_digit"] = second["digit"]
        best["second_score"] = second["score"]
        best["margin"] = best["score"] - second["score"]

        return best, ranked[:4]

    def predict(self, roi_bgr):
        if not self.is_ready():
            return None

        inner, mask = _inner_gray_and_mask(roi_bgr)
        boxes = _segment_digit_boxes(mask, expected_count=None)

        if len(boxes) == 0:
            return None

        if len(boxes) > 2:
            boxes = boxes[:2]

        digit_results = []
        alternatives = []

        for box in boxes:
            feature = _feature_from_box(inner, box, self.digit_size)
            best, top = self._classify_digit(feature)
            digit_results.append(best)
            alternatives.append(top)

        candidates = []

        if len(digit_results) == 1:
            d = digit_results[0]
            value = d["digit"]

            if 0 <= value <= 9:
                candidates.append({
                    "label": value,
                    "digits": [value],
                    "score": d["score"],
                    "margin": d["margin"],
                    "digit_results": digit_results
                })

        elif len(digit_results) == 2:
            # Try combinations from top alternatives. Choose best valid roulette number.
            for a in alternatives[0]:
                for b in alternatives[1]:
                    value = int(f"{a['digit']}{b['digit']}")

                    if 10 <= value <= 36:
                        score = (a["score"] + b["score"]) / 2.0
                        margin = min(a["margin"], b["margin"])

                        candidates.append({
                            "label": value,
                            "digits": [a["digit"], b["digit"]],
                            "score": float(score),
                            "margin": float(margin),
                            "digit_results": [a, b]
                        })

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x["score"], x["margin"]), reverse=True)

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        best["source"] = "digit"
        best["second_label"] = second["label"] if second else None
        best["second_score"] = second["score"] if second else -1.0
        best["candidate_count"] = len(candidates)

        return best
