from pathlib import Path
import cv2
import numpy as np


def roi_rect_from_shape(frame_shape, roi_cfg):
    h, w = frame_shape[:2]

    rw = int(float(roi_cfg["w"]) * w)
    rh = int(float(roi_cfg["h"]) * h)

    mode = roi_cfg.get("mode", "xy")

    if mode == "center":
        cx = float(roi_cfg.get("cx", 0.5))
        cy = float(roi_cfg.get("cy", 0.5))
        x = int(cx * w - rw / 2)
        y = int(cy * h - rh / 2)
    else:
        x = int(float(roi_cfg["x"]) * w)
        y = int(float(roi_cfg["y"]) * h)

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))

    return x, y, rw, rh


def crop_relative_roi(frame, roi_cfg):
    x, y, rw, rh = roi_rect_from_shape(frame.shape, roi_cfg)
    return frame[y:y + rh, x:x + rw]


def crop_config_roi(frame, cfg, key="roi"):
    roi_cfg = cfg.get(key) or cfg["roi"]
    return crop_relative_roi(frame, roi_cfg)


def letterbox_resize(gray, size):
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


def extract_digit_content_gray(img_bgr, size):
    """
    Extracts only the bright digit content from inside the result box.
    This makes matching robust against small ROI shifts and blue-background differences.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Remove outer border area of the blue box.
    x1 = int(0.08 * w)
    x2 = int(0.92 * w)
    y1 = int(0.10 * h)
    y2 = int(0.90 * h)

    inner = gray[y1:y2, x1:x2]

    if inner.size == 0:
        return np.zeros((size[1], size[0]), dtype=np.float32)

    blur = cv2.GaussianBlur(inner, (3, 3), 0)

    # Digit is white/bright. Use a strict threshold to ignore blue background.
    threshold_value = max(135, int(np.percentile(blur, 82)))
    mask = (blur >= threshold_value).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.where(mask > 0)

    # If no digit pixels are found, return a normalized inner crop.
    if len(xs) < 12 or len(ys) < 12:
        normalized = cv2.equalizeHist(inner)
        return letterbox_resize(normalized, size).astype(np.float32)

    bx1 = int(xs.min())
    bx2 = int(xs.max())
    by1 = int(ys.min())
    by2 = int(ys.max())

    bw = bx2 - bx1 + 1
    bh = by2 - by1 + 1

    pad_x = max(4, int(0.18 * bw))
    pad_y = max(4, int(0.20 * bh))

    bx1 = max(0, bx1 - pad_x)
    by1 = max(0, by1 - pad_y)
    bx2 = min(inner.shape[1] - 1, bx2 + pad_x)
    by2 = min(inner.shape[0] - 1, by2 + pad_y)

    digit_crop = inner[by1:by2 + 1, bx1:bx2 + 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    digit_crop = clahe.apply(digit_crop)

    # Convert to a clean high-contrast digit shape.
    _, binary = cv2.threshold(
        digit_crop,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return letterbox_resize(binary, size).astype(np.float32)


def normalize_for_match(img_bgr, size):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return gray.astype(np.float32)


def normalize_digit_mask(img_bgr, size):
    return extract_digit_content_gray(img_bgr, size)


def ncc_score(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a = a - a.mean()
    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.sum(a * b) / denom)


def roi_metrics(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    white_mask = gray > 170
    overexposed_mask = gray > 245

    red_mask_1 = cv2.inRange(hsv, (0, 50, 25), (12, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (165, 50, 25), (179, 255, 255))
    red_mask = (red_mask_1 | red_mask_2) > 0

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    unsharp = cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)
    digit_edge_sharpness = cv2.Laplacian(unsharp, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "white_ratio": float(white_mask.mean()),
        "overexposed_ratio": float(overexposed_mask.mean()),
        "red_ratio": float(red_mask.mean()),
        "sharpness": float(sharpness),
        "digit_edge_sharpness": float(digit_edge_sharpness),
        "edge_density": edge_density
    }


class TemplateMatcher:
    def __init__(self, template_root, match_cfg):
        self.template_root = Path(template_root)
        self.size = (
            int(match_cfg["match_size_w"]),
            int(match_cfg["match_size_h"])
        )
        self.top_k = int(match_cfg.get("top_k", 20))
        self.templates = []
        self.load_templates()

    def load_templates(self):
        self.templates.clear()

        if not self.template_root.exists():
            return

        for label_dir in sorted(self.template_root.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            if not label.isdigit():
                continue

            for img_path in sorted(label_dir.glob("*.png")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                norm_gray = normalize_for_match(img, self.size)
                norm_digit = normalize_digit_mask(img, self.size)

                self.templates.append({
                    "label": int(label),
                    "path": str(img_path),
                    "gray": norm_gray,
                    "digit": norm_digit
                })

    def is_ready(self):
        return len(self.templates) > 0

    def labels(self):
        return sorted({t["label"] for t in self.templates})

    def predict(self, roi_bgr):
        if not self.templates:
            return None

        from collections import Counter

        current_gray = normalize_for_match(roi_bgr, self.size)
        current_digit = normalize_digit_mask(roi_bgr, self.size)

        all_scores = []
        best_by_label = {}

        for template in self.templates:
            gray_score = ncc_score(current_gray, template["gray"])
            digit_score = ncc_score(current_digit, template["digit"])

            score = 0.15 * gray_score + 0.85 * digit_score
            label = template["label"]

            item = {
                "label": label,
                "score": float(score),
                "gray_score": float(gray_score),
                "digit_score": float(digit_score),
                "template_path": template["path"]
            }

            all_scores.append(item)

            if label not in best_by_label or item["score"] > best_by_label[label]["score"]:
                best_by_label[label] = item

        ranked_labels = sorted(best_by_label.values(), key=lambda x: x["score"], reverse=True)

        best = dict(ranked_labels[0])
        second_score = ranked_labels[1]["score"] if len(ranked_labels) > 1 else -1.0
        second_label = ranked_labels[1]["label"] if len(ranked_labels) > 1 else None

        best["second_score"] = float(second_score)
        best["second_label"] = second_label
        best["margin"] = float(best["score"] - second_score)

        ranked_templates = sorted(all_scores, key=lambda x: x["score"], reverse=True)
        top_templates = ranked_templates[:max(1, self.top_k)]
        counts = Counter(t["label"] for t in top_templates)

        topk_count = counts.get(best["label"], 0)
        topk_total = len(top_templates)
        topk_agreement = topk_count / max(1, topk_total)

        best["top_k"] = topk_total
        best["topk_count"] = topk_count
        best["topk_agreement"] = float(topk_agreement)
        best["topk_counts"] = dict(counts.most_common(5))

        return best


class EmptyMatcher:
    def __init__(self, empty_root, match_cfg):
        self.empty_root = Path(empty_root)
        self.size = (
            int(match_cfg["match_size_w"]),
            int(match_cfg["match_size_h"])
        )
        self.templates = []
        self.load_templates()

    def load_templates(self):
        self.templates.clear()

        if not self.empty_root.exists():
            return

        for img_path in sorted(self.empty_root.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            self.templates.append(normalize_for_match(img, self.size))

    def score(self, roi_bgr):
        if not self.templates:
            return None

        current = normalize_for_match(roi_bgr, self.size)
        return max(ncc_score(current, t) for t in self.templates)
