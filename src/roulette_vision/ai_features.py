import cv2
import numpy as np


def _resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _augment_bgr(img, rng):
    out = img.copy()

    h, w = out.shape[:2]

    angle = float(rng.uniform(-2.0, 2.0))
    scale = float(rng.uniform(0.97, 1.03))
    tx = float(rng.uniform(-2.0, 2.0))
    ty = float(rng.uniform(-2.0, 2.0))

    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty

    out = cv2.warpAffine(
        out,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    alpha = float(rng.uniform(0.88, 1.12))
    beta = float(rng.uniform(-12, 12))
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.20:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


def extract_ai_feature(roi_bgr, augment=False, rng=None):
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros(1, dtype=np.float32)

    img = _resize(roi_bgr, (96, 96))

    if augment:
        rng = rng or np.random.default_rng()
        img = _augment_bgr(img, rng)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray)

    gray64 = _resize(gray_eq, (64, 64))
    gray32 = _resize(gray_eq, (32, 32)).astype(np.float32) / 255.0
    bgr16 = _resize(img, (16, 16)).astype(np.float32) / 255.0

    hog = cv2.HOGDescriptor(
        (64, 64),
        (16, 16),
        (8, 8),
        (8, 8),
        9
    )
    hog_feat = hog.compute(gray64).reshape(-1).astype(np.float32)

    h, s, v = cv2.split(hsv)

    hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)

    hist_h = hist_h / max(1.0, float(hist_h.sum()))
    hist_s = hist_s / max(1.0, float(hist_s.sum()))
    hist_v = hist_v / max(1.0, float(hist_v.sum()))

    white = ((v >= 145) & (s <= 125)).mean()
    green = ((h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)).mean()
    red = (((h <= 12) | (h >= 165)) & (s >= 45) & (v >= 45)).mean()
    dark = (v <= 60).mean()

    stats = np.array([
        gray.mean() / 255.0,
        gray.std() / 255.0,
        h.mean() / 180.0,
        s.mean() / 255.0,
        v.mean() / 255.0,
        s.std() / 255.0,
        v.std() / 255.0,
        float(white),
        float(green),
        float(red),
        float(dark),
    ], dtype=np.float32)

    feature = np.concatenate([
        hog_feat,
        gray32.reshape(-1),
        bgr16.reshape(-1),
        hist_h.astype(np.float32),
        hist_s.astype(np.float32),
        hist_v.astype(np.float32),
        stats,
    ]).astype(np.float32)

    return feature
