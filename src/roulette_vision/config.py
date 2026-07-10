import json
from pathlib import Path


DEFAULT_CONFIG = {
    "camera": {"width": 1280, "height": 720, "fps": 15},
    "roi": {"x": 0.058, "y": 0.004, "w": 0.060, "h": 0.090},
    "state": {
        "empty_confirm_frames": 3,
        "empty_template_score_min": 0.88,
        "empty_white_ratio_max": 0.006,
        "empty_std_max": 18.0,
        "bright_white_ratio_min": 0.020,
        "bright_std_min": 30.0
    },
    "matching": {
        "match_size_w": 96,
        "match_size_h": 72,
        "score_min": 0.72,
        "margin_min": 0.06
    },
    "decision": {
        "vote_window": 12,
        "min_votes": 7,
        "min_agreement": 0.70
    }
}


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path="data/config.json"):
    path = Path(path)
    cfg = DEFAULT_CONFIG

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = deep_merge(DEFAULT_CONFIG, user_cfg)

    return cfg


def ensure_project_dirs():
    for directory in [
        "data/templates",
        "data/empty",
        "data/evidence",
        "data/logs",
        "data/debug"
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
