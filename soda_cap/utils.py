import json
import os
from typing import Dict, List, Tuple

import torch

from config.config import CONFIG
from models.ast_model import StandaloneASTClassifier


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "soda_cap", "output")
DEFAULT_UNMIX_DIR = os.path.join(PROJECT_ROOT, "unmix")

BASELINE_NAME = "baseline"
SODA_NAME = "soda"

MODEL_FILE_TO_NAME = {
    "best_pure_audio_model.pth": BASELINE_NAME,
    "best_pure_audio_model_soda.pth": SODA_NAME,
    "best_pure_audio_model_horizontal_roll.pth": "horizontal_roll",
    "best_pure_audio_model_vertical_roll.pth": "vertical_roll",
    "best_pure_audio_model_time_mask.pth": "time_mask",
    "best_pure_audio_model_frequency_mask.pth": "frequency_mask",
    "best_pure_audio_model_offline_pitch_shift.pth": "offline_pitch_shift",
    "best_pure_audio_model_gaussian_noise.pth": "gaussian_noise",
    "best_pure_audio_model_pink_noise.pth": "pink_noise",
    "best_pure_audio_model_cutout.pth": "cutout",
    "best_pure_audio_model_bird_mixup.pth": "bird_mixup",
    "best_pure_audio_model_spectrogram_mixup.pth": "spectrogram_mixup",
}

# SODA-CAP trains per sample, so batch label-mixing augmentations are excluded by
# default. They can still be used as single-augmentation baselines in reports.
DEFAULT_AUX_CANDIDATES = [
    "horizontal_roll",
    "vertical_roll",
    "time_mask",
    "frequency_mask",
    "gaussian_noise",
    "pink_noise",
    "cutout",
]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def discover_unmix_models(unmix_dir: str) -> Dict[str, str]:
    found = {}
    for filename, name in MODEL_FILE_TO_NAME.items():
        path = os.path.join(unmix_dir, filename)
        if os.path.exists(path):
            found[name] = path
    return found


def build_model_from_checkpoint(path: str, device: torch.device) -> StandaloneASTClassifier:
    ckpt = safe_torch_load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = StandaloneASTClassifier(
        num_classes=cfg.get("num_classes", CONFIG["num_classes"]),
        fstride=cfg.get("ast_fstride", CONFIG["ast_fstride"]),
        tstride=cfg.get("ast_tstride", CONFIG["ast_tstride"]),
        input_fdim=cfg.get("mel_bins", CONFIG["mel_bins"]),
        input_tdim=cfg.get("target_length", CONFIG["target_length"]),
        imagenet_pretrain=False,
        model_size=CONFIG.get("ast_model_size", "base384"),
        ast_pretrained_path=None,
        classifier_dropout=CONFIG.get("dropout", 0.5),
        verbose=False,
    )
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    return model.to(device)


def save_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def option_names(aux_names: List[str]) -> List[str]:
    return ["soda_only"] + [f"soda+{name}" for name in aux_names]


def parse_option(option: str) -> Tuple[bool, str]:
    if option == "soda_only":
        return True, ""
    if option.startswith("soda+"):
        return True, option.split("+", 1)[1]
    raise ValueError(f"Unsupported SODA-CAP option: {option}")
