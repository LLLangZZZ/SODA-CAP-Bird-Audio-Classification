"""Training-time SODA-CAP augmenter.

For each sample, SODA-CAP samples a class-specific option from a policy matrix.
Every option is SODA-centered: either SODA-only or SODA followed by one
auxiliary augmentation.
"""

import random
from typing import Dict, List

import numpy as np
import torch

from augmentations import (
    CutoutAugment,
    FrequencyMaskAugment,
    GaussianNoiseAugment,
    HorizontalRollAugment,
    PinkNoiseAugment,
    SODAAugment,
    TimeMaskAugment,
    VerticalRollAugment,
)
from config.config import AUGMENTATION_CONFIG
from .utils import parse_option


def normalize_like_dataset(fbank: torch.Tensor, input_is_log: bool = True) -> torch.Tensor:
    """Apply the same min-max normalization used by the dataset pipeline."""

    x = torch.exp(fbank) if input_is_log else fbank
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    x = torch.where(
        (x_max - x_min) > 1e-6,
        (x - x_min) / (x_max - x_min + 1e-6),
        torch.full_like(x, 0.5),
    )
    return (x - 0.5) / 0.5


class SodaCapAugmenter:
    """Apply SODA-only or SODA-plus-auxiliary augmentation per class."""

    def __init__(
        self,
        policy_matrix: np.ndarray,
        options: List[str],
        seed: int = 42,
        soda_apply_prob: float = None,
        input_is_log_fbank: bool = True,
    ):
        self.policy_matrix = np.asarray(policy_matrix, dtype=np.float64)
        self.options = list(options)
        self.rng = np.random.RandomState(seed)
        self.input_is_log_fbank = input_is_log_fbank
        self._stats = {
            "samples_seen": 0,
            "samples_augmented": 0,
            "samples_skipped_by_prob": 0,
        }
        self._option_counts = {option: 0 for option in self.options}
        self._class_option_counts: Dict[int, Dict[str, int]] = {}

        soda_cfg = AUGMENTATION_CONFIG["soda"]
        self.augment_prob = soda_apply_prob if soda_apply_prob is not None else soda_cfg.get("apply_prob", 0.5)
        self.soda = SODAAugment(
            enable=True,
            apply_prob=1.0,
            kernel_len=soda_cfg.get("kernel_len", 31),
            style_shift_range=soda_cfg.get("style_shift_range", 0.5),
            noise_mix_ratio=soda_cfg.get("noise_mix_ratio", 0.4),
        )
        self.single_augs = {
            "horizontal_roll": HorizontalRollAugment(enable=True, **_without_enable("horizontal_roll")),
            "vertical_roll": VerticalRollAugment(enable=True, **_without_enable("vertical_roll")),
            "time_mask": TimeMaskAugment(enable=True, **_without_enable("time_mask")),
            "frequency_mask": FrequencyMaskAugment(enable=True, **_without_enable("frequency_mask")),
            "gaussian_noise": GaussianNoiseAugment(enable=True, **_without_enable("gaussian_noise")),
            "pink_noise": PinkNoiseAugment(enable=True, **_without_enable("pink_noise")),
            "cutout": CutoutAugment(enable=True, **_without_enable("cutout")),
        }

    def _sample_option(self, class_idx: int) -> str:
        """Sample an option from the policy row for one class."""

        probs = np.maximum(self.policy_matrix[class_idx].astype(np.float64), 0.0)
        probs = probs / probs.sum() if probs.sum() > 1e-12 else np.ones(len(self.options)) / len(self.options)
        return self.options[int(self.rng.choice(len(self.options), p=probs))]

    def _apply_aux(self, fbank: torch.Tensor, aux_name: str) -> torch.Tensor:
        if not aux_name:
            return fbank
        if aux_name not in self.single_augs:
            raise ValueError(f"Unknown SODA-CAP auxiliary augmentation: {aux_name}")
        return self.single_augs[aux_name](fbank)

    def __call__(self, batch_fbank: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        out = []
        hard_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
        for i in range(batch_fbank.size(0)):
            self._stats["samples_seen"] += 1
            x = batch_fbank[i]
            class_idx = int(hard_labels[i].item())

            if random.random() > self.augment_prob:
                self._stats["samples_skipped_by_prob"] += 1
                out.append(normalize_like_dataset(x, input_is_log=self.input_is_log_fbank))
                continue

            self._stats["samples_augmented"] += 1
            option = self._sample_option(class_idx)
            self._option_counts[option] = self._option_counts.get(option, 0) + 1
            self._class_option_counts.setdefault(class_idx, {name: 0 for name in self.options})
            self._class_option_counts[class_idx][option] = self._class_option_counts[class_idx].get(option, 0) + 1

            _, aux_name = parse_option(option)
            x = self.soda(x)
            x = self._apply_aux(x, aux_name)
            out.append(normalize_like_dataset(x, input_is_log=self.input_is_log_fbank))
        return torch.stack(out, dim=0)

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def option_counts(self) -> Dict[str, int]:
        return dict(self._option_counts)

    def class_option_counts(self) -> Dict[int, Dict[str, int]]:
        return {int(k): dict(v) for k, v in self._class_option_counts.items()}

    def reset_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0
        self._option_counts = {option: 0 for option in self.options}
        self._class_option_counts = {}


def _without_enable(name: str) -> Dict:
    """Return augmentation config without the duplicated enable argument."""

    cfg = dict(AUGMENTATION_CONFIG.get(name, {}))
    cfg.pop("enable", None)
    return cfg
