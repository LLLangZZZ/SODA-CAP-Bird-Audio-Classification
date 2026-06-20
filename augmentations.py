"""Augmentation modules used by the SODA-CAP paper.

This file keeps only SODA and the seven auxiliary candidates used by the
SODA-CAP policy: horizontal roll, vertical roll, time mask, frequency mask,
Gaussian noise, pink noise, and cutout.
"""

import math
import random
from typing import Dict, List

import torch
import torch.nn.functional as F


class HorizontalRollAugment:
    """Circularly shift a spectrogram along the time axis."""

    def __init__(self, enable: bool = False, max_roll_ratio: float = 0.5, apply_prob: float = 0.5):
        self.enable = enable
        self.max_roll_ratio = max_roll_ratio
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank
        max_roll = int(fbank.shape[0] * self.max_roll_ratio)
        shift = random.randint(-max_roll, max_roll)
        return torch.roll(fbank, shifts=shift, dims=0) if shift else fbank


class VerticalRollAugment:
    """Circularly shift a spectrogram along the frequency axis."""

    def __init__(self, enable: bool = False, max_roll_bins: int = 15, apply_prob: float = 0.5):
        self.enable = enable
        self.max_roll_bins = max_roll_bins
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank
        shift = random.randint(-self.max_roll_bins, self.max_roll_bins)
        return torch.roll(fbank, shifts=shift, dims=1) if shift else fbank


class TimeMaskAugment:
    """Mask contiguous time frames in the spectrogram."""

    def __init__(self, enable: bool = False, max_mask_ratio: float = 0.2, num_masks: int = 2, apply_prob: float = 0.5):
        self.enable = enable
        self.max_mask_ratio = max_mask_ratio
        self.num_masks = num_masks
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank
        out = fbank.clone()
        time_len = out.shape[0]
        max_width = max(1, int(time_len * self.max_mask_ratio))
        for _ in range(self.num_masks):
            width = random.randint(1, max_width)
            start = random.randint(0, max(0, time_len - width))
            out[start : start + width, :] = 0.0
        return out


class FrequencyMaskAugment:
    """Mask contiguous Mel-frequency bands in the spectrogram."""

    def __init__(self, enable: bool = False, max_mask_bins: int = 20, num_masks: int = 2, apply_prob: float = 0.5):
        self.enable = enable
        self.max_mask_bins = max_mask_bins
        self.num_masks = num_masks
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank
        out = fbank.clone()
        freq_len = out.shape[1]
        for _ in range(self.num_masks):
            width = random.randint(1, min(self.max_mask_bins, freq_len))
            start = random.randint(0, max(0, freq_len - width))
            out[:, start : start + width] = 0.0
        return out


class GaussianNoiseAugment:
    """Add small Gaussian noise to a spectrogram."""

    def __init__(self, enable: bool = False, min_std: float = 0.005, max_std: float = 0.03, apply_prob: float = 0.5):
        self.enable = enable
        self.min_std = min_std
        self.max_std = max_std
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank
        sigma = random.uniform(self.min_std, self.max_std)
        return fbank + torch.randn_like(fbank) * sigma


class PinkNoiseAugment:
    """Add frequency-weighted noise that approximates pink noise."""

    def __init__(self, enable: bool = False, min_std: float = 0.003, max_std: float = 0.02, apply_prob: float = 0.5):
        self.enable = enable
        self.min_std = min_std
        self.max_std = max_std
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank
        time_len, freq_len = fbank.shape
        sigma = random.uniform(self.min_std, self.max_std)
        weights = torch.arange(1, freq_len + 1, dtype=fbank.dtype, device=fbank.device)
        weights = (1.0 / weights.sqrt()).unsqueeze(0)
        weights = weights / weights.mean()
        noise = torch.randn(time_len, freq_len, dtype=fbank.dtype, device=fbank.device)
        return fbank + noise * weights * sigma


class CutoutAugment:
    """Erase a random rectangular time-frequency region."""

    def __init__(
        self,
        enable: bool = False,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.4,
        aspect_ratio_min: float = 0.3,
        apply_prob: float = 0.5,
    ):
        self.enable = enable
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_ratio_min = aspect_ratio_min
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank
        time_len, freq_len = fbank.shape
        area = time_len * freq_len
        target_area = random.uniform(self.min_area_ratio, self.max_area_ratio) * area
        aspect = random.uniform(self.aspect_ratio_min, 1.0 / self.aspect_ratio_min)
        height = min(time_len, int(round(math.sqrt(target_area * aspect))))
        width = min(freq_len, int(round(math.sqrt(target_area / aspect))))
        if height <= 0 or width <= 0:
            return fbank
        top = random.randint(0, time_len - height)
        left = random.randint(0, freq_len - width)
        out = fbank.clone()
        out[top : top + height, left : left + width] = 0.0
        return out


class SODAAugment:
    """Source-oriented data augmentation for bird spectrograms.

    SODA estimates a bird-dominant component with directional smoothing, treats
    the positive residual as noise-dominant content, perturbs the residual style
    and strength, and recomposes the augmented spectrogram.
    """

    def __init__(
        self,
        enable: bool = False,
        apply_prob: float = 0.5,
        kernel_len: int = 31,
        style_shift_range: float = 0.5,
        noise_mix_ratio: float = 0.4,
    ):
        self.enable = enable
        self.apply_prob = apply_prob
        self.kernel_len = kernel_len
        self.style_shift_range = style_shift_range
        self.noise_mix_ratio = noise_mix_ratio

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank

        x = fbank.unsqueeze(0).unsqueeze(0)
        pad = self.kernel_len // 2

        time_smooth = F.avg_pool2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), (self.kernel_len, 1), stride=1)
        freq_smooth = F.avg_pool2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), (1, self.kernel_len), stride=1)
        bird_component = torch.maximum(time_smooth, freq_smooth)

        noise_component = F.relu(x - bird_component)
        noise_mean = noise_component.mean(dim=2, keepdim=True)
        style_shift = (torch.rand_like(noise_mean) * 2.0 - 1.0) * self.style_shift_range
        virtual_noise = noise_component + style_shift * noise_component
        gain = 1.0 + random.random() * self.noise_mix_ratio
        augmented = bird_component + virtual_noise * gain
        return augmented.squeeze(0).squeeze(0)


class SpectrogramAugmentPipeline:
    """Apply enabled single-sample spectrogram augmentations in a fixed order."""

    def __init__(self, augment_config: Dict):
        self.augment_chain = [
            SODAAugment(**augment_config.get("soda", {})),
            HorizontalRollAugment(**augment_config.get("horizontal_roll", {})),
            VerticalRollAugment(**augment_config.get("vertical_roll", {})),
            TimeMaskAugment(**augment_config.get("time_mask", {})),
            FrequencyMaskAugment(**augment_config.get("frequency_mask", {})),
            GaussianNoiseAugment(**augment_config.get("gaussian_noise", {})),
            PinkNoiseAugment(**augment_config.get("pink_noise", {})),
            CutoutAugment(**augment_config.get("cutout", {})),
        ]

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        for augment in self.augment_chain:
            fbank = augment(fbank)
        return fbank


def get_enabled_augmentations(augment_config: Dict) -> List[str]:
    """Return enabled augmentation names for experiment logging."""

    return sorted(
        name
        for name, cfg in augment_config.items()
        if isinstance(cfg, dict) and cfg.get("enable", False)
    )
