import random
import os
import hashlib
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio

from config.config import AUGMENTATION_CONFIG, CONFIG
from augmentations import (
    CutoutAugment,
    FrequencyMaskAugment,
    HorizontalRollAugment,
    SODAAugment,
    TimeMaskAugment,
    VerticalRollAugment,
)
from .utils import parse_option


def normalize_like_dataset(fbank: torch.Tensor, input_is_log: bool = True) -> torch.Tensor:
    x = fbank
    if input_is_log:
        x = torch.exp(x)
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    x = torch.where(
        (x_max - x_min) > 1e-6,
        (x - x_min) / (x_max - x_min + 1e-6),
        torch.full_like(x, 0.5),
    )
    return (x - 0.5) / 0.5


class SodaCapAugmenter:
    def __init__(
        self,
        policy_matrix: np.ndarray,
        options: List[str],
        seed: int = 42,
        soda_apply_prob: float = None,
        audio_dir: str = "",
        offline_pitch_dir: str = "",
        offline_cache_dir: str = "",
        input_is_log_fbank: bool = True,
        quality_csv_path: str = "",
        augment_quality_levels: Optional[Set[int]] = None,
    ):
        self.policy_matrix = np.asarray(policy_matrix, dtype=np.float64)
        self.options = list(options)
        self.rng = np.random.RandomState(seed)
        self.audio_dir = os.path.normpath(audio_dir) if audio_dir else ""
        cfg_offline = AUGMENTATION_CONFIG.get("offline_pitch_shift", {})
        self.offline_pitch_dir = os.path.normpath(
            offline_pitch_dir or cfg_offline.get("offline_dir", "")
        )
        self.offline_cache_dir = os.path.normpath(
            offline_cache_dir
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "offline_pitch_fbank_cache")
        )
        os.makedirs(self.offline_cache_dir, exist_ok=True)
        self.input_is_log_fbank = input_is_log_fbank
        self.quality_csv_path = quality_csv_path
        self.augment_quality_levels = set(augment_quality_levels) if augment_quality_levels is not None else None
        self.sample_rate = CONFIG["audio_sample_rate"]
        self.max_len = CONFIG["audio_max_len"]
        self.mel_bins = CONFIG["mel_bins"]
        self.target_length = CONFIG["target_length"]
        self._resamplers = {}
        self._offline_index = self._build_offline_pitch_index()
        self._quality_index = self._build_quality_index()
        self._bad_offline_paths = set()
        self._stats = {
            "samples_seen": 0,
            "samples_augmented": 0,
            "samples_skipped_by_prob": 0,
            "samples_skipped_by_quality": 0,
            "offline_pitch_real": 0,
            "offline_pitch_fallback_to_soda_only": 0,
            "offline_pitch_bad_file": 0,
        }
        self._option_counts = {option: 0 for option in self.options}
        self._class_option_counts: Dict[int, Dict[str, int]] = {}
        cfg_soda = AUGMENTATION_CONFIG.get("soda", {})
        self.augment_prob = (
            soda_apply_prob if soda_apply_prob is not None else cfg_soda.get("apply_prob", 0.5)
        )
        # Important: keep the original SODA global usage ratio through
        # augment_prob, but force the internal SODA transform when an
        # augmented branch is actually selected.
        self.soda = SODAAugment(
            enable=True,
            apply_prob=1.0,
            kernel_len=cfg_soda.get("kernel_len", 31),
            style_shift_range=cfg_soda.get("style_shift_range", 0.5),
            noise_mix_ratio=cfg_soda.get("noise_mix_ratio", 0.4),
        )
        self.single_augs = {
            "horizontal_roll": HorizontalRollAugment(enable=True, max_roll_ratio=AUGMENTATION_CONFIG.get("horizontal_roll", {}).get("max_roll_ratio", 0.5), apply_prob=AUGMENTATION_CONFIG.get("horizontal_roll", {}).get("apply_prob", 0.5)),
            "vertical_roll": VerticalRollAugment(enable=True, max_roll_bins=AUGMENTATION_CONFIG.get("vertical_roll", {}).get("max_roll_bins", 15), apply_prob=AUGMENTATION_CONFIG.get("vertical_roll", {}).get("apply_prob", 0.5)),
            "time_mask": TimeMaskAugment(enable=True, max_mask_ratio=AUGMENTATION_CONFIG.get("time_mask", {}).get("max_mask_ratio", 0.2), num_masks=AUGMENTATION_CONFIG.get("time_mask", {}).get("num_masks", 2), apply_prob=AUGMENTATION_CONFIG.get("time_mask", {}).get("apply_prob", 0.5)),
            "frequency_mask": FrequencyMaskAugment(enable=True, max_mask_bins=AUGMENTATION_CONFIG.get("frequency_mask", {}).get("max_mask_bins", 20), num_masks=AUGMENTATION_CONFIG.get("frequency_mask", {}).get("num_masks", 2), apply_prob=AUGMENTATION_CONFIG.get("frequency_mask", {}).get("apply_prob", 0.5)),
            "cutout": CutoutAugment(
                enable=True,
                min_area_ratio=AUGMENTATION_CONFIG.get("cutout", {}).get("min_area_ratio", 0.02),
                max_area_ratio=AUGMENTATION_CONFIG.get("cutout", {}).get("max_area_ratio", 0.4),
                aspect_ratio_min=AUGMENTATION_CONFIG.get("cutout", {}).get("aspect_ratio_min", 0.3),
                apply_prob=AUGMENTATION_CONFIG.get("cutout", {}).get("apply_prob", 0.5),
            ),
        }

    def _build_quality_index(self) -> Dict[str, int]:
        index: Dict[str, int] = {}
        if not self.quality_csv_path or not os.path.exists(self.quality_csv_path):
            return index
        try:
            df = pd.read_csv(self.quality_csv_path)
        except Exception:
            return index

        for _, row in df.iterrows():
            rel = str(row["file_path"]).replace("\\", os.sep)
            key = os.path.normcase(os.path.normpath(rel))
            try:
                index[key] = int(row["quality_level"])
            except Exception:
                continue
        return index

    def _lookup_quality_level(self, source_path: str) -> Optional[int]:
        if not source_path or not self.audio_dir or not self._quality_index:
            return None
        try:
            rel = os.path.relpath(os.path.normpath(source_path), os.path.normpath(self.audio_dir))
        except ValueError:
            return None
        return self._quality_index.get(os.path.normcase(os.path.normpath(rel)))

    def _build_offline_pitch_index(self) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        if not self.offline_pitch_dir or not os.path.isdir(self.offline_pitch_dir):
            return index
        for class_name in os.listdir(self.offline_pitch_dir):
            class_dir = os.path.join(self.offline_pitch_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            try:
                filenames = os.listdir(class_dir)
            except OSError:
                continue
            for filename in filenames:
                stem, ext = os.path.splitext(filename)
                if "_ps_" not in stem:
                    continue
                original_stem = stem.split("_ps_", 1)[0]
                key = f"{class_name}/{original_stem}{ext.lower()}"
                index.setdefault(key, []).append(os.path.join(class_dir, filename))
        return index

    def _sample_option(self, class_idx: int) -> str:
        probs = self.policy_matrix[class_idx].astype(np.float64)
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 1e-12:
            probs = np.ones(len(self.options), dtype=np.float64) / len(self.options)
        else:
            probs = probs / total
        return self.options[int(self.rng.choice(len(self.options), p=probs))]

    def _offline_pitch_candidates(self, source_path: str) -> List[str]:
        if not source_path or not self.audio_dir or not self.offline_pitch_dir:
            return []
        class_name = os.path.basename(os.path.dirname(source_path))
        filename = os.path.basename(source_path)
        stem, ext = os.path.splitext(filename)
        key = f"{class_name}/{stem}{ext.lower()}"
        return self._offline_index.get(key, [])

    def _cache_path_for_audio(self, path: str) -> str:
        digest = hashlib.md5(os.path.normpath(path).encode("utf-8")).hexdigest()
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(self.offline_cache_dir, f"{stem}_{digest}.pt")

    def _path_to_raw_fbank(self, path: str, device: torch.device) -> torch.Tensor:
        if path in self._bad_offline_paths:
            raise RuntimeError(f"Known bad offline pitch file: {path}")

        cache_path = self._cache_path_for_audio(path)
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location=device, weights_only=True)
            except TypeError:
                return torch.load(cache_path, map_location=device)

        try:
            waveform, sr = torchaudio.load(path)
        except Exception as exc:
            self._bad_offline_paths.add(path)
            raise RuntimeError(f"Failed to load offline pitch file: {path}") from exc

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self._resamplers[sr](waveform)
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            waveform = F.pad(waveform, (0, self.max_len - waveform.shape[1]))
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=self.mel_bins,
            frame_length=25,
            frame_shift=10,
            htk_compat=True,
        )
        if fbank.shape[0] < self.target_length:
            fbank = F.pad(fbank, (0, 0, 0, self.target_length - fbank.shape[0]))
        else:
            fbank = fbank[: self.target_length, :]
        try:
            torch.save(fbank.cpu(), cache_path)
        except OSError:
            pass
        return fbank.to(device)

    def _apply_aux(self, fbank: torch.Tensor, aux_name: str) -> torch.Tensor:
        if not aux_name:
            return fbank
        if aux_name == "gaussian_noise":
            sigma = random.uniform(0.005, 0.03)
            return fbank + torch.randn_like(fbank) * sigma
        if aux_name == "pink_noise":
            t_dim, f_dim = fbank.shape
            sigma = random.uniform(0.003, 0.02)
            weights = 1.0 / (
                torch.arange(1, f_dim + 1, dtype=fbank.dtype, device=fbank.device).sqrt() + 1e-8
            )
            weights = weights / weights.mean()
            return fbank + torch.randn(t_dim, f_dim, device=fbank.device, dtype=fbank.dtype) * weights.unsqueeze(0) * sigma
        if aux_name == "offline_pitch_shift":
            return fbank
        if aux_name in self.single_augs:
            return self.single_augs[aux_name](fbank)
        return fbank

    def __call__(self, batch_fbank: torch.Tensor, labels: torch.Tensor, paths=None) -> torch.Tensor:
        out = []
        hard_labels = labels.argmax(dim=1) if labels.ndim == 2 else labels
        for i in range(batch_fbank.size(0)):
            self._stats["samples_seen"] += 1
            class_idx = int(hard_labels[i].item())
            x = batch_fbank[i]
            source_path = paths[i] if paths is not None else None

            quality_level = self._lookup_quality_level(source_path) if source_path is not None else None
            if self.augment_quality_levels is not None:
                if quality_level is None or quality_level not in self.augment_quality_levels:
                    self._stats["samples_skipped_by_quality"] += 1
                    out.append(normalize_like_dataset(x, input_is_log=self.input_is_log_fbank))
                    continue

            if random.random() > self.augment_prob:
                self._stats["samples_skipped_by_prob"] += 1
                out.append(normalize_like_dataset(x, input_is_log=self.input_is_log_fbank))
                continue

            self._stats["samples_augmented"] += 1
            option = self._sample_option(class_idx)
            self._option_counts[option] = self._option_counts.get(option, 0) + 1
            if class_idx not in self._class_option_counts:
                self._class_option_counts[class_idx] = {name: 0 for name in self.options}
            self._class_option_counts[class_idx][option] = self._class_option_counts[class_idx].get(option, 0) + 1
            _, aux_name = parse_option(option)

            # SODA-CAP train loader returns raw fbank before dataset-level
            # augmentation/normalization, so the policy is applied at the same
            # stage as the original SODA implementation.
            used_real_offline_pitch = False
            if aux_name == "offline_pitch_shift" and paths is not None:
                candidates = [
                    candidate for candidate in self._offline_pitch_candidates(source_path)
                    if candidate not in self._bad_offline_paths
                ]
                random.shuffle(candidates)
                for candidate in candidates:
                    try:
                        x = self._path_to_raw_fbank(candidate, batch_fbank.device)
                        used_real_offline_pitch = True
                        self._stats["offline_pitch_real"] += 1
                        break
                    except RuntimeError:
                        self._stats["offline_pitch_bad_file"] += 1
                        continue
                if not used_real_offline_pitch:
                    self._stats["offline_pitch_fallback_to_soda_only"] += 1
                    aux_name = ""
            x = self.soda(x)
            if not used_real_offline_pitch:
                x = self._apply_aux(x, aux_name)
            x = normalize_like_dataset(x, input_is_log=self.input_is_log_fbank)
            out.append(x)
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
