"""
PureAudio Augmentations - 独立可插拔的音频增强模块
专为消融实验设计，每个增强完全独立，支持单独开关与参数配置
更新：移除主/次背景噪声，新增7种鸟类分类专用增强
"""
import random
import math
import torch
import torch.nn.functional as F
import torchaudio
from typing import Optional, List, Tuple, Union

# 全局设备适配：强制设为 CPU，彻底根除 DataLoader 多进程中 pitch_shift 等 CUDA 子进程冲突
device = torch.device("cpu")


def calculate_rms(signal: torch.Tensor) -> torch.Tensor:
    """计算音频信号的均方根（用于信噪比校准）"""
    return torch.sqrt(torch.mean(signal ** 2))


def adjust_noise_gain(signal: torch.Tensor, noise: torch.Tensor, target_snr_db: float) -> torch.Tensor:
    """根据目标信噪比调整噪声增益"""
    signal_rms = calculate_rms(signal)
    noise_rms = calculate_rms(noise)
    target_noise_rms = signal_rms / (10 ** (target_snr_db / 20))
    noise_gain = target_noise_rms / (noise_rms + 1e-8)
    return noise * noise_gain


# ===================== 第一部分：时域波形增强（作用于原始音频）=====================
# 1. 高斯噪声增强（保留原有，独立模块）
class GaussianNoiseAugment:
    def __init__(self,
                 enable: bool = False,
                 min_snr_db: float = 5,
                 max_snr_db: float = 20,
                 apply_prob: float = 0.5):
        self.enable = enable
        self.min_snr = min_snr_db
        self.max_snr = max_snr_db
        self.apply_prob = apply_prob

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return waveform

        target_snr = random.uniform(self.min_snr, self.max_snr)
        noise = torch.randn_like(waveform)
        noise = adjust_noise_gain(waveform, noise, target_snr)
        return waveform + noise


# 2. 粉噪声增强（保留原有，独立模块）
class PinkNoiseAugment:
    def __init__(self,
                 enable: bool = False,
                 min_snr_db: float = 5,
                 max_snr_db: float = 20,
                 apply_prob: float = 0.5):
        self.enable = enable
        self.min_snr = min_snr_db
        self.max_snr = max_snr_db
        self.apply_prob = apply_prob

    def _generate_pink_noise(self, length: int) -> torch.Tensor:
        """生成1/f功率谱的粉噪声"""
        X = torch.randn(length, dtype=torch.float32)
        fft_X = torch.fft.rfft(X)
        freq_bins = torch.arange(1, len(fft_X) + 1, dtype=torch.float32)
        fft_X = fft_X / torch.sqrt(freq_bins)
        pink_noise = torch.fft.irfft(fft_X, n=length)
        pink_noise = pink_noise / (torch.max(torch.abs(pink_noise)) + 1e-8)
        return pink_noise

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return waveform

        target_snr = random.uniform(self.min_snr, self.max_snr)
        noise = self._generate_pink_noise(waveform.shape[1]).unsqueeze(0)
        noise = adjust_noise_gain(waveform, noise, target_snr)
        return waveform + noise


# 3. 音调偏移增强（Pitch shift）- 新增独立模块
class PitchShiftAugment:
    def __init__(self,
                 enable: bool = False,
                 sample_rate: int = 16000,
                 min_shift: int = -4,
                 max_shift: int = 4,
                 apply_prob: float = 0.5):
        """
        音调偏移，不改变音频时长
        :param min_shift/max_shift: 半音偏移范围，负数降调、正数升调
        """
        self.enable = enable
        self.sample_rate = sample_rate
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.apply_prob = apply_prob
        self.shifter_cache = {}
        # 启动时预建所有 PitchShift，避免训练过程中首次命中某半音时卡顿/拖慢 worker
        for n in range(self.min_shift, self.max_shift + 1):
            if n != 0:
                self.shifter_cache[n] = torchaudio.transforms.PitchShift(
                    sample_rate=self.sample_rate, n_steps=n
                )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return waveform

        n_steps = random.randint(self.min_shift, self.max_shift)
        if n_steps == 0:
            return waveform

        return self.shifter_cache[n_steps](waveform)


# 4. 随机鸟类样本混合（Mixed up random bird species）- 修复版
class BirdMixUpAugment:
    def __init__(self,
                 enable: bool = False,
                 alpha: float = 0.5,
                 apply_prob: float = 0.5,
                 use_soft_label: bool = True,
                 max_pool_size: int = 2000,
                 sample_rate: int = 32000,
                 audio_paths: List[str] = None):
        """
        鸟类样本 MixUp 增强
        加入 RMS 能量对齐与 Soft Label 边界修复
        """
        self.enable = enable
        self.alpha = alpha
        self.apply_prob = apply_prob
        self.use_soft_label = use_soft_label
        self.sample_rate = sample_rate
        self.audio_paths = audio_paths or []
        self.audio_labels: List[int] = []
        self._resamplers = {}

    def set_audio_paths(self, paths: List[str], labels: List[int]):
        """单独设置路径列表（由 Dataset 调用，与构造参数二选一）"""
        self.audio_paths = paths
        self.audio_labels = labels

    def _load_audio(self, path: str, target_len: int) -> torch.Tensor:
        """直接从磁盘加载音频（不经过 Dataset/增强流程），避免递归"""
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self._resamplers[sr](waveform)
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
        return waveform

    def _get_fallback_label(self, label: int) -> Union[torch.Tensor, int]:
        """[新增] 关键修复：开启 soft_label 时，如果未触发 mixup，也必须返回 one-hot 格式以保持 DataLoader 维度统一"""
        if self.use_soft_label:
            num_classes = max(self.audio_labels) + 1 if self.audio_labels else 36
            soft_label = torch.zeros(num_classes)
            soft_label[label] = 1.0
            return soft_label
        return label

    def __call__(self, waveform: torch.Tensor, label: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        if not self.enable or random.random() > self.apply_prob or len(self.audio_paths) == 0:
            return waveform, self._get_fallback_label(label)

        # 随机选一个其他样本（避免与自己混合）
        candidates = [i for i, l in enumerate(self.audio_labels) if l != label]
        if not candidates:
            return waveform, self._get_fallback_label(label)
        idx = random.choice(candidates)

        try:
            mix_waveform = self._load_audio(
                self.audio_paths[idx],
                target_len=waveform.shape[1]
            )
        except Exception:
            return waveform, self._get_fallback_label(label)

        # === [新增] 基于 RMS 能量的音量对齐 ===
        rms_main = calculate_rms(waveform)
        rms_mix = calculate_rms(mix_waveform)
        if rms_mix > 1e-8:
            mix_waveform = mix_waveform * (rms_main / rms_mix)

        lam = random.betavariate(self.alpha, self.alpha)
        mixed_waveform = lam * waveform + (1 - lam) * mix_waveform

        # === [新增] 重新归一化，防止音频叠加导致爆音 (Clipping) ===
        max_val = torch.max(torch.abs(mixed_waveform))
        if max_val > 1.0:
            mixed_waveform = mixed_waveform / (max_val + 1e-8)

        if self.use_soft_label:
            num_classes = max(self.audio_labels) + 1
            soft_label = torch.zeros(num_classes)
            soft_label[label] = lam
            soft_label[self.audio_labels[idx]] = 1 - lam
            return mixed_waveform, soft_label
        else:
            final_label = label if lam >= 0.5 else self.audio_labels[idx]
            return mixed_waveform, final_label


# ===================== 第二部分：频域频谱增强（作用于Mel频谱）=====================
# 5. 水平滚动增强（Horizontal roll）- 新增独立模块
class HorizontalRollAugment:
    def __init__(self,
                 enable: bool = False,
                 max_roll_ratio: float = 0.5,
                 apply_prob: float = 0.5):
        """频谱时间轴（水平）循环移位，增强时序平移不变性"""
        self.enable = enable
        self.max_roll_ratio = max_roll_ratio
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        """输入fbank形状: [Time, Freq]"""
        if not self.enable or random.random() > self.apply_prob:
            return fbank

        time_length = fbank.shape[0]
        max_roll = int(time_length * self.max_roll_ratio)
        roll_steps = random.randint(-max_roll, max_roll)
        if roll_steps == 0:
            return fbank

        return torch.roll(fbank, shifts=roll_steps, dims=0)


# 6. 垂直滚动增强（Vertical roll）- 新增独立模块
class VerticalRollAugment:
    def __init__(self,
                 enable: bool = False,
                 max_roll_bins: int = 15,
                 apply_prob: float = 0.5):
        """频谱频率轴（垂直）循环移位，增强频率偏移鲁棒性"""
        self.enable = enable
        self.max_roll_bins = max_roll_bins
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank

        roll_steps = random.randint(-self.max_roll_bins, self.max_roll_bins)
        if roll_steps == 0:
            return fbank

        return torch.roll(fbank, shifts=roll_steps, dims=1)


# 7. 时间掩码增强（Time mask）- 新增独立模块
class TimeMaskAugment:
    def __init__(self,
                 enable: bool = False,
                 max_mask_ratio: float = 0.2,
                 num_masks: int = 2,
                 apply_prob: float = 0.5):
        """时间轴连续掩码，和SpecAugment解耦，独立控制"""
        self.enable = enable
        self.max_mask_ratio = max_mask_ratio
        self.num_masks = num_masks
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank

        time_length = fbank.shape[0]
        max_mask_length = max(1, int(time_length * self.max_mask_ratio))
        masked_fbank = fbank.clone()

        for _ in range(self.num_masks):
            mask_length = random.randint(0, max_mask_length)
            if mask_length == 0:
                continue
            start_idx = random.randint(0, max(0, time_length - mask_length))
            masked_fbank[start_idx:start_idx + mask_length, :] = 0.0

        return masked_fbank


# 8. 频率掩码增强（Frequency mask）- 新增独立模块
class FrequencyMaskAugment:
    def __init__(self,
                 enable: bool = False,
                 max_mask_bins: int = 20,
                 num_masks: int = 2,
                 apply_prob: float = 0.5):
        """频率轴连续掩码，和SpecAugment解耦，独立控制"""
        self.enable = enable
        self.max_mask_bins = max_mask_bins
        self.num_masks = num_masks
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank

        freq_length = fbank.shape[1]
        masked_fbank = fbank.clone()

        for _ in range(self.num_masks):
            mask_length = random.randint(0, min(self.max_mask_bins, freq_length))
            if mask_length == 0:
                continue
            start_idx = random.randint(0, max(0, freq_length - mask_length))
            masked_fbank[:, start_idx:start_idx + mask_length] = 0.0

        return masked_fbank


# ===================== 第三部分：新增的高阶频谱增强 =====================
# 9. Cutout 随机擦除（作用于单张频谱图）
class CutoutAugment:
    def __init__(self, enable=False, min_area_ratio=0.02, max_area_ratio=0.4, aspect_ratio_min=0.3, apply_prob=0.5):
        self.enable = enable
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.aspect_ratio_min = aspect_ratio_min
        self.apply_prob = apply_prob

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        """输入 fbank 形状: [Time, Freq]"""
        if not self.enable or random.random() > self.apply_prob:
            return fbank

        T, F = fbank.shape
        total_area = T * F
        target_area = random.uniform(self.min_area_ratio, self.max_area_ratio) * total_area
        aspect_ratio = random.uniform(self.aspect_ratio_min, 1.0 / self.aspect_ratio_min)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        h = min(h, T)
        w = min(w, F)

        if h == 0 or w == 0:
            return fbank

        top = random.randint(0, T - h)
        left = random.randint(0, F - w)

        masked_fbank = fbank.clone()
        masked_fbank[top:top + h, left:left + w] = 0.0
        return masked_fbank


# 10. Mixed Frequency Masking（作用于 Batch，非线性特征替换 + 线性标签混合）
class MixedFrequencyMaskingAugment:
    def __init__(self, enable=False, max_mask_bins=30, num_masks=1, apply_prob=0.5, num_classes=36):
        self.enable = enable
        self.max_mask_bins = max_mask_bins
        self.num_masks = num_masks
        self.apply_prob = apply_prob
        self.num_classes = num_classes

    def __call__(self, batch_fbank: torch.Tensor, batch_labels: torch.Tensor):
        """
        batch_fbank: [B, T, F]
        batch_labels: [B] 或 [B, num_classes] (Soft Label)
        """
        if not self.enable or random.random() > self.apply_prob:
            return batch_fbank, batch_labels

        B, T, F_dim = batch_fbank.shape

        # 统一转为 Soft Label 以支持混合
        if batch_labels.ndim == 1:
            soft_labels = F.one_hot(batch_labels, num_classes=self.num_classes).float()
        else:
            soft_labels = batch_labels.clone()

        mixed_fbank = batch_fbank.clone()
        mixed_labels = soft_labels.clone()

        # 在 Batch 内部打乱顺序，寻找混合的 "Partner"
        indices = torch.randperm(B, device=batch_fbank.device)
        partner_fbank = batch_fbank[indices]
        partner_labels = soft_labels[indices]

        for i in range(B):
            total_replaced_width = 0
            for _ in range(self.num_masks):
                mask_width = random.randint(1, self.max_mask_bins)
                start_f = random.randint(0, max(0, F_dim - mask_width))

                # 用 Partner 的频带替换当前样本的频带
                mixed_fbank[i, :, start_f:start_f + mask_width] = partner_fbank[i, :, start_f:start_f + mask_width]
                total_replaced_width += mask_width

            # 计算替换比例 beta
            beta = min(1.0, total_replaced_width / F_dim)

            # 标签按替换比例混合
            mixed_labels[i] = (1 - beta) * soft_labels[i] + beta * partner_labels[i]

        return mixed_fbank, mixed_labels


# 11. 频谱 Mixup（作用于 Batch，类似于 VH-Mixup 的线性版本）
class SpectrogramMixupAugment:
    def __init__(self, enable=False, alpha=0.5, apply_prob=0.5, num_classes=36):
        self.enable = enable
        self.alpha = alpha
        self.apply_prob = apply_prob
        self.num_classes = num_classes

    def __call__(self, batch_fbank: torch.Tensor, batch_labels: torch.Tensor):
        if not self.enable or random.random() > self.apply_prob:
            return batch_fbank, batch_labels

        B = batch_fbank.shape[0]
        if batch_labels.ndim == 1:
            soft_labels = F.one_hot(batch_labels, num_classes=self.num_classes).float()
        else:
            soft_labels = batch_labels.clone()

        indices = torch.randperm(B, device=batch_fbank.device)
        lam = random.betavariate(self.alpha, self.alpha)

        mixed_fbank = lam * batch_fbank + (1 - lam) * batch_fbank[indices]
        mixed_labels = lam * soft_labels + (1 - lam) * soft_labels[indices]

        return mixed_fbank, mixed_labels


class SODAAugment:
    def __init__(self,
                 enable: bool = False,
                 apply_prob: float = 0.5,
                 kernel_len: int = 31,
                 style_shift_range: float = 0.5,
                 noise_mix_ratio: float = 0.4):
        self.enable = enable
        self.apply_prob = apply_prob
        self.kernel_len = kernel_len
        self.style_shift_range = style_shift_range
        self.noise_mix_ratio = noise_mix_ratio

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        if not self.enable or random.random() > self.apply_prob:
            return fbank

        x = fbank.unsqueeze(0).unsqueeze(0)

        pad_len = self.kernel_len // 2

        x_padded_h = F.pad(x, (0, 0, pad_len, pad_len), mode='reflect')
        harmonic = F.avg_pool2d(x_padded_h, kernel_size=(self.kernel_len, 1), stride=1)

        x_padded_p = F.pad(x, (pad_len, pad_len, 0, 0), mode='reflect')
        percussive = F.avg_pool2d(x_padded_p, kernel_size=(1, self.kernel_len), stride=1)

        X_bird = torch.maximum(harmonic, percussive)
        X_noise = F.relu(x - X_bird)

        noise_mean_freq = X_noise.mean(dim=2, keepdim=True)
        random_style_shift = (torch.rand_like(noise_mean_freq) * 2 - 1) * self.style_shift_range
        X_noise_virtual = X_noise + random_style_shift * X_noise

        volume_gain = 1.0 + (random.random() * self.noise_mix_ratio)
        X_aug = X_bird + X_noise_virtual * volume_gain

        return X_aug.squeeze(0).squeeze(0)


# ===================== 增强流水线封装（统一调度）=====================
class WaveformAugmentPipeline:
    """
    时域波形增强流水线：统一调度所有作用于原始音频（Waveform）的增强。
    支持独立开关，严格遵循 AUGMENTATION_CONFIG。
    【关键设计】通过 audio_paths 参数彻底断开 Dataset 递归，
    MixUp 时直接从磁盘读取，不持有任何 Dataset 对象引用。
    """
    def __init__(self, augment_config: dict, sample_rate: int, audio_paths: List[str] = None):
        self.config = augment_config
        self.sample_rate = sample_rate

        # 1. 实例化各个基础增强模块（即使 disable，类内部也会自动跳过）
        self.gaussian_noise = GaussianNoiseAugment(**augment_config.get('gaussian_noise', {}))
        self.pink_noise = PinkNoiseAugment(**augment_config.get('pink_noise', {}))
        self.pitch_shift = PitchShiftAugment(
            **augment_config.get('pitch_shift', {}), sample_rate=sample_rate
        )

        # 2. 基础增强链（不含 Mixup，按配置独立控制开关）
        self.base_chain = [
            self.gaussian_noise,
            self.pink_noise,
            self.pitch_shift,
        ]

        # 3. BirdMixup 实例化（【关键】传入路径列表，不持有 Dataset 引用）
        mixup_cfg = augment_config.get('bird_mixup', {'enable': False})
        self.bird_mixup = BirdMixUpAugment(
            **mixup_cfg, sample_rate=sample_rate, audio_paths=audio_paths or []
        )

    def __call__(self, waveform: torch.Tensor, label: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        # 确保在 CPU 上运行
        waveform = waveform.to(device)

        # A. 基础增强（按配置开关，disable 时瞬间返回）
        for augment in self.base_chain:
            waveform = augment(waveform)

        # B. BirdMixup（按配置开关，disable 时瞬间返回）
        waveform, label = self.bird_mixup(waveform, label)

        return waveform, label


class SpectrogramAugmentPipeline:
    """频域频谱增强流水线，统一调度所有频谱层面增强"""
    def __init__(self, augment_config: dict):
        self.horizontal_roll = HorizontalRollAugment(**augment_config.get('horizontal_roll', {}))
        self.vertical_roll = VerticalRollAugment(**augment_config.get('vertical_roll', {}))
        self.time_mask = TimeMaskAugment(**augment_config.get('time_mask', {}))
        self.frequency_mask = FrequencyMaskAugment(**augment_config.get('frequency_mask', {}))
        self.cutout = CutoutAugment(**augment_config.get('cutout', {}))

        self.augment_chain = [
            self.horizontal_roll, self.vertical_roll,
            self.time_mask, self.frequency_mask, self.cutout
        ]

    def __call__(self, fbank: torch.Tensor) -> torch.Tensor:
        for augment in self.augment_chain:
            fbank = augment(fbank)
        return fbank


class BatchSpectrogramAugmentPipeline:
    """专为需要在 GPU 上进行多样本融合（Mixup/替换）的增强设计"""
    def __init__(self, augment_config: dict, num_classes: int):
        self.mixed_freq_masking = MixedFrequencyMaskingAugment(
            **augment_config.get('mixed_frequency_masking', {}), num_classes=num_classes
        )
        self.spec_mixup = SpectrogramMixupAugment(
            **augment_config.get('spectrogram_mixup', {}), num_classes=num_classes
        )
        self.augment_chain = [self.mixed_freq_masking, self.spec_mixup]

    def __call__(self, batch_fbank: torch.Tensor, batch_labels: torch.Tensor):
        for augment in self.augment_chain:
            batch_fbank, batch_labels = augment(batch_fbank, batch_labels)
        return batch_fbank, batch_labels


# ===================== 工具函数：获取当前启用的增强列表 =====================
def get_enabled_augmentations(augment_config: dict) -> list:
    """
    根据配置获取当前启用的增强手段列表
    :param augment_config: AUGMENTATION_CONFIG 字典
    :return: 启用的增强名称列表（按字母序）
    """
    enabled = []
    # 跳过 spec_aug（已在别处单独处理）
    for key, cfg in augment_config.items():
        if key == 'spec_aug':
            continue
        if isinstance(cfg, dict) and cfg.get('enable', False):
            enabled.append(key)
    return sorted(enabled)
