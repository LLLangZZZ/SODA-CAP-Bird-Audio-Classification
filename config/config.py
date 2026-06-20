"""Configuration values matching the SODA-CAP paper setup.

This file intentionally avoids absolute local paths. Fill MODEL_PATHS with your
own dataset/checkpoint locations if you want to run the code.
"""

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = {
    "num_classes": 36,
    "batch_size": 24,
    "epochs": 50,
    "lr": 1e-4,
    "backbone_lr": 5e-5,
    "weight_decay": 5e-5,
    "patience": 10,
    "min_lr": 1e-6,
    "data_split": [0.6, 0.2, 0.2],
    "audio_sample_rate": 16000,
    "audio_max_len": 160000,
    "mel_bins": 384,
    "target_length": 384,
    "ast_fstride": 10,
    "ast_tstride": 10,
    "ast_model_size": "base384",
    "ast_trainable_blocks": 4,
    "hidden_dim": 512,
    "dropout": 0.5,
    "quality_csv_path": "",
    "allowed_qualities": [0, 1, 2],
}


TRAINING_CONFIG = {
    "num_workers": 0,
    "gradient_clip_norm": 1.0,
    "warmup_epochs": 3,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "label_smoothing": 0.1,
    "gradient_accumulation_steps": 1,
    "use_layer_wise_lr": True,
    "llrd_decay_rate": 0.8,
}


AUGMENTATION_CONFIG = {
    "spec_aug": {
        "enable": False,
        "time_drop_width": 64,
        "time_stripes_num": 2,
        "freq_drop_width": 8,
        "freq_stripes_num": 2,
    },
    "gaussian_noise": {
        "enable": False,
        "min_snr_db": 5,
        "max_snr_db": 20,
        "apply_prob": 0.5,
    },
    "pink_noise": {
        "enable": False,
        "min_snr_db": 5,
        "max_snr_db": 20,
        "apply_prob": 0.5,
    },
    "pitch_shift": {
        "enable": False,
        "min_shift": -4,
        "max_shift": 4,
        "apply_prob": 0.35,
    },
    "bird_mixup": {
        "enable": False,
        "alpha": 0.5,
        "apply_prob": 0.5,
        "use_soft_label": True,
        "max_pool_size": 2000,
    },
    "horizontal_roll": {
        "enable": False,
        "max_roll_ratio": 0.5,
        "apply_prob": 0.5,
    },
    "vertical_roll": {
        "enable": False,
        "max_roll_bins": 15,
        "apply_prob": 0.5,
    },
    "time_mask": {
        "enable": False,
        "max_mask_ratio": 0.2,
        "num_masks": 2,
        "apply_prob": 0.5,
    },
    "frequency_mask": {
        "enable": False,
        "max_mask_bins": 20,
        "num_masks": 2,
        "apply_prob": 0.5,
    },
    "cutout": {
        "enable": False,
        "min_area_ratio": 0.02,
        "max_area_ratio": 0.4,
        "aspect_ratio_min": 0.3,
        "apply_prob": 0.5,
    },
    "spectrogram_mixup": {
        "enable": False,
        "alpha": 0.5,
        "apply_prob": 0.5,
    },
    "mixed_frequency_masking": {
        "enable": False,
        "max_mask_bins": 30,
        "num_masks": 1,
        "apply_prob": 0.5,
    },
    "soda": {
        "enable": True,
        "apply_prob": 0.5,
        "kernel_len": 31,
        "style_shift_range": 0.5,
        "noise_mix_ratio": 0.4,
    },
    "offline_pitch_shift": {
        "enable": False,
        "offline_dir": "",
    },
}


MODEL_PATHS = {
    "audio_dir": "",
    "ast_pretrained_path": "",
    "best_model_path": "best_pure_audio_model_soda.pth",
    "final_model_path": "final_pure_audio_model_soda.pth",
}


def get_config():
    return {
        "CONFIG": CONFIG,
        "TRAINING_CONFIG": TRAINING_CONFIG,
        "AUGMENTATION_CONFIG": AUGMENTATION_CONFIG,
        "MODEL_PATHS": MODEL_PATHS,
        "device": device,
    }
