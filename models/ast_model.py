"""AST classifier used by the SODA-CAP paper.

The paper focuses on SODA and SODA-CAP rather than on a new classifier
architecture. This module defines the AST-based spectrogram classifier used as
the stable backbone for 384 x 384 log-Mel features.
"""

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASTModel(nn.Module):
    """Adapt an image ViT model to single-channel audio spectrograms."""

    _feature_shape_cache = {}

    def __init__(
        self,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 384,
        input_tdim: int = 384,
        imagenet_pretrain: bool = True,
        model_size: str = "base384",
        verbose: bool = True,
    ):
        super().__init__()
        model_map = {
            "tiny224": "vit_deit_tiny_distilled_patch16_224",
            "small224": "vit_deit_small_distilled_patch16_224",
            "base224": "vit_deit_base_distilled_patch16_224",
            "base384": "vit_base_patch16_384",
        }
        if model_size not in model_map:
            raise ValueError(f"model_size must be one of {list(model_map)}")

        offline = os.environ.get("TIMM_DISABLE_DOWNLOAD") == "1"
        try:
            self.v = timm.create_model(model_map[model_size], pretrained=imagenet_pretrain and not offline)
        except Exception as exc:
            if verbose:
                print(f"[AST] pretrained initialization failed; using random weights: {exc}")
            self.v = timm.create_model(model_map[model_size], pretrained=False)

        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.original_hw = int(self.v.patch_embed.num_patches ** 0.5)

        freq_patches, time_patches = self.get_patch_shape()
        num_patches = freq_patches * time_patches
        self.v.patch_embed.num_patches = num_patches

        new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain and hasattr(self.v.patch_embed, "proj"):
            new_proj.weight = nn.Parameter(self.v.patch_embed.proj.weight.sum(dim=1, keepdim=True))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        cls_token = self.v.pos_embed[:, :1, :]
        has_dist_token = hasattr(self.v, "dist_token") and self.v.dist_token is not None
        dist_token = self.v.pos_embed[:, 1:2, :] if has_dist_token else None
        patch_start = 2 if has_dist_token else 1
        patch_pos = self.v.pos_embed[:, patch_start:, :]
        patch_pos = patch_pos.reshape(1, self.original_hw, self.original_hw, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(),
            size=(freq_patches, time_patches),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, num_patches, -1)

        tokens = [cls_token]
        if dist_token is not None:
            tokens.append(dist_token)
        tokens.append(patch_pos)
        self.v.pos_embed = nn.Parameter(torch.cat(tokens, dim=1))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, 768),
        )

    def get_patch_shape(self):
        """Compute the spectrogram patch grid after the AST patch projection."""

        key = (self.fstride, self.tstride, self.input_fdim, self.input_tdim)
        if key not in self._feature_shape_cache:
            with torch.no_grad():
                conv = nn.Conv2d(1, 1, kernel_size=(16, 16), stride=(self.fstride, self.tstride))
                out = conv(torch.randn(1, 1, self.input_fdim, self.input_tdim))
            self._feature_shape_cache[key] = (out.shape[2], out.shape[3])
        return self._feature_shape_cache[key]

    def forward(self, x: torch.Tensor):
        """Run the AST backbone.

        The expected input shape is [batch, time, mel], matching the 384 x 384
        log-Mel features described in the paper.
        """

        x = x.unsqueeze(1)
        batch_size = x.shape[0]
        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(batch_size, -1, -1)
        if hasattr(self.v, "dist_token") and self.v.dist_token is not None:
            dist_tokens = self.v.dist_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.v.pos_drop(x + self.v.pos_embed)
        for block in self.v.blocks:
            x = block(x)
        x = self.v.norm(x)
        return {"embedding": self.mlp_head(x[:, 0])}


class StandaloneASTClassifier(nn.Module):
    """AST backbone followed by a two-layer MLP classifier."""

    def __init__(
        self,
        num_classes: int = 36,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 384,
        input_tdim: int = 384,
        imagenet_pretrain: bool = True,
        model_size: str = "base384",
        ast_pretrained_path: str = None,
        classifier_dropout: float = 0.5,
        verbose: bool = True,
    ):
        super().__init__()
        self.ast = ASTModel(
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            model_size=model_size,
            verbose=verbose,
        )
        if ast_pretrained_path and os.path.exists(ast_pretrained_path):
            self._load_pretrained_weights(ast_pretrained_path, verbose)

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(512, num_classes),
        )

    def _load_pretrained_weights(self, pretrained_path: str, verbose: bool = True):
        """Load local AST weights while skipping incompatible classifier layers."""

        ckpt = torch.load(pretrained_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "").replace("ast.", "")
            if not new_key.startswith(("classifier", "mlp_head")):
                cleaned[new_key] = value

        current = self.ast.state_dict()
        loadable = {}
        for key, value in cleaned.items():
            if key not in current:
                continue
            if key == "v.patch_embed.proj.weight" and value.shape != current[key].shape:
                loadable[key] = value.mean(dim=1, keepdim=True)
            elif value.shape == current[key].shape:
                loadable[key] = value

        self.ast.load_state_dict(loadable, strict=False)
        if verbose:
            print(f"Loaded AST weights: {len(loadable)}/{len(current)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.ast(x)["embedding"]
        return self.classifier(features)
