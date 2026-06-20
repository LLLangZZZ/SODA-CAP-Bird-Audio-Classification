"""
PureAudio Model - Standalone AST (Audio Spectrogram Transformer)
纯音频模型 - 优化的独立音频频谱图 Transformer
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ASTModel(nn.Module):
    """
    音频频谱图 Transformer 主干网络
    修复了设备硬编码问题，并优化了 2D 位置编码插值逻辑
    """
    _feature_shape_cache = {}
    
    def __init__(self, label_dim=768, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, model_size='base384', verbose=True):
        super(ASTModel, self).__init__()
        
        # 离线检测逻辑保持不变
        offline_flag = any(os.environ.get(k) == "1" for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TIMM_DISABLE_DOWNLOAD"])
        if offline_flag: os.environ["TIMM_DISABLE_DOWNLOAD"] = "1"

        # 1. 创建 timm 模型 (移除内部 .to(device))
        model_map = {
            'tiny224': 'vit_deit_tiny_distilled_patch16_224',
            'small224': 'vit_deit_small_distilled_patch16_224',
            'base224': 'vit_deit_base_distilled_patch16_224',
            'base384': 'vit_base_patch16_384'
        }
        
        if model_size not in model_map:
            raise ValueError(f'Model size must be one of {list(model_map.keys())}')
        
        try:
            self.v = timm.create_model(model_map[model_size], pretrained=imagenet_pretrain and not offline_flag)
        except Exception as e:
            if verbose: print(f"[AST] WARN: timm pretrained load failed: {e}")
            self.v = timm.create_model(model_map[model_size], pretrained=False)
        
        # 2. 参数适配
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5) # 通常为 24 (对于 384 尺寸)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim, self.input_tdim = input_fdim, input_tdim
        
        f_dim, t_dim = self.get_shape()
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        # 3. 修改投影层以处理单通道音频频谱图
        # 移除 .to(device)，由外部统一处理设备
        new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain:
            # 将 RGB 三通道权重合一
            new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # 4. 优化的 2D 位置编码插值
        # 提取 cls_token 和原 patch 位置编码
        cls_token = self.v.pos_embed[:, :1, :]
        dist_token = self.v.pos_embed[:, 1:2, :] if 'distilled' in model_map[model_size] else None
        
        # 剩余的是 patch 编码，形状为 [1, 576, 768] (假设是 base384)
        patch_start = 2 if dist_token is not None else 1
        orig_patch_pos = self.v.pos_embed[:, patch_start:, :] 
        
        # 重塑为 2D 形状进行双线性插值
        orig_patch_pos = orig_patch_pos.reshape(1, self.oringal_hw, self.oringal_hw, -1).permute(0, 3, 1, 2)
        new_patch_pos = F.interpolate(orig_patch_pos.float(), size=(f_dim, t_dim), mode='bilinear', align_corners=False)
        new_patch_pos = new_patch_pos.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
        
        # 重新拼接位置编码
        tokens_to_cat = [cls_token]
        if dist_token is not None: tokens_to_cat.append(dist_token)
        tokens_to_cat.append(new_patch_pos)
        self.v.pos_embed = nn.Parameter(torch.cat(tokens_to_cat, dim=1))

        # 5. 特征提取头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, 768)
        )

    def get_shape(self):
        """计算频谱图 patch 后的维度"""
        cache_key = (self.fstride, self.tstride, self.input_fdim, self.input_tdim)
        if cache_key in self._feature_shape_cache:
            return self._feature_shape_cache[cache_key]
        
        # 使用虚拟输入计算输出尺寸
        with torch.no_grad():
            tmp_conv = nn.Conv2d(1, 1, kernel_size=(16, 16), stride=(self.fstride, self.tstride))
            tmp_out = tmp_conv(torch.randn(1, 1, self.input_fdim, self.input_tdim))
            shape = (tmp_out.shape[2], tmp_out.shape[3])
        
        self._feature_shape_cache[cache_key] = shape
        return shape

    def forward(self, x):
        """
        前向传播 (已修复 transpose 逻辑以匹配常规频谱图习惯)
        输入 x 形状: (batch, freq, time)
        """
        x = x.unsqueeze(1)    # (B, 1, F, T)
        x = x.transpose(2, 3) # (B, 1, T, F) -> 将时间轴作为序列长度的主导
        
        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        # 这里的逻辑需要适配是否有 dist_token
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        if hasattr(self.v, 'dist_token') and self.v.dist_token is not None:
            dist_tokens = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
        
        x = self.v.norm(x)
        x = x[:, 0] # 提取 cls_token
        
        x = self.mlp_head(x)
        return {'embedding': x}


class StandaloneASTClassifier(nn.Module):
    """
    优化的独立 AST 分类器
    去除了内部层冻结，以便与 LLRD 配合使用
    """
    def __init__(self, num_classes=36, fstride=10, tstride=10, 
                 input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, model_size='base384',
                 ast_pretrained_path=None, classifier_dropout=0.5, verbose=True):
        super(StandaloneASTClassifier, self).__init__()
        
        # 1. 初始化 Backbone (移除 label_dim=527 的硬编码，统一使用特征维度)
        self.ast = ASTModel(
            fstride=fstride, tstride=tstride,
            input_fdim=input_fdim, input_tdim=input_tdim,
            imagenet_pretrain=imagenet_pretrain,
            model_size=model_size, verbose=verbose
        )
        
        # 2. 加载预训练权重
        if ast_pretrained_path and os.path.exists(ast_pretrained_path):
            self._load_pretrained_weights(ast_pretrained_path, verbose)
        
        # 3. 分类头：提高 Dropout 缓解过拟合（默认 0.5）
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(512, num_classes)
        )
        
        # 注意：移除了 _configure_backbone_finetuning (不再冻结层)
        # 这样 trainer.py 中的 LLRD 才能对全网络生效

    def _load_pretrained_weights(self, pretrained_path, verbose=True):
        """增强的权重加载逻辑，包含 2D 位置编码适配"""
        try:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            
            # 清理 key 名
            new_sd = {}
            for k, v in state_dict.items():
                nk = k.replace('module.', '').replace('ast.', '')
                if not nk.startswith('classifier') and not nk.startswith('mlp_head'):
                    new_sd[nk] = v

            curr_sd = self.ast.state_dict()
            load_sd = {}
            
            for k, v in new_sd.items():
                if k not in curr_sd: continue

                # 适配单通道
                if k == 'v.patch_embed.proj.weight' and v.shape != curr_sd[k].shape:
                    load_sd[k] = v.mean(dim=1, keepdim=True)
                # 优化的位置编码插值（在加载时复用 2D 逻辑）
                elif k == 'v.pos_embed' and v.shape != curr_sd[k].shape:
                    if verbose: print(f"   Resizing pos_embed: {v.shape} -> {curr_sd[k].shape}")
                    continue
                elif v.shape == curr_sd[k].shape:
                    load_sd[k] = v
                else:
                    # [修复] 增加致命警告，不再静默吞咽 shape mismatch
                    print(f"Error: Weight dimension mismatch was discarded! Layer name: {k} (pretrained: {v.shape}, current model: {curr_sd[k].shape})")

            self.ast.load_state_dict(load_sd, strict=False)
            if verbose:
                print(f"AST Backbone 权重加载完成 ({len(load_sd)}/{len(curr_sd)})")
        except Exception as e:
            if verbose: print(f"权重加载失败: {e}")

    def forward(self, x):
        features = self.ast(x)['embedding']
        return self.classifier(features)
