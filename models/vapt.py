# -*- coding: utf-8 -*-
import os
from copy import deepcopy

from PIL.features import features
from sympy.matrices.tests.test_matrixbase import mutable_classes

os.environ['TORCH_HOME'] = './pretrained_model'
import torch
import torch.nn as nn

from .cav_mae import CAVMAEFT
import math
from functools import reduce
from operator import mul


class PromptVA(nn.Module):
    """CAV-MAE with added prompts"""
    def __init__(self, va_model: CAVMAEFT, num_prompts_a=10, num_prompts_v=10, prompt_layers=11):
        super().__init__()
        self.va_model = va_model
        self.num_prompts_a = num_prompts_a  # num of prompts for audio
        self.num_prompts_v = num_prompts_v  # num of prompts for video
        self.prompt_dim = va_model.embed_dim  # the same as the embed_dim of cavmae: 768

        # setup prompts
        self.prompt_layers = prompt_layers
        self.prompts_a = nn.Parameter(torch.zeros(prompt_layers, num_prompts_a, self.prompt_dim)) if num_prompts_a > 0 else None  # (prompt_layers, num_prompts, 768)
        self.prompts_v = nn.Parameter(torch.zeros(prompt_layers, num_prompts_v, self.prompt_dim)) if num_prompts_v > 0 else None
        # initialize
        self.reset()

    def reset(self):
        val = 1e-4
        if self.prompts_a is not None:
            nn.init.uniform_(self.prompts_a.data, -val, val)
        if self.prompts_v is not None:
            nn.init.uniform_(self.prompts_v.data, -val, val)

    def restore(self, modal):
        """for cmmtta"""
        val = 1e-4
        if modal == 'a':
            nn.init.uniform_(self.prompts_a.data, -val, val)
        if modal == 'v':
            nn.init.uniform_(self.prompts_v.data, -val, val)

    def add_prompt(self, x, modal, layer):
        """Append prompts"""
        if self.num_prompts_a > 0 and modal == 'a':
            x = torch.cat((
                self.prompts_a[layer].expand(x.shape[0], -1, -1),  # prompts: (B, num_prompts_a, 768)
                x  # 512 * patches
            ), dim=1)  # (B, 512, 768) -> (B, 512 + num_prompts_a, 768)
        if self.num_prompts_v > 0 and modal == 'v':
            x = torch.cat((
                self.prompts_v[layer].expand(x.shape[0], -1, -1),  # prompts: (B, num_prompts_v, 768)
                x  # 196 * patches
            ), dim=1)  # (B, 196, 768) -> (B, 196 + num_prompts_v, 768)

        return x

    def random_masking(self, x, mask_ratio=0.5):
        """
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # (N, len_keep)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # (N, len_keep, D)

        return x_masked

    def forward(self, a, v, mode, prompt=False, ema=False, mask=0.5):
        if prompt:  # w/ prompts
            if mode == 'all':  # for BriMPR
                return self.forward_with_prompts(a, v, mask)
            elif mode == 'multimodal':  # w/o adaptation
                return self.forward_with_prompts_test(a, v)
        elif mode == 'features':  # return features w/o prompts
            return self.layers_features(a, v)
        else:  # return logits w/o prompts
            return self.va_model(a, v, mode, test=True)

    def forward_with_prompts(self, a, v, mask):
        mask_ratio = mask

        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.va_model.patch_embed_a(a)
        a = a + self.va_model.pos_embed_a
        a = a + self.va_model.modality_a
        # mask audio
        if mask:
            a_m = self.random_masking(a, mask_ratio=mask_ratio)
        a = self.add_prompt(a, modal='a', layer=0)
        a_m = self.add_prompt(a_m, modal='a', layer=0)

        v = self.va_model.patch_embed_v(v)
        v = v + self.va_model.pos_embed_v
        v = v + self.va_model.modality_v
        # mask video
        if mask:
            v_m = self.random_masking(v, mask_ratio=mask_ratio)
        v = self.add_prompt(v, modal='v', layer=0)
        v_m = self.add_prompt(v_m, modal='v', layer=0)

        features_a, features_v = [], []
        for i, blk in enumerate(self.va_model.blocks_a):
            a, _ = blk(a)
            a_m, _ = blk(a_m)
            if i < len(self.va_model.blocks_a) - 1:
                features_a.append(self.va_model.blocks_a[i + 1].norm1(a[:, self.num_prompts_a:, :]).mean(dim=1))
            else:
                features_a.append(self.va_model.blocks_u[0].norm1(a[:, self.num_prompts_a:, :]).mean(dim=1))

            if i + 1 < self.prompt_layers:
                a = self.add_prompt(a[:, self.num_prompts_a:, :], modal='a', layer=i+1)
                a_m = self.add_prompt(a_m[:, self.num_prompts_a:, :], modal='a', layer=i+1)
        for i, blk in enumerate(self.va_model.blocks_v):
            v, _ = blk(v)
            v_m, _ = blk(v_m)
            if i < len(self.va_model.blocks_v) - 1:
                features_v.append(self.va_model.blocks_v[i + 1].norm1(v[:, self.num_prompts_v:, :]).mean(dim=1))
            else:
                features_v.append(self.va_model.blocks_u[0].norm1(v[:, self.num_prompts_v:, :]).mean(dim=1))

            if i + 1 < self.prompt_layers:
                v = self.add_prompt(v[:, self.num_prompts_v:, :], modal='v', layer=i+1)
                v_m = self.add_prompt(v_m[:, self.num_prompts_v:, :], modal='v', layer=i+1)
        a = a[:, self.num_prompts_a:, :]
        a_m = a_m[:, self.num_prompts_a:, :]
        v = v[:, self.num_prompts_v:, :]
        v_m = v_m[:, self.num_prompts_v:, :]
        features_a = torch.stack(features_a, dim=1)
        features_v = torch.stack(features_v, dim=1)

        # Multimodal
        x = torch.cat((a, v), dim=1)
        if mask:
            x_avm = torch.cat((a, v_m), dim=1)
            x_amv = torch.cat((a_m, v), dim=1)

        x = x.detach()  # stop grad
        # AV
        for blk in self.va_model.blocks_u:
            x, attn = blk(x, ft=False)
        x = self.va_model.norm(x)
        x = x.mean(dim=1)
        features_u = x.unsqueeze(1)
        x = self.va_model.mlp_head(x)
        # AVm
        for blk in self.va_model.blocks_u:
            x_avm, attn_avm = blk(x_avm, ft=False)
        x_avm = self.va_model.norm(x_avm)
        x_avm = x_avm.mean(dim=1)
        x_avm = self.va_model.mlp_head(x_avm)
        # AmV
        for blk in self.va_model.blocks_u:
            x_amv, attn_amv = blk(x_amv, ft=False)
        x_amv = self.va_model.norm(x_amv)
        x_amv = x_amv.mean(dim=1)
        x_amv = self.va_model.mlp_head(x_amv)

        # audio only
        ua = a
        for blk in self.va_model.blocks_u:
            ua, _ = blk(ua)  # note here use unified normalization
        ua = self.va_model.norm(ua)
        ua = ua.mean(dim=1)
        for blk in self.va_model.blocks_u:
            a, _ = blk(a, 'a')  # note here use modality-specific normalization
        a = self.va_model.norm_a(a)
        a = a.mean(dim=1)
        ca = (ua + a) / 2  # average the output of the two forward passes

        # video only
        uv = v
        for blk in self.va_model.blocks_u:
            uv, _ = blk(uv)  # note here use unified normalization
        uv = self.va_model.norm(uv)
        uv = uv.mean(dim=1)
        for blk in self.va_model.blocks_u:
            v, _ = blk(v, 'v')  # note here use modality-specific normalization
        v = self.va_model.norm_v(v)
        v = v.mean(dim=1)
        cv = (uv + v) / 2  # average the output of the two forward passes

        return x, attn, ca, cv, features_a, features_v, features_u, x_avm, x_amv

    def forward_with_prompts_test(self, a, v):
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.va_model.patch_embed_a(a)
        a = a + self.va_model.pos_embed_a
        a = a + self.va_model.modality_a
        a = self.add_prompt(a, modal='a', layer=0)

        v = self.va_model.patch_embed_v(v)
        v = v + self.va_model.pos_embed_v
        v = v + self.va_model.modality_v
        v = self.add_prompt(v, modal='v', layer=0)

        for i, blk in enumerate(self.va_model.blocks_a):
            a, _ = blk(a)
            if i + 1 < self.prompt_layers:
                a = self.add_prompt(a[:, self.num_prompts_a:, :], modal='a', layer=i+1)
        for i, blk in enumerate(self.va_model.blocks_v):
            v, _ = blk(v)
            if i + 1 < self.prompt_layers:
                v = self.add_prompt(v[:, self.num_prompts_v:, :], modal='v', layer=i+1)
        a = a[:, self.num_prompts_a:, :]
        v = v[:, self.num_prompts_v:, :]

        # Multimodal
        x = torch.cat((a, v), dim=1)

        # AV
        for blk in self.va_model.blocks_u:
            x, attn = blk(x, ft=False)
        x = self.va_model.norm(x).mean(dim=1)
        x = self.va_model.mlp_head(x)

        return x, attn

    def layers_features(self, a, v):
        """pre-calculate source stats (return features w/o prompts)"""
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.va_model.patch_embed_a(a)
        a = a + self.va_model.pos_embed_a
        a = a + self.va_model.modality_a

        v = self.va_model.patch_embed_v(v)
        v = v + self.va_model.pos_embed_v
        v = v + self.va_model.modality_v

        features_a, features_v = [], []
        for i, blk in enumerate(self.va_model.blocks_a):
            a, _ = blk(a)
            if i < len(self.va_model.blocks_a) - 1:
                features_a.append(self.va_model.blocks_a[i + 1].norm1(a).mean(dim=1))
            else:
                features_a.append(self.va_model.blocks_u[0].norm1(a).mean(dim=1))
        for i, blk in enumerate(self.va_model.blocks_v):
            v, _ = blk(v)
            if i < len(self.va_model.blocks_v) - 1:
                features_v.append(self.va_model.blocks_v[i + 1].norm1(v).mean(dim=1))
            else:
                features_v.append(self.va_model.blocks_u[0].norm1(v).mean(dim=1))
        features_a = torch.stack(features_a, dim=1)
        features_v = torch.stack(features_v, dim=1)

        # Multimodal
        x = torch.cat((a, v), dim=1)
        for blk in self.va_model.blocks_u:
            x, _ = blk(x, ft=False)
        features_u = self.va_model.norm(x).mean(dim=1).unsqueeze(1)

        return features_a, features_v, features_u
