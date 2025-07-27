# utils/model/prompt_varnet_v2.py
import torch
import torch.nn as nn
import fastmri
import math
import torch.nn.functional as F
from typing import List, Tuple

from fastmri.data import transforms
from utils.common.utils import center_crop
# [FIX] 기존 unet 대신 새로 만든 prompt_unet을 임포트합니다.
from .prompt_unet import PromptUnet

class NormPromptUnet(nn.Module):
    def __init__(self, chans, num_pools, in_chans=2, out_chans=2, prompt_embed_dim=0):
        super().__init__()
        # [FIX] Unet -> PromptUnet으로 교체
        self.unet = PromptUnet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            prompt_embed_dim=prompt_embed_dim
        )
        self.prompt_embed_dim = prompt_embed_dim

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)
        mean, std = x.mean(dim=2).view(b, c, 1, 1), x.std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / (std + 1e-6), mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult, h_mult = ((w - 1) | 15) + 1, ((h - 1) | 15) + 1
        w_pad, h_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)], [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor, h_pad, w_pad, h_mult, w_mult) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    # [FIX] forward 시그니처 변경: mask_prompt 제거, prompt만 받음
    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2: raise ValueError("Last dimension must be 2 for complex.")
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        
        # [FIX] 프롬프트를 unet에 직접 전달
        x = self.unet(x, prompt)
        
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)
        return x

class SensitivityModelV2(nn.Module):
    # This remains mostly the same, using the standard Unet for simplicity
    def __init__(self, chans, num_pools):
        super().__init__()
        from .unet import Unet # Import original Unet here
        self.norm_unet = nn.Sequential(
            nn.Conv2d(2, chans, 3, padding=1),
            Unet(in_chans=chans, out_chans=chans, chans=chans, num_pool_layers=num_pools),
            nn.Conv2d(chans, 2, 1)
        )
    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b
    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)
    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)
    def forward(self, masked_kspace, mask):
        x = fastmri.ifft2c(masked_kspace)
        x, b = self.chans_to_batch_dim(x)
        x_in = x.permute(0,4,1,2,3).reshape(x.shape[0], 2, x.shape[2], x.shape[3])
        x_out = self.norm_unet(x_in).reshape(x.shape[0], 2, 1, x.shape[2], x.shape[3]).permute(0,2,3,4,1)
        x = self.batch_chans_to_chan_dim(x_out, b)
        x = self.divide_root_sum_of_squares(x)
        return x

class VarNetBlockV2(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(self, current_kspace, ref_kspace, mask, sens_maps, prompt):
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
        
        # [FIX] self.model(NormPromptUnet) 호출 시 prompt 전달
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps), prompt), sens_maps
        )
        return current_kspace - soft_dc - model_term

class PromptVarNetV2(nn.Module):
    def __init__(self, num_cascades=8, chans=18, pools=4, sens_chans=8, sens_pools=4,
                 num_domains=2, prompt_embed_dim=128):
        super().__init__()
        self.sens_net = SensitivityModelV2(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [VarNetBlockV2(NormPromptUnet(chans, pools, prompt_embed_dim=prompt_embed_dim)) 
             for _ in range(num_cascades)]
        )
        if prompt_embed_dim > 0:
            self.domain_embedding = nn.Embedding(num_domains, prompt_embed_dim)
        self.prompt_embed_dim = prompt_embed_dim

    def forward(self, masked_kspace, mask, domain_indices):
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        
        prompt = self.domain_embedding(domain_indices) if self.prompt_embed_dim > 0 else None
        
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps, prompt)

        result_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        result = center_crop(result_img, 384, 384)
        return result