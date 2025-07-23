# utils/model/prompt_varnet.py
import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fastmri.data import transforms

from unet import Unet  # 기존 unet.py 재사용
from utils.common.utils import center_crop

# [PROMPT-MR] PromptUnet의 기반이 되는 NormUnet 수정
class NormUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        prompt_embed_dim: int = 0, # 프롬프트 임베딩 차원
        mask_embed_dim: int = 0,   # 마스크 임베딩 차원
    ):
        super().__init__()
        self.prompt_embed_dim = prompt_embed_dim
        self.mask_embed_dim = mask_embed_dim
        
        # Unet에 들어갈 최종 입력 채널 수 계산
        total_in_chans = in_chans + prompt_embed_dim + mask_embed_dim

        self.unet = Unet(
            in_chans=total_in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

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
        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / (std + 1e-6), mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor, h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, mask_prompt: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        # [PROMPT-MR] 프롬프트 주입
        if self.prompt_embed_dim > 0 and prompt is not None:
            # prompt: (B, prompt_dim) -> (B, prompt_dim, H, W)
            prompt = prompt.view(prompt.shape[0], self.prompt_embed_dim, 1, 1).expand(
                -1, -1, x.shape[2], x.shape[3]
            )
            x = torch.cat([x, prompt], dim=1)
        
        if self.mask_embed_dim > 0 and mask_prompt is not None:
            # mask_prompt: (B, mask_dim, H, W)
            mask_prompt = F.interpolate(mask_prompt, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, mask_prompt], dim=1)

        x = self.unet(x)
        
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x

# 기존 SensitivityModel은 NormUnet을 사용하므로, 프롬프트를 받도록 수정해야 하지만,
# 민감도 맵 추정은 도메인에 크게 의존하지 않는다고 가정하고 여기서는 프롬프트 없이 진행합니다.
# 만약 필요하다면 SensitivityModel의 forward에도 prompt를 전달하도록 수정할 수 있습니다.
class SensitivityModel(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        # 여기의 NormUnet은 프롬프트 기능을 사용하지 않음
        self.norm_unet = NormUnet(chans, num_pools, in_chans, out_chans, drop_prob)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(2 * torch.min(left, right), torch.ones_like(left))
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)
        # 프롬프트 없이 NormUnet 호출
        x = self.norm_unet(x, prompt=None, mask_prompt=None)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        return x

# [PROMPT-MR] VarNetBlock 수정
class VarNetBlock(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        prompt: torch.Tensor,
        mask_prompt: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        
        # [PROMPT-MR] self.model 호출 시 프롬프트 전달
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps), prompt, mask_prompt), sens_maps
        )
        return current_kspace - soft_dc - model_term

# [PROMPT-MR] VarNet -> PromptVarNet으로 변경 및 수정
class PromptVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        use_checkpoint: bool = False,
        num_domains: int = 4,
        prompt_embed_dim: int = 128,
        mask_embed_dim: int = 16,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_domains = num_domains
        self.prompt_embed_dim = prompt_embed_dim
        self.mask_embed_dim = mask_embed_dim

        self.sens_net = SensitivityModel(sens_chans, sens_pools)

        # [PROMPT-MR] 도메인 임베딩 레이어
        if self.prompt_embed_dim > 0:
            self.domain_embedding = nn.Embedding(num_domains, prompt_embed_dim)
        
        # [PROMPT-MR] 마스크 임베딩 레이어
        if self.mask_embed_dim > 0:
            self.mask_embedding_conv = nn.Conv2d(1, mask_embed_dim, kernel_size=3, padding=1, stride=2)

        # [PROMPT-MR] Cascade의 NormUnet에 프롬프트 차원 전달
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools, 
                                  prompt_embed_dim=prompt_embed_dim, 
                                  mask_embed_dim=mask_embed_dim)) 
             for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, domain_indices: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        
        # [PROMPT-MR] 프롬프트 생성
        prompt = None
        if self.prompt_embed_dim > 0 and domain_indices is not None:
            prompt = self.domain_embedding(domain_indices)
        
        mask_prompt = None
        if self.mask_embed_dim > 0 and mask is not None:
            # mask: (B, 1, 1, W, 2) -> (B, 1, H, W)
            # 여기서는 마스크의 phase가 없다고 가정하고 real 파트만 사용
            mask_real = mask[..., 0].squeeze(1)
            mask_prompt = self.mask_embedding_conv(mask_real)

        for cascade in self.cascades:
            if self.use_checkpoint:
                kspace_pred = checkpoint(
                    cascade,
                    kspace_pred, masked_kspace, mask, sens_maps, prompt, mask_prompt,
                    use_reentrant=False,
                )
            else:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps, prompt, mask_prompt)

        # 최종 결과는 기존과 동일하게 RSS로 재구성
        result_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        
        # fastmri 챌린지는 보통 384x384 또는 320x320으로 평가하므로 crop 추가
        h, w = result_img.shape[-2:]
        if h > 384 or w > 384:
            result = center_crop(result_img, (384, 384))
        else: # brain의 경우 320x320
            result = center_crop(result_img, (320, 320))

        return result
