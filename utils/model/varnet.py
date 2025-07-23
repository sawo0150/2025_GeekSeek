"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple, Optional

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  
from fastmri.data import transforms

from unet import Unet
from utils.common.utils import center_crop


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
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
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
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

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2

        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)

        # convert to image space
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)

        # estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)

        return x

class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        use_checkpoint: bool = False,                  # ★ NEW 토글 파라미터
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
        self.use_checkpoint = use_checkpoint           # ★ NEW 저장

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades[:-1]:
            if self.use_checkpoint:
                kspace_pred, _ = checkpoint(
                    cascade,
                    kspace_pred,
                    masked_kspace,
                    mask,
                    sens_maps,
                    False,  # return_image
                    use_reentrant=False,
                )
            else:
                kspace_pred, _ = cascade(kspace_pred, masked_kspace, mask, sens_maps, return_image=False)

        # last cascade
        if self.use_checkpoint:
            kspace_pred, final_image = checkpoint(
                self.cascades[-1],
                kspace_pred,
                masked_kspace,
                mask,
                sens_maps,
                True,  # return_image
                use_reentrant=False,
            )
        else:
            kspace_pred, final_image = self.cascades[-1](kspace_pred, masked_kspace, mask, sens_maps, return_image=True)

        result = fastmri.complex_abs(final_image).squeeze(1)
        result = center_crop(result, 384, 384)
        return result
    

class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        return_image: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight

        # 1. k-space -> image space
        image = self.sens_reduce(current_kspace, sens_maps)  # shape: (b, 1, h, w, 2)

        # 2. 원래 차원 기억
        b, c, orig_h, orig_w, comp = image.shape
        assert comp == 2, "Last dim must be 2 for complex."
        assert c == 1, "Channel should be 1."

        # 3. 채널로 변환
        image_chan = image.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, orig_h, orig_w)  # (b, 2, orig_h, orig_w)

        # Target size
        target_h = 384
        target_w = 384

        # Compute crop amounts (for larger dims)
        crop_h_top = max(0, (orig_h - target_h) // 2)
        crop_h_bottom = max(0, orig_h - target_h - crop_h_top)
        central_h_size = orig_h - crop_h_top - crop_h_bottom

        crop_w_left = max(0, (orig_w - target_w) // 2)
        crop_w_right = max(0, orig_w - target_w - crop_w_left)
        central_w_size = orig_w - crop_w_left - crop_w_right

        # Extract central crop
        central = image_chan[
            :,
            :,
            crop_h_top : crop_h_top + central_h_size,
            crop_w_left : crop_w_left + central_w_size,
        ]  # (b, 2, central_h_size, central_w_size)

        # Interpolate the central to target size (instead of padding)
        resized_central = F.interpolate(
            central,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False,
        )  # (b, 2, target_h, target_w)

        # 4. model (NormUnet) 처리 - complex로 변환
        resized_image_complex = resized_central.view(b, 2, c, target_h, target_w).permute(0, 2, 3, 4, 1).contiguous()
        model_output = self.model(resized_image_complex)  # (b, 1, target_h, target_w, 2)

        # 5. 채널로 변환
        model_output_chan = model_output.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, target_h, target_w)  # (b, 2, target_h, target_w)

        # Interpolate back to central size (instead of unpadding)
        processed_central = F.interpolate(
            model_output_chan,
            size=(central_h_size, central_w_size),
            mode='bilinear',
            align_corners=False,
        )  # (b, 2, central_h_size, central_w_size)

        # 6. 원본 clone하고 central만 processed로 대체 (peripheral은 원본 유지)
        restored_image_chan = image_chan.clone()
        restored_image_chan[
            :,
            :,
            crop_h_top : crop_h_top + central_h_size,
            crop_w_left : crop_w_left + central_w_size,
        ] = processed_central

        # 7. complex로 변환
        restored_image = restored_image_chan.view(b, 2, c, orig_h, orig_w).permute(0, 2, 3, 4, 1).contiguous()

        # 8. image -> k-space
        model_term = self.sens_expand(restored_image, sens_maps)

        # 9. DC 적용
        updated_kspace = current_kspace - soft_dc - model_term

        if return_image:
            return updated_kspace, restored_image
        else:
            return updated_kspace, None