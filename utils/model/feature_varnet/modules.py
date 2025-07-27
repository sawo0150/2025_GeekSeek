import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fastmri.coil_combine import rss, rss_complex
from fastmri.data.transforms import batched_mask_center, center_crop
from fastmri.fftc import fft2c_new as fft2c
from fastmri.fftc import ifft2c_new as ifft2c

from .deformable_LKA import deformable_LKA_Attention
from .LSKA import Attention as LSKA_Attention           # ⭐️ LSKA Attention

class NormStats(nn.Module):
    # def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
    #     # group norm
    #     batch, chans, _, _ = data.shape

    #     if batch != 1:
    #         raise ValueError("Unexpected input dimensions.")

    #     data = data.view(chans, -1)

    #     mean = data.mean(dim=1)
    #     variance = data.var(dim=1, unbiased=False)

    #     assert mean.ndim == 1
    #     assert variance.ndim == 1
    #     assert mean.shape[0] == chans
    #     assert variance.shape[0] == chans

    #     return mean, variance
    
    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        # data: (B, C, H, W)
        B, C, H, W = data.shape
        # (B, C, H*W)
        flattened = data.view(B, C, -1)
        # 채널별로, 각 샘플마다 mean, var 계산 → (B, C)
        mean = flattened.mean(dim=2)
        var  = flattened.var(dim=2, unbiased=False)
        return mean, var


class FeatureEncoder(nn.Module):
    def __init__(self, in_chans: int, feature_chans: int = 32, drop_prob: float = 0.0):
        super().__init__()
        self.feature_chans = feature_chans

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=feature_chans,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
        )

    # def forward(self, image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    #     means = means.view(1, -1, 1, 1)
    #     variances = variances.view(1, -1, 1, 1)
    #     return self.encoder((image - means) * torch.rsqrt(variances))

    def forward(self, image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        # image: (B, C, H, W)
        # means, variances: (B, C)
        # -> (B, C, 1, 1) 로 바꿔서 브로드캐스트
        means = means.unsqueeze(-1).unsqueeze(-1)          # (B, C, 1, 1)
        variances = variances.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        # 정규화
        normed = (image - means) * torch.rsqrt(variances)
        return self.encoder(normed)

class FeatureDecoder(nn.Module):
    def __init__(self, feature_chans: int = 32, out_chans: int = 2):
        super().__init__()
        self.feature_chans = feature_chans

        self.decoder = nn.Conv2d(
            in_channels=feature_chans,
            out_channels=out_chans,
            kernel_size=5,
            padding=2,
            bias=True,
        )

    # def forward(self, features: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    #     means = means.view(1, -1, 1, 1)
    #     variances = variances.view(1, -1, 1, 1)
    #     return self.decoder(features) * torch.sqrt(variances) + means
    
    def forward(self, features: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        # features: (B, F, H, W)
        # means, variances: (B, out_chans)
        means = means.unsqueeze(-1).unsqueeze(-1)          # (B, out_chans, 1, 1)
        variances = variances.unsqueeze(-1).unsqueeze(-1)  # (B, out_chans, 1, 1)
        out = self.decoder(features)                       # (B, out_chans, H, W)
        return out * torch.sqrt(variances) + means


# U-Net, Unet2d, ConvBlock, TransposeConvBlock, UnetLevel 등
class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class Unet2d(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        output_bias: bool = False,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_planes = out_chans
        self.factor = 2**num_pool_layers

        # Build from the middle of the UNet outwards
        planes = 2 ** (num_pool_layers)
        layer = None
        for _ in range(num_pool_layers):
            planes = planes // 2
            layer = UnetLevel(
                layer,
                in_planes=planes * chans,
                out_planes=2 * planes * chans,
                drop_prob=drop_prob,
            )

        self.layer = UnetLevel(
            layer, in_planes=in_chans, out_planes=chans, drop_prob=drop_prob
        )

        if output_bias:
            self.final_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=chans,
                    out_channels=out_chans,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=chans,
                    out_channels=out_chans,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def pad_input_image(self, image: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # pad image if it's not divisible by downsamples
        _, _, height, width = image.shape
        pad_height = (self.factor - (height - self.factor)) % self.factor
        pad_width = (self.factor - (width - self.factor)) % self.factor
        if pad_height != 0 or pad_width != 0:
            image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")

        return image, (height, width)

    def forward(self, image: Tensor) -> Tensor:
        image, (output_y, output_x) = self.pad_input_image(image)
        return self.final_conv(self.layer(image))[:, :, :output_y, :output_x]


class UnetLevel(nn.Module):
    def __init__(
        self,
        child: Optional[nn.Module],
        in_planes: int,
        out_planes: int,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.left_block = ConvBlock(
            in_chans=in_planes, out_chans=out_planes, drop_prob=drop_prob
        )

        self.child = child

        if child is not None:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            if isinstance(child, UnetLevel):  # Ensure child is an instance of UnetLevel
                self.upsample = TransposeConvBlock(
                    in_chans=child.out_planes, out_chans=out_planes
                )
            else:
                raise TypeError("Child must be an instance of UnetLevel")

            self.right_block = ConvBlock(
                in_chans=2 * out_planes, out_chans=out_planes, drop_prob=drop_prob
            )

    def down_up(self, image: Tensor) -> Tensor:
        if self.child is None:
            raise ValueError("self.child is None, cannot call down_up.")
        downsampled = self.downsample(image)
        child_output = self.child(downsampled)
        upsampled = self.upsample(child_output)
        return upsampled

    def forward(self, image: Tensor) -> Tensor:
        image = self.left_block(image)

        if self.child is not None:
            image = self.right_block(torch.cat((image, self.down_up(image)), 1))

        return image


class ConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class NormUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):

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
        x = x.view(b, c, h * w)

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

        # attention_goes_here
        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class Norm1DUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):

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
        x = x.view(b, c, h * w)

        mean = x.mean()
        std = x.std()

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

        # attention_goes_here
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
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
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
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = torch.div(
            mask.shape[-2] - num_low_frequencies_tensor + 1, 2, rounding_mode="trunc"
        )

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = batched_mask_center(masked_kspace, pad, pad + num_low_freqs)

        # convert to image space
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class DLKAConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, 1, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_chans)
        self.lka  = deformable_LKA_Attention(out_chans)
        self.act  = nn.LeakyReLU(0.2, inplace=True)
    def forward(self,x):
        return self.act(self.lka(self.norm(self.conv(x))))

class DLKAUnet2d(Unet2d):
    """Unet2d 구현을 그대로 상속하되 ConvBlock → DLKAConvBlock 로 치환"""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # down / up 모듈 내부 ConvBlock 치환
        def _replace(module):
            for name, m in module.named_children():
                if isinstance(m, ConvBlock):
                    setattr(module, name,
                            DLKAConvBlock(m.in_chans, m.out_chans, m.drop_prob))
                else:
                    _replace(m)
        _replace(self)


class DLKADeepUnet2d(Unet2d):
    """
    DLKAConvBlock을 인코더의 Stage-2,3(E2,E3)와 병목에만 적용한 Unet2d.
    나머지 레이어(얕은 인코더·전체 디코더)는 기존 ConvBlock 사용.
    """
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 chans: int = 32,
                 num_pool_layers: int = 4,
                 drop_prob: float = 0.0,
                 output_bias: bool = False):
        super().__init__(in_chans, out_chans, chans,
                         num_pool_layers, drop_prob, output_bias)


        # ------------------------------------------------------------------ #
        # ❶ 재귀 치환 함수 – depth 를 인자로 받아 encoder 깊은 두 stage(E2,E3)와
        #    bottleneck(conv) 에 한해 ConvBlock → DLKAConvBlock 교체
        # ------------------------------------------------------------------ #
        def _replace_recursive(module: nn.Module, cur_depth: int):
            """DFS 로 Unet level 을 내려가며 left_block 만 치환.

            depth: 0 → E0, 1 → E1, 2 → E2, 3 → E3 …
            """
            # ① 현재 노드의 left_block 확인
            if hasattr(module, "left_block") and isinstance(module.left_block, ConvBlock):
                if cur_depth >= num_pool_layers - 2:           # E2, E3
                    dlka_blk = DLKAConvBlock(
                        module.left_block.in_chans,
                        module.left_block.out_chans,
                        module.left_block.drop_prob,
                    )
                    setattr(module, "left_block", dlka_blk)

            # ② child 가 있으면 계속 내려가기
            if hasattr(module, "child") and isinstance(module.child, nn.Module):
                _replace_recursive(module.child, cur_depth + 1)

        # 👉 실제 치환 실행
        _replace_recursive(self.layer, cur_depth=0)

        # # ------------------------------------------------------------------ #
        # # ❷ bottleneck(self.conv) 치환 – 항상 depth==num_pool_layers
        # # ------------------------------------------------------------------ #
        # if isinstance(self.conv, ConvBlock):
        #     self.conv = DLKAConvBlock(
        #         self.conv.in_chans, self.conv.out_chans, self.conv.drop_prob
        #     )

        # ------------------------------------------------------------------ #
        # ❸ debug 플래그 – DLKA 블록 삽입 여부 확인용
        # ------------------------------------------------------------------ #
        self._dlka_applied = any(isinstance(m, DLKAConvBlock) for m in self.modules())



# ------------------------------------------------------------------ #
# ① LSKAConvBlock - ConvBlock → Norm → LSKA → LeakyReLU
# ------------------------------------------------------------------ #
class LSKAConvBlock(nn.Module):
    """
    Conv-Norm 후 LSKA Attention(커널 k_size 기본 7) 을 적용한 경량 블록.
    기존 DLKAConvBlock 과 동일한 인터페이스로 동작한다.
    """
    def __init__(self,
                 in_chans:  int,
                 out_chans: int,
                 drop_prob: float = 0.0,
                 k_size:    int   = 7):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, 1, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_chans)
        self.attn = LSKA_Attention(out_chans, k_size)   # ⭐️ LSKA
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.attn(x)
        x = self.act(x)
        return x

# ------------------------------------------------------------------ #
# ② LKSAUnet2d – 기존 Unet2d 의 ConvBlock 을 LSKAConvBlock 으로 전부 교체
# ------------------------------------------------------------------ #
class LKSAUnet2d(Unet2d):
    """모든 ConvBlock 을 LSKAConvBlock 으로 치환한 UNet2d."""
    def __init__(self, *a, k_size: int = 7, **kw):
        super().__init__(*a, **kw)

        def _replace(module: nn.Module):
            for name, m in module.named_children():
                if isinstance(m, ConvBlock):
                    setattr(module, name,
                            LSKAConvBlock(m.in_chans,
                                          m.out_chans,
                                          m.drop_prob,
                                          k_size=k_size))
                else:
                    _replace(m)
        _replace(self)

# ------------------------------------------------------------------ #
# ③ LKSADeepUnet2d – 인코더 깊은 E2·E3 & bottleneck 만 LSKAConvBlock 적용
# ------------------------------------------------------------------ #
class LKSADeepUnet2d(Unet2d):
    """
    인코더의 Stage-2,3(E2,E3)와 bottleneck(conv) 에만 LSKAConvBlock 적용.
    나머지 레이어는 원본 ConvBlock 유지 → 연산/메모리 과다 증가 방지.
    """
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 chans: int = 32,
                 num_pool_layers: int = 4,
                 drop_prob: float = 0.0,
                 output_bias: bool = False,
                 k_size: int = 7):
        super().__init__(in_chans, out_chans, chans,
                         num_pool_layers, drop_prob, output_bias)

        def _replace_recursive(module: nn.Module, cur_depth: int):
            """
            depth: 0 → E0, 1 → E1, 2 → E2, 3 → E3 …
            """
            # ① 현재 노드의 left_block 확인
            if hasattr(module, "left_block") and isinstance(module.left_block, ConvBlock):
                if cur_depth >= num_pool_layers - 2:           # E2, E3
                    lksa_blk = LSKAConvBlock(
                        module.left_block.in_chans,
                        module.left_block.out_chans,
                        module.left_block.drop_prob,
                        k_size=k_size,
                    )
                    setattr(module, "left_block", lksa_blk)
            # ② child 가 있으면 계속 내려가기
            if hasattr(module, "child") and isinstance(module.child, nn.Module):
                _replace_recursive(module.child, cur_depth + 1)

        # 👉 실제 치환 실행
        _replace_recursive(self.layer, cur_depth=0)

                    
        self._lksa_applied = any(isinstance(m, LSKAConvBlock) for m in self.modules())
        print("self._lksa_applied : ", self._lksa_applied)
