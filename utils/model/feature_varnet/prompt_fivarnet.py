# utils/model/feature_varnet/prompt_fivarnet.py

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import List, NamedTuple, Optional, Tuple
from fastmri.fftc import ifft2c_new as ifft2c
from fastmri.math import complex_abs
from fastmri.coil_combine import rss
from utils.common.utils import center_crop
from .utils import sens_reduce, sens_expand, FeatureImage
from .modules import FeatureEncoder, FeatureDecoder, NormStats, SensitivityModel, ConvBlock, TransposeConvBlock
from .attention import AttentionPE
from .blocks import VarNetBlock


# ----------------------------------------------------------------------
# [NEW CLASS] 1. 동적 프롬프트 생성 블록
# ----------------------------------------------------------------------
class DynamicPromptBlock(nn.Module):
    """
    U-Net의 피처맵과 전역 프롬프트를 기반으로 동적 프롬프트를 생성합니다.
    """
    def __init__(
        self,
        feature_chans: int,
        global_prompt_dim: int,
        prompt_bank_size: int = 8,
        prompt_dim_multiscale: int = 16,
    ):
        super().__init__()
        self.prompt_dim_multiscale = prompt_dim_multiscale
        self.prompt_bank = nn.Parameter(torch.randn(prompt_bank_size, prompt_dim_multiscale))
        self.feature_embed_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_chans, feature_chans),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.global_prompt_mapper = nn.Sequential(
            nn.Linear(global_prompt_dim, global_prompt_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        combined_dim = feature_chans + global_prompt_dim
        self.weight_generator = nn.Linear(combined_dim, prompt_bank_size)

    def forward(self, feature_map: Tensor, global_prompt: Tensor) -> Tensor:
        B, _, H, W = feature_map.shape
        feature_embedding = self.feature_embed_net(feature_map)
        global_prompt_embedding = self.global_prompt_mapper(global_prompt)
        context = torch.cat([feature_embedding, global_prompt_embedding], dim=1)
        prompt_weights = F.softmax(self.weight_generator(context), dim=1)
        dynamic_prompt_vector = torch.matmul(prompt_weights, self.prompt_bank)
        return dynamic_prompt_vector.view(B, self.prompt_dim_multiscale, 1, 1).expand(-1, -1, H, W)


# ----------------------------------------------------------------------
# [NEW CLASS] 2. 다중 스케일 프롬프트 주입 U-Net
# ----------------------------------------------------------------------
class PromptUnet(nn.Module):
    """
    디코더 각 층에 DynamicPromptBlock을 내장하여 다중 스케일 프롬프트를 주입하는 U-Net.
    """
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        global_prompt_dim: int = 0,
        prompt_bank_size: int = 8,
        prompt_dim_multiscale: int = 16,
    ):
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
        ch = ch * 2

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.prompt_blocks = nn.ModuleList()

        for _ in range(num_pool_layers):
            transpose_in_ch = ch
            transpose_out_ch = ch // 2
            self.up_transpose_conv.append(TransposeConvBlock(transpose_in_ch, transpose_out_ch))
            self.prompt_blocks.append(
                DynamicPromptBlock(
                    feature_chans=transpose_out_ch,
                    global_prompt_dim=global_prompt_dim,
                    prompt_bank_size=prompt_bank_size,
                    prompt_dim_multiscale=prompt_dim_multiscale
                )
            )
            conv_in_chans = transpose_out_ch + transpose_out_ch + prompt_dim_multiscale
            if _ == num_pool_layers - 1:
                self.up_conv.append(
                    nn.Sequential(
                        ConvBlock(conv_in_chans, transpose_out_ch, drop_prob),
                        nn.Conv2d(transpose_out_ch, self.out_chans, kernel_size=1, stride=1),
                    )
                )
            else:
                self.up_conv.append(ConvBlock(conv_in_chans, transpose_out_ch, drop_prob))
            ch //= 2

    def forward(self, image: torch.Tensor, global_prompt: Optional[Tensor] = None) -> torch.Tensor:
        stack = []
        output = image
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
        output = self.conv(output)
        for i, (transpose_conv, conv) in enumerate(zip(self.up_transpose_conv, self.up_conv)):
            skip_connection = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != skip_connection.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != skip_connection.shape[-2]:
                padding[3] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")
            if global_prompt is not None:
                dynamic_prompt = self.prompt_blocks[i](output, global_prompt)
                output = torch.cat([output, skip_connection, dynamic_prompt], dim=1)
            else:
                output = torch.cat([output, skip_connection], dim=1)
            output = conv(output)
        return output


# ----------------------------------------------------------------------
# 3. 새로운 모듈을 사용하도록 기존 블록 및 메인 모델 수정
# ----------------------------------------------------------------------
class PromptAttentionFeatureVarNetBlock(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        feature_processor: PromptUnet,
        attention_layer: AttentionPE,
        use_extra_feature_conv: bool = False,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.attention_layer = attention_layer
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))
        feature_chans = self.encoder.feature_chans
        self.input_norm = nn.InstanceNorm2d(feature_chans)
        self.kspace_mult_factor = kspace_mult_factor
        if use_extra_feature_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(feature_chans, feature_chans, 5, padding=2, bias=False),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_chans, feature_chans, 5, padding=2, bias=False),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:
        image = sens_reduce(kspace, feature_image.sens_maps)
        return self.encoder(image, means=feature_image.means, variances=feature_image.variances)

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(feature_image.features, means=feature_image.means, variances=feature_image.variances)
        return sens_expand(image, feature_image.sens_maps)

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)
        return self.dc_weight * self.encode_from_kspace(
            torch.where(feature_image.mask.bool(), est_kspace - feature_image.ref_kspace, self.zero),
            feature_image,
        )

    def forward(self, feature_image: FeatureImage, prompt: Optional[Tensor] = None) -> FeatureImage:
        fi_normed = feature_image._replace(features=self.input_norm(feature_image.features))
        dc_term = self.compute_dc_term(fi_normed)
        accel = 8 if torch.mean(feature_image.mask.float()) < 0.16 else 4
        attn_term = self.attention_layer(feature_image.features, accel)
        model_term = self.feature_processor(attn_term, prompt)
        new_features = fi_normed.features - dc_term - model_term
        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)
        return feature_image._replace(features=new_features)

class DynamicPromptVarNetBlock(VarNetBlock):
    def __init__(self, model: nn.Module):
        super().__init__(model)
    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        prompt: Optional[Tensor] = None,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps), prompt),
            sens_maps
        )
        return current_kspace - soft_dc - model_term

class NormWrapper(nn.Module):
    def __init__(self, model: PromptUnet):
        super().__init__()
        self.model = model
    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)
    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    def _calc_norm_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        flattened = x.reshape(B, C, -1)
        mean = flattened.mean(dim=2).view(B, C, 1, 1)
        std = flattened.std(dim=2).view(B, C, 1, 1)
        return mean, std + 1e-6
    def _apply_norm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean) / std
    def _apply_unnorm(self, x: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        mean, std = self._calc_norm_params(x_orig)
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
    def forward(self, x_complex: torch.Tensor, prompt: Optional[Tensor] = None) -> torch.Tensor:
        if not x_complex.shape[-1] == 2: raise ValueError("Last dimension must be 2 for complex.")
        x_chan = self.complex_to_chan_dim(x_complex)
        mean, std = self._calc_norm_params(x_chan)
        x_normed = self._apply_norm(x_chan, mean, std)
        x_padded, pad_sizes = self.pad(x_normed)
        x_unet_out = self.model(x_padded, prompt)
        x_unpadded = self.unpad(x_unet_out, *pad_sizes)
        x_unnormed = self._apply_unnorm(x_unpadded, x_chan)
        return self.chan_complex_to_last_dim(x_unnormed)


# ----------------------------------------------------------------------
# 4. 메인 PromptFIVarNet 클래스 수정
# ----------------------------------------------------------------------
class PromptFIVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 6,
        num_image_cascades: int = 0,
        chans: int = 9,
        pools: int = 4,
        sens_chans: int = 8,
        sens_pools: int = 4,
        mask_center: bool = True,
        num_domains: int = 2,
        domain_embed_dim: int = 32,
        num_acc: int = 2,
        acc_embed_dim: int = 32,
        prompt_bank_size: int = 8,
        prompt_dim_multiscale: int = 16,
        use_checkpoint: bool = False,
        kspace_mult_factor: float = 1e6,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.kspace_mult_factor = kspace_mult_factor
        self.sens_net = SensitivityModel(chans=sens_chans, num_pools=sens_pools, mask_center=mask_center)
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        global_prompt_dim = domain_embed_dim + acc_embed_dim

        self.feat_cascades = nn.ModuleList(
            [ PromptAttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    feature_processor=PromptUnet(
                        in_chans=chans, out_chans=chans, chans=chans, num_pool_layers=pools,
                        global_prompt_dim=global_prompt_dim,
                        prompt_bank_size=prompt_bank_size,
                        prompt_dim_multiscale=prompt_dim_multiscale
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    kspace_mult_factor=self.kspace_mult_factor)
              for _ in range(num_cascades) ])

        self.num_image_cascades = num_image_cascades
        if self.num_image_cascades > 0:
            self.image_cascades = nn.ModuleList(
                [DynamicPromptVarNetBlock(
                    model=NormWrapper(
                        PromptUnet(
                            in_chans=2, out_chans=2, chans=chans, num_pool_layers=pools,
                            global_prompt_dim=global_prompt_dim,
                            prompt_bank_size=prompt_bank_size,
                            prompt_dim_multiscale=prompt_dim_multiscale
                        ))) for _ in range(num_image_cascades) ])

        self.domain_embedding = nn.Embedding(num_domains, domain_embed_dim)
        self.acc_embedding = nn.Embedding(num_acc, acc_embed_dim)
        self.decode_norm = nn.InstanceNorm2d(chans)
        self.norm_fn = NormStats()

    def _decode_output(self, fi: FeatureImage) -> Tensor:
        image = self.decoder(self.decode_norm(fi.features), means=fi.means, variances=fi.variances)
        return sens_expand(image, fi.sens_maps)

    def _encode_input(self, masked_kspace: Tensor, mask: Tensor) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask)
        image = sens_reduce(masked_kspace, sens_maps)
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)
        return FeatureImage(features=features, sens_maps=sens_maps, means=means,
                            variances=variances, ref_kspace=masked_kspace, mask=mask)

    def forward(self, masked_kspace: Tensor, mask: Tensor, domain_indices: Tensor, acc_indices: Tensor) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        feature_image = self._encode_input(masked_kspace=masked_kspace, mask=mask)
        domain_prompt = self.domain_embedding(domain_indices)
        acc_prompt = self.acc_embedding(acc_indices)
        combined_prompt = torch.cat([domain_prompt, acc_prompt], dim=1)

        # [MODIFIED] 체크포인팅 로직 수정
        for cascade_block in self.feat_cascades:
            if self.use_checkpoint and self.training:
                # `block=cascade_block`으로 현재 루프의 블록을 캡처
                def run_feat_block(fi, prompt, block=cascade_block):
                    return block(fi, prompt)
                feature_image = checkpoint(run_feat_block, feature_image, combined_prompt, use_reentrant=False)
            else:
                feature_image = cascade_block(feature_image, combined_prompt)

        kspace_pred = self._decode_output(feature_image)

        # [MODIFIED] 체크포인팅 로직 수정
        if self.num_image_cascades > 0:
            for cascade_block in self.image_cascades:
                if self.use_checkpoint and self.training:
                    # `block=cascade_block`으로 현재 루프의 블록을 캡처
                    def run_image_block(current_k, ref_k, m, sm, p, block=cascade_block):
                        return block(current_k, ref_k, m, sm, p)
                    kspace_pred = checkpoint(
                        run_image_block,
                        kspace_pred,
                        masked_kspace,
                        mask,
                        feature_image.sens_maps,
                        combined_prompt,
                        use_reentrant=False
                    )
                else:
                    kspace_pred = cascade_block(
                        kspace_pred,
                        masked_kspace,
                        mask,
                        feature_image.sens_maps,
                        combined_prompt
                    )

        kspace_pred = kspace_pred / self.kspace_mult_factor
        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        img = center_crop(img, 384, 384)
        return img