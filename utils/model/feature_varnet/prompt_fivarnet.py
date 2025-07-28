# utils/model/feature_varnet/prompt_fivarnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple

from fastmri.fftc import ifft2c_new as ifft2c
from fastmri.math import complex_abs
from fastmri.coil_combine import rss
from utils.common.utils import center_crop

# 기존 feature_varnet의 구성요소를 그대로 가져옵니다.
from .utils import sens_reduce, sens_expand, FeatureImage
from .modules import FeatureEncoder, FeatureDecoder, NormStats, Unet2d, SensitivityModel
from .attention import AttentionPE

# ----------------------------------------------------------------------
# 1. Prompt 기능을 주입할 새로운 VarNet Block
# ----------------------------------------------------------------------
class PromptAttentionFeatureVarNetBlock(nn.Module):
    """
    기존 AttentionFeatureVarNetBlock에 Prompt 주입 기능을 추가한 버전입니다.
    """
    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        feature_processor: Unet2d,
        attention_layer: AttentionPE,
        prompt_embed_dim: int = 0, # 프롬프트 벡터의 차원
        use_extra_feature_conv: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.attention_layer = attention_layer
        self.prompt_embed_dim = prompt_embed_dim
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))
        
        feature_chans = self.encoder.feature_chans
        self.input_norm = nn.InstanceNorm2d(feature_chans)

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
        # [FIX] UserWarning을 방지하기 위해 .bool()을 사용하여 명시적으로 boolean 타입으로 변환합니다.
        return self.dc_weight * self.encode_from_kspace(
            torch.where(feature_image.mask.bool(), est_kspace - feature_image.ref_kspace, self.zero),
            feature_image,
        )

    def forward(self, feature_image: FeatureImage, prompt: Optional[Tensor] = None) -> FeatureImage:
        fi_normed = feature_image._replace(features=self.input_norm(feature_image.features))
        
        dc_term = self.compute_dc_term(fi_normed)
        
        # [FIX] mask를 .float()으로 변환하여 torch.mean() 연산을 수행합니다.
        accel = 8 if torch.mean(feature_image.mask.float()) < 0.16 else 4
        attn_term = self.attention_layer(feature_image.features, accel)
        
        features_with_prompt = attn_term
        if prompt is not None and self.prompt_embed_dim > 0:
            prompt_spatial = prompt.view(prompt.shape[0], self.prompt_embed_dim, 1, 1).expand(
                -1, -1, attn_term.shape[2], attn_term.shape[3]
            )
            features_with_prompt = torch.cat([attn_term, prompt_spatial], dim=1)
            
        model_term = self.feature_processor(features_with_prompt)

        new_features = fi_normed.features - dc_term - model_term

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)

# ----------------------------------------------------------------------
# 2. 새로운 PromptFIVarNet 메인 클래스
# ----------------------------------------------------------------------
class PromptFIVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 6, chans: int = 9, pools: int = 4,
        sens_chans: int = 8, sens_pools: int = 4, mask_center: bool = True,
        num_domains: int = 2, domain_embed_dim: int = 64,
        num_acc: int = 2, acc_embed_dim: int = 64,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.sens_net = SensitivityModel(chans=sens_chans, num_pools=sens_pools, mask_center=mask_center)
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        
        total_prompt_dim = domain_embed_dim + acc_embed_dim
        unet_in_chans = chans + total_prompt_dim
        
        self.cascades = nn.ModuleList(
            [ PromptAttentionFeatureVarNetBlock(
                    encoder=self.encoder, decoder=self.decoder,
                    feature_processor=Unet2d(in_chans=unet_in_chans, out_chans=chans, num_pool_layers=pools),
                    attention_layer=AttentionPE(in_chans=chans),
                    prompt_embed_dim=total_prompt_dim)
              for _ in range(num_cascades) ])
        
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
        feature_image = self._encode_input(masked_kspace=masked_kspace, mask=mask)
        
        domain_prompt = self.domain_embedding(domain_indices)
        acc_prompt = self.acc_embedding(acc_indices)
        combined_prompt = torch.cat([domain_prompt, acc_prompt], dim=1)

        for cascade in self.cascades:
            if self.use_checkpoint and self.training:
                feature_image = checkpoint(cascade, feature_image, combined_prompt, use_reentrant=False)
            else:
                feature_image = cascade(feature_image, combined_prompt)

        kspace_pred = self._decode_output(feature_image)
        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        img = center_crop(img, 384, 384)
        return img