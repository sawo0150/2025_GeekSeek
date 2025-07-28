# utils/model/feature_varnet/prompt_fivarnet.py
import torch
import math
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
# [MODIFIED] NormUnet 및 Unet import 추가 (modules.py의 NormUnet이 필요하므로)
from .modules import FeatureEncoder, FeatureDecoder, NormStats, Unet2d, SensitivityModel, NormUnet, Unet, ConvBlock, TransposeConvBlock, UnetLevel
from .attention import AttentionPE
# VarNetBlock을 가져옵니다.
from .blocks import VarNetBlock


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
        kspace_mult_factor: float = 1e6, # [NEW] kspace_mult_factor 추가
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
        self.kspace_mult_factor = kspace_mult_factor # [NEW] 인스턴스 변수로 저장

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
        # [MODIFIED] kspace를 역스케일링하여 sens_reduce에 전달
        image = sens_reduce(kspace / self.kspace_mult_factor, feature_image.sens_maps) # [NEW]
        return self.encoder(image, means=feature_image.means, variances=feature_image.variances)

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(feature_image.features, means=feature_image.means, variances=feature_image.variances)
        # [MODIFIED] 디코딩된 kspace를 스케일링하여 반환
        return sens_expand(image, feature_image.sens_maps) * self.kspace_mult_factor # [NEW]

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)
        # UserWarning을 방지하기 위해 .bool()을 사용하여 명시적으로 boolean 타입으로 변환합니다.
        # [MODIFIED] feature_image.ref_kspace도 스케일링된 상태이므로, est_kspace와 직접 비교 가능.
        # 그러나 encode_from_kspace가 다시 역스케일링하므로 est_kspace - feature_image.ref_kspace 결과를 역스케일링해야 함.
        # FeatureVarNetBlock의 compute_dc_term 로직을 따릅니다.
        # 여기서 중요한 것은 est_kspace와 feature_image.ref_kspace 모두 동일한 스케일이어야 합니다.
        # feature_image.ref_kspace는 forward 진입시 한번 스케일링된 값입니다.
        # est_kspace는 decode_to_kspace에서 스케일링되어 나오므로, 둘은 같은 스케일입니다.
        return self.dc_weight * self.encode_from_kspace(
            torch.where(feature_image.mask.bool(), est_kspace - feature_image.ref_kspace, self.zero),
            feature_image,
        )

    def forward(self, feature_image: FeatureImage, prompt: Optional[Tensor] = None) -> FeatureImage:
        fi_normed = feature_image._replace(features=self.input_norm(feature_image.features))
        dc_term = self.compute_dc_term(fi_normed)
        # mask를 .float()으로 변환하여 torch.mean() 연산을 수행합니다.
        accel = 8 if torch.mean(feature_image.mask.float()) < 0.16 else 4
        attn_term = self.attention_layer(feature_image.features, accel)
        
        # prompt를 attention_term에 결합
        features_with_prompt = attn_term
        if prompt is not None and self.prompt_embed_dim > 0:
            # attn_term의 (B, C, H, W)에서 H, W를 가져옴
            _, _, H, W = attn_term.shape
            # prompt (B, D_p)를 (B, D_p, H, W)로 확장
            prompt_spatial = prompt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            features_with_prompt = torch.cat([attn_term, prompt_spatial], dim=1)

        model_term = self.feature_processor(features_with_prompt)
        new_features = fi_normed.features - dc_term - model_term
        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)
        return feature_image._replace(features=new_features)

# [NEW CLASS] 프롬프트가 주입된 NormUnet
class PromptNormUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 1, # 복소수 입력 채널 수 (예: sens_reduce 후 1)
        out_chans: int = 1, # 복소수 출력 채널 수 (예: sens_expand 전 1)
        drop_prob: float = 0.0,
        prompt_embed_dim: int = 0, # 추가될 프롬프트 차원
    ):
        super().__init__()

        self.in_chans_complex = in_chans
        self.out_chans_complex = out_chans
        self.prompt_embed_dim = prompt_embed_dim

        # Unet에 전달될 실제 입력 채널 계산
        # complex_to_chan_dim 변환 후의 채널 (in_chans_complex * 2) + prompt 실수 채널 (prompt_embed_dim * 2)
        unet_input_chans = self.in_chans_complex * 2 + (self.prompt_embed_dim * 2 if self.prompt_embed_dim > 0 else 0)

        # Unet의 출력 채널은 항상 실수 채널 (out_chans_complex * 2)
        unet_output_chans = self.out_chans_complex * 2

        self.unet = Unet(
            in_chans=unet_input_chans,
            out_chans=unet_output_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    # --- NormUnet의 유틸리티 함수들을 여기에 복사 & 필요에 따라 수정 ---
    # 이 함수들은 특정 assert (e.g. c==1)을 포함한 utils.py의 함수들과는 다르게
    # 이 클래스 내부의 in_chans_complex, out_chans_complex를 기준으로 동작합니다.
    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        # `c`는 입력 텐서의 복소수 채널 수이므로, self.in_chans_complex와 일치해야 함.
        # assert c == self.in_chans_complex # 필요하면 추가
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        # `c`는 Unet의 실수 채널 출력을 복소수 채널로 바꾼 것이므로, self.out_chans_complex와 일치해야 함.
        # assert c == self.out_chans_complex # 필요하면 추가
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    # [MODIFIED] norm 함수: 정규화 파라미터만 계산 및 반환
    def _calc_norm_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        flattened = x.reshape(B, C, -1) # [FIX] view 대신 reshape 사용
        mean = flattened.mean(dim=2).view(B, C, 1, 1)
        std = flattened.std(dim=2).view(B, C, 1, 1)
        # 0으로 나누는 것을 방지
        std = std + 1e-6
        return mean, std

    # [MODIFIED] apply_norm 함수: 실제 정규화 적용
    def _apply_norm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean) / std

    # [MODIFIED] apply_unnorm 함수: 실제 역정규화 적용
    def _apply_unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, prompt: Optional[Tensor] = None) -> torch.Tensor:
        # x는 (B, C_in_complex, H, W, 2) 형태 (PromptVarNetBlock에서 sens_reduce 결과)

        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # 1. 복소수 텐서를 Unet 입력 실수 채널로 변환
        x_chan_dim = self.complex_to_chan_dim(x) # (B, 2 * self.in_chans_complex, H, W) -> (B, 2, H, W)

        # 2. 프롬프트 결합
        if prompt is not None and self.prompt_embed_dim > 0:
            _, _, H_spatial, W_spatial = x_chan_dim.shape
            prompt_spatial = prompt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_spatial, W_spatial)
            prompt_for_unet_input = torch.cat([prompt_spatial, torch.zeros_like(prompt_spatial)], dim=1) # (B, 2 * prompt_embed_dim, H, W)
            x_chan_dim = torch.cat([x_chan_dim, prompt_for_unet_input], dim=1) # (B, (2 * in_chans_complex) + (2 * prompt_embed_dim), H, W)

        # 3. 정규화 (Unet 입력에 대한 정규화)
        mean_in, std_in = self._calc_norm_params(x_chan_dim)
        x_normed = self._apply_norm(x_chan_dim, mean_in, std_in)

        # 4. 패딩
        x_padded, pad_sizes = self.pad(x_normed)

        # 5. Unet 통과
        x_unet_out = self.unet(x_padded) # x_unet_out 차원: (B, 2 * self.out_chans_complex, H', W') -> (B, 2, H', W')

        # 6. 역패딩
        x_unpadded = self.unpad(x_unet_out, *pad_sizes)

        # 7. 역정규화 (Unet 출력에 대한 역정규화 - 입력과 출력의 채널 수가 다름)
        # VarNet의 NormUnet은 전역적인 스케일링을 위해 입력 정규화에 사용한 mean/std로 출력도 역정규화합니다.
        # 그러나 U-Net의 입력(132채널)과 출력(2채널)의 채널 수가 다르기 때문에
        # mean_in/std_in을 직접 사용할 수 없습니다.
        # 여기서는 U-Net의 출력 채널에 해당하는 부분만 역정규화 파라미터를 사용해야 합니다.
        # 가장 일반적인 해결책은 U-Net 출력에 대해 새로운 통계를 계산하여 역정규화하는 것입니다.
        mean_out, std_out = self._calc_norm_params(x_unpadded) # [MODIFIED] U-Net 출력에 대해 새로운 통계 계산
        x_unnormed = self._apply_unnorm(x_unpadded, mean_out, std_out) # [MODIFIED] 새로운 통계로 역정규화

        # 8. 다시 복소수 마지막 차원으로 변환
        return self.chan_complex_to_last_dim(x_unnormed)


# [NEW] Image Space에서 Prompt를 적용하는 VarNet Block
class PromptVarNetBlock(VarNetBlock):
    """
    기존 VarNetBlock에 Prompt 주입 기능을 추가한 버전입니다.
    Prompt는 모델 입력(image)과 concat하여 사용합니다.
    """
    def __init__(self, model: PromptNormUnet, prompt_embed_dim: int = 0): # [MODIFIED] model 타입을 PromptNormUnet으로 명시
        super().__init__(model)
        self.prompt_embed_dim = prompt_embed_dim


    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        prompt: Optional[Tensor] = None, # [NEW] prompt 인자 추가
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        # [MODIFIED] ref_kspace와 current_kspace가 이미 kspace_mult_factor로 스케일링되었다고 가정합니다.
        # DC term은 스케일링된 K-space 차원에서 연산됩니다.
        soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight

        # sens_reduce 결과에 prompt를 결합
        # self.sens_reduce는 K-space를 이미지로, utils.py의 sens_reduce를 사용.
        # utils.sens_reduce는 K-space를 다시 1로 역스케일링하지 않습니다.
        # 즉, sens_reduce의 출력은 스케일링된 이미지입니다.
        # self.model (PromptNormUnet)은 스케일링된 이미지를 입력으로 받아 스케일링된 이미지를 반환합니다.
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps), prompt),
            sens_maps
        )

        return current_kspace - soft_dc - model_term


# ----------------------------------------------------------------------
# 2. 새로운 PromptFIVarNet 메인 클래스
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
        domain_embed_dim: int = 64,
        num_acc: int = 4,
        acc_embed_dim: int = 64,
        use_checkpoint: bool = False,
        kspace_mult_factor: float = 1e6, # [NEW] kspace_mult_factor 추가
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.kspace_mult_factor = kspace_mult_factor # [NEW] 인스턴스 변수로 저장
        self.sens_net = SensitivityModel(chans=sens_chans, num_pools=sens_pools, mask_center=mask_center)
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)

        total_prompt_dim = domain_embed_dim + acc_embed_dim
        
        # Feature Space UNet의 입력 채널
        unet_in_chans_feature = chans + total_prompt_dim
        self.feat_cascades = nn.ModuleList(
            [ PromptAttentionFeatureVarNetBlock(
                    encoder=self.encoder, decoder=self.decoder,
                    feature_processor=Unet2d(in_chans=unet_in_chans_feature, out_chans=chans, num_pool_layers=pools),
                    attention_layer=AttentionPE(in_chans=chans),
                    prompt_embed_dim=total_prompt_dim,
                    kspace_mult_factor=self.kspace_mult_factor) # [NEW] kspace_mult_factor 전달
              for _ in range(num_cascades) ])

        # [NEW] Image Space Cascades
        self.num_image_cascades = num_image_cascades
        if self.num_image_cascades > 0:
            self.image_cascades = nn.ModuleList(
                [PromptVarNetBlock(
                    # [MODIFIED] PromptNormUnet 인스턴스를 사용
                    model=PromptNormUnet(
                        chans=chans,
                        num_pools=pools,
                        in_chans=1, # [MODIFIED] sens_reduce의 출력은 복소수 1채널이므로 여기를 1로 설정
                        out_chans=1, # [MODIFIED] Unet의 출력도 복소수 1채널로 설정 (sens_expand 전)
                        prompt_embed_dim=total_prompt_dim # PromptNormUnet에 프롬프트 차원 전달
                    ),
                    prompt_embed_dim=total_prompt_dim)
                 for _ in range(num_image_cascades) ])


        self.domain_embedding = nn.Embedding(num_domains, domain_embed_dim)
        self.acc_embedding = nn.Embedding(num_acc, acc_embed_dim)
        self.decode_norm = nn.InstanceNorm2d(chans)
        self.norm_fn = NormStats()

    def _decode_output(self, fi: FeatureImage) -> Tensor:
        image = self.decoder(self.decode_norm(fi.features), means=fi.means, variances=fi.variances)
        # [MODIFIED] 여기서는 kspace_mult_factor를 적용하지 않습니다.
        # 왜냐하면 decode_to_kspace는 PromptAttentionFeatureVarNetBlock 내부에서 처리하고,
        # 거기서 kspace_mult_factor를 이미 적용했기 때문입니다.
        # 이 함수는 FeatureDecoder의 이미지를 sens_expand로 K-space로 바꾸는 역할만 합니다.
        return sens_expand(image, fi.sens_maps)

    def _encode_input(self, masked_kspace: Tensor, mask: Tensor) -> FeatureImage:
        # [MODIFIED] sens_net은 kspace_mult_factor로 스케일링된 kspace를 그대로 받습니다.
        sens_maps = self.sens_net(masked_kspace, mask)
        # [MODIFIED] sens_reduce는 kspace_mult_factor로 스케일링된 kspace를 그대로 받습니다.
        image = sens_reduce(masked_kspace, sens_maps)
        
        # [MODIFIED] norm_fn은 스케일링된 이미지에 대해 통계를 계산
        means, variances = self.norm_fn(image)
        # [MODIFIED] FeatureEncoder는 스케일링된 이미지와 통계를 받습니다.
        features = self.encoder(image, means=means, variances=variances)
        return FeatureImage(features=features, sens_maps=sens_maps, means=means,
                            variances=variances, ref_kspace=masked_kspace, mask=mask)

    def forward(self, masked_kspace: Tensor, mask: Tensor, domain_indices: Tensor, acc_indices: Tensor) -> Tensor:
        # [NEW] masked_kspace 스케일링 (전체 모델 파이프라인의 시작)
        masked_kspace = masked_kspace * self.kspace_mult_factor
        
        # Feature Image 인코딩 (스케일링된 masked_kspace 사용)
        feature_image = self._encode_input(masked_kspace=masked_kspace, mask=mask)
        
        domain_prompt = self.domain_embedding(domain_indices)
        acc_prompt = self.acc_embedding(acc_indices)
        combined_prompt = torch.cat([domain_prompt, acc_prompt], dim=1) # (B, total_prompt_dim)

        # Feature Space Cascades
        new_fi = feature_image
        for cascade in self.feat_cascades:
            if self.use_checkpoint and self.training:
                # [MODIFIED] checkpoint closure for FeatureImage and prompt
                feats = new_fi.features
                sens_maps = new_fi.sens_maps
                means = new_fi.means
                variances = new_fi.variances
                mask0 = new_fi.mask
                ref_ksp = new_fi.ref_kspace
                crop_sz = new_fi.crop_size

                def run_feat_block(feats_input,
                                   _cascade=cascade,
                                   _combined_prompt=combined_prompt, # Prompt 캡처
                                   sens_maps=sens_maps,
                                   means=means,
                                   variances=variances,
                                   mask0=mask0,
                                   ref_ksp=ref_ksp,
                                   crop_sz=crop_sz):
                    fi = FeatureImage(
                        features=feats_input,
                        sens_maps=sens_maps,
                        crop_size=crop_sz,
                        means=means,
                        variances=variances,
                        mask=mask0,
                        ref_kspace=ref_ksp,
                    )
                    out_fi = _cascade(fi, _combined_prompt)
                    return out_fi.features

                new_feats = checkpoint(run_feat_block, feats, use_reentrant=False)
                new_fi = FeatureImage(
                    features=new_feats,
                    sens_maps=sens_maps,
                    crop_size=crop_sz,
                    means=means,
                    variances=variances,
                    mask=mask0,
                    ref_kspace=ref_ksp,
                )
            else:
                new_fi = cascade(new_fi, combined_prompt)
        feature_image = new_fi

        kspace_pred = self._decode_output(feature_image) # decode_to_kspace가 kspace_mult_factor를 적용하여 반환

        # [NEW] Image Space Cascades
        if self.num_image_cascades > 0:
            current_kspace_for_image_cascades = kspace_pred # Feature space에서 예측한 kspace를 초기값으로 사용

            for cascade in self.image_cascades:
                if self.use_checkpoint and self.training:
                    # [MODIFIED] checkpoint 호출 시 전달되는 인자들을 명확히 합니다.
                    current_kspace_for_image_cascades = checkpoint(
                        cascade,
                        current_kspace_for_image_cascades,
                        masked_kspace, # ref_kspace는 원본 masked_kspace (스케일링된 상태)
                        mask,
                        feature_image.sens_maps,
                        combined_prompt, # Image cascade에도 prompt 전달
                        use_reentrant=False
                    )
                else:
                    current_kspace_for_image_cascades = cascade(
                        current_kspace_for_image_cascades,
                        masked_kspace,
                        mask,
                        feature_image.sens_maps,
                        combined_prompt # Image cascade에도 prompt 전달
                    )
            kspace_pred = current_kspace_for_image_cascades # Image cascade의 최종 출력을 사용

        # 최종 이미지 변환 전에 kspace_pred 역스케일링
        kspace_pred = kspace_pred / self.kspace_mult_factor # [NEW] K-space 예측 결과를 역스케일링

        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        img = center_crop(img, 384, 384)
        return img