
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from fastmri.coil_combine import rss, rss_complex
from fastmri.data.transforms import batched_mask_center
from fastmri.fftc import fft2c_new as fft2c
from fastmri.fftc import ifft2c_new as ifft2c
from fastmri.math import complex_abs, complex_conj, complex_mul
from utils.common.utils import center_crop

from .utils import sens_reduce, sens_expand, FeatureImage
from .modules import FeatureEncoder, FeatureDecoder, NormStats
from .blocks import FeatureVarNetBlock, AttentionFeatureVarNetBlock, VarNetBlock
from .modules import Unet2d, NormUnet, SensitivityModel
from .attention import AttentionPE
from typing import List, NamedTuple, Optional, Tuple

class FIVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        acceleration: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
        use_checkpoint: bool = False,              # ★ NEW 토글 파라미터
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.use_checkpoint = use_checkpoint       # ★ NEW 저장
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        feat_blocks = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            feat_blocks.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    acceleration=acceleration,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, num_pool_layers=pools
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    use_extra_feature_conv=use_image_conv,
                )
            )
        self.feat_cascades = nn.ModuleList(feat_blocks)
        self.feat_cascades = nn.ModuleList(feat_blocks)

        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*feat_blocks)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space

        # print(f"Memory after _encode_input: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
        # print(f"  sens_maps size: {feature_image.sens_maps.element_size() * feature_image.sens_maps.numel() / (1024**2):.2f} MB")
        # print(f"  ref_kspace size: {feature_image.ref_kspace.element_size() * feature_image.ref_kspace.numel() / (1024**2):.2f} MB")
        # print(f"  features size: {feature_image.features.element_size() * feature_image.features.numel() / (1024**2):.2f} MB")


        # feature_image = self.cascades(feature_image)
        # — feature‐space cascades (checkpoint 옵션 적용) —
        new_fi = feature_image
        for block in self.feat_cascades:
            if self.use_checkpoint:
                # 1) gradient 필요한 텐서만 꺼내기
                feats = new_fi.features

                # 2) closure 로 나머지 인자 캡처하기
                #    참고: Python의 클로저(closure)는 외부 스코프의 변수를 "캡처"하여
                #    함수 호출 시에도 해당 값을 유지하게 합니다.
                #    여기서는 FeatureImage의 변경되지 않는 멤버들을 캡처합니다.
                sens_maps = new_fi.sens_maps
                means     = new_fi.means
                variances = new_fi.variances
                mask0     = new_fi.mask
                ref_ksp   = new_fi.ref_kspace
                crop_sz   = new_fi.crop_size

                # 3) 순수 함수 정의: Tensor (features) -> Tensor (new features)
                #    checkpoint는 입력으로 받는 텐서들(여기서는 `feats`)에 대해서만
                #    autograd 연산을 추적하고 재계산을 수행합니다.
                #    캡처된 나머지 인자들은 재계산 대상이 아니므로 메모리 오버헤드를 줄입니다.
                def run_block(feats_input,
                              _block=block,  # 현재 cascade 블록 인스턴스 고정
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
                    out = _block(fi)
                    return out.features

                # 4) checkpoint 호출: `feats`에 대해서만 재계산 로직 적용
                new_feats = checkpoint(run_block, feats, use_reentrant=False)

                # 5) NamedTuple 재조합: 변경된 features와 기존 캡처된 멤버들로 새로운 FeatureImage 생성
                new_fi = FeatureImage(
                    features   = new_feats,
                    sens_maps  = sens_maps,
                    crop_size  = crop_sz,
                    means      = means,
                    variances  = variances,
                    mask       = mask0,
                    ref_kspace = ref_ksp,
                )
            else:
                new_fi = block(new_fi)

        feature_image = new_fi

        # print(f"Memory after feat_cascades: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")



        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Run E2EVN
        # for cascade in self.image_cascades:
        #     kspace_pred = cascade(
        #         kspace_pred, feature_image.ref_kspace, mask, feature_image.sens_maps
        #     )

        # — image‐space DC cascades (checkpoint 옵션 적용) —
        for block in self.image_cascades:
            if self.use_checkpoint:
                kspace_pred = checkpoint(
                    block, kspace_pred, feature_image.ref_kspace, mask, feature_image.sens_maps
                )
            else:
                kspace_pred = block(kspace_pred, feature_image.ref_kspace, mask, feature_image.sens_maps)

        # Divide with k-space factor and Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor

        img = rss(
            complex_abs(ifft2c(kspace_pred)), dim=1
        )  # Ensure kspace_pred is a Tensor
        
        # VarNet처럼 마지막에(384×384) center_crop
        img = center_crop(img, 384, 384)
        return img


class IFVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        acceleration: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
        use_checkpoint: bool = False,          # ★ NEW
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.use_checkpoint     = use_checkpoint   # ★ NEW
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    acceleration=acceleration,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, num_pool_layers=pools
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        ref_kspace: Tensor,
        sens_maps: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
    ) -> FeatureImage:
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=ref_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:

        masked_kspace = masked_kspace * self.kspace_mult_factor

        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        # Run E2EVN
        # for cascade in self.image_cascades:
        #     kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        for block in self.image_cascades:          # ★ allow ckpt
            if self.use_checkpoint:
                kspace_pred = checkpoint(
                    block, kspace_pred, masked_kspace, mask, sens_maps
                )
            else:
                kspace_pred = block(kspace_pred, masked_kspace, mask, sens_maps)

        feature_image = self._encode_input(
            masked_kspace=kspace_pred,
            ref_kspace=masked_kspace,
            sens_maps=sens_maps,
            mask=mask,
            crop_size=crop_size,
        )
        # feature_image = self.cascades(feature_image)

        # --- feature-space cascades (ckpt 지원) ---
        for block in self.cascades:
            if self.use_checkpoint:
                feature_image = checkpoint(block, feature_image) 
            else:
                feature_image = block(feature_image)
        
        kspace_pred = self._decode_output(feature_image)
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        # return rss(
        #     complex_abs(ifft2c(kspace_pred)), dim=1
        # )  # Ensure kspace_pred is a Tensor
        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        return center_crop(img, 384, 384)          # ★ VarNet 스타일 crop


class FeatureVarNet_sh_w(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
        use_checkpoint: bool = False,              # ★ NEW
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.use_checkpoint     = use_checkpoint   # ★ NEW
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                FeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, num_pool_layers=pools
                    ),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space
        # feature_image = self.cascades(feature_image)
        # --- feature-space cascades (ckpt 지원) ---
        for block in self.cascades:
            feature_image = checkpoint(block, feature_image) if self.use_checkpoint else block(feature_image)

        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        # return rss(
        #     complex_abs(ifft2c(kspace_pred)), dim=1
        # )  # Ensure kspace_pred is a Tensor
        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        return center_crop(img, 384, 384)

class FeatureVarNet_n_sh_w(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.use_checkpoint     = use_checkpoint
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                FeatureVarNetBlock(
                    encoder=FeatureEncoder(in_chans=2, feature_chans=chans),
                    decoder=FeatureDecoder(feature_chans=chans, out_chans=2),
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, num_pool_layers=pools
                    ),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space
        # feature_image = self.cascades(feature_image)
        for block in self.cascades:
            feature_image = checkpoint(block, feature_image) if self.use_checkpoint else block(feature_image)

        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        return center_crop(img, 384, 384)


class AttentionFeatureVarNet_n_sh_w(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        acceleration: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.use_checkpoint     = use_checkpoint
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        cascades = []
        for ind in range(num_cascades):
            use_image_conv = ind in self.image_conv_cascades
            cascades.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    acceleration=acceleration,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, num_pool_layers=pools
                    ),
                    attention_layer=AttentionPE(in_chans=chans),
                    use_extra_feature_conv=use_image_conv,
                )
            )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.cascades = nn.Sequential(*cascades)
        self.norm_fn = NormStats()

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(image, feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = sens_reduce(masked_kspace, sens_maps)
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor
        # Encode to features and get sensitivities
        feature_image = self._encode_input(
            masked_kspace=masked_kspace,
            mask=mask,
            crop_size=crop_size,
            num_low_frequencies=num_low_frequencies,
        )
        # Do DC in feature-space
        # feature_image = self.cascades(feature_image)

        for block in self.cascades:
            feature_image = checkpoint(block, feature_image) if self.use_checkpoint else block(feature_image)

        # Find last k-space
        kspace_pred = self._decode_output(feature_image)
        # Return Final Image
        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  # Ensure kspace_pred is a Tensor
        # return rss(
        #     complex_abs(ifft2c(kspace_pred)), dim=1
        # )  # Ensure kspace_pred is a Tensor
        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        return center_crop(img, 384, 384)

class E2EVarNet(nn.Module):
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
        mask_center: bool = True,
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
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return rss(complex_abs(ifft2c(kspace_pred)), dim=1)

