
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
from .blocks import FeatureVarNetBlock, AttentionFeatureVarNetBlock, VarNetBlock, PSFVarNetBlock
from .modules import Unet2d, NormUnet, SensitivityModel, DLKAUnet2d, DLKADeepUnet2d, LKSADeepUnet2d
from .attention import AttentionPE
from typing import List, NamedTuple, Optional, Tuple
import math
import inspect        # ★ NEW

# utils/model/flexible_varnet.py
class FlexibleCascadeVarNet(nn.Module):
    """
    cascade_cfg 예시 :
      variant='psf'      => [psf, feat, img]
      variant='dlka'     => [dlka, feat, img]
      variant='psf_dlka' => [psf_dlka, dlka, feat, img]
    """
    def __init__(self,
        cascade_counts: List[int],              # [n1,n2,…]
        variant: str,
        image_conv_cascades: Optional[List[int]] = None,
        **kw, ):
        super().__init__()
        self.variant = variant


        # ── 새 하이퍼파라미터 분리 ──────────────────────────────────────
        feature_chans = kw.get("feature_chans", kw.get("chans", 18))
        unet_chans    = kw.get("unet_chans",    kw.get("chans", 32))

        enc = FeatureEncoder(in_chans=2, feature_chans=feature_chans)
        dec = FeatureDecoder(feature_chans=feature_chans, out_chans=2)


        # ───────── extra 1×1 conv 를 걸 cascade 인덱스 목록 ──────────
        if image_conv_cascades is None:
            image_conv_cascades = [
                i for i in range(sum(cascade_counts)) if i % 3 == 0
            ]
        self.image_conv_cascades = image_conv_cascades

        # ------------------------------------------------------------------ #
        #  공통 모듈 & 하이퍼파라미터
        # ------------------------------------------------------------------ #
        self.use_checkpoint     = kw.get("use_checkpoint", False)
        self.kspace_mult_factor = kw.get("kspace_mult_factor", 1e6)
        # self.crop_size          = kw.get("crop_size", (640, 640))
        # Read crop_size from config; allow tuple, list, or explicit 'none'
        self.crop_size = kw.get("crop_size", (640, 640))
        # If someone wrote crop_size: none (as a string), treat it as no cropping
        if isinstance(self.crop_size, str) and self.crop_size.lower() == "none":
            self.crop_size = None
        # Convert list to tuple for consistency
        elif isinstance(self.crop_size, list):
            self.crop_size = tuple(self.crop_size)

        # ------------------------------------------------------------- #
        #  PSF-guided deformable attention용 하이퍼파라미터
        # ------------------------------------------------------------- #
        self.psf_K      = kw.get("psf_K", 8)     # anchor 개수
        self.psf_radius = kw.get("psf_radius", 4)  # anchor 반경(pixel)

        # |PSF| 중앙피크(Δx=0)를 top-K 후보에서 제외할지 여부
        self.psf_exclude_center = kw.get("psf_exclude_center", True)

        self.sens_net   = SensitivityModel(
            chans      = kw.get("sens_chans", 8),
            num_pools  = kw.get("sens_pools", 4),
            mask_center= kw.get("mask_center", True),
        )

        self.encoder, self.decoder = enc, dec
        self.decode_norm = nn.InstanceNorm2d(enc.feature_chans)
        self.norm_fn     = NormStats()
        
        # ------------------------------------------------------------------ #
        # ❶ helper: 매 cascade 마다 **새로운** feature_processor 생성 ----------- #
        # ------------------------------------------------------------------ #
        blocks: List[nn.Module] = []
        cur_idx = 0            # 전 cascade 인덱스

        def _add(
            block_cls,
            n: int,
            fp_cls,
            fp_kw: dict,
            **extra_kw,
        ):
            """
            block_cls           : FeatureVarNetBlock / PSFVarNetBlock …
            n                   : 반복 횟수
            fp_cls, fp_kw       : feature-processor(Unet) class & kwargs
            extra_kw            : block-specific 인자 (psf_K 등)
            """
            nonlocal cur_idx
            for _ in range(n):
                use_img_conv = cur_idx in self.image_conv_cascades
                fp = fp_cls(**fp_kw)              # ☆ fresh UNet per cascade

                # 블록 생성 시 인자 구성
                init_kwargs = dict(
                    encoder=enc,
                    decoder=dec,
                    feature_processor=fp,
                    **extra_kw,
                )
                # 일부 블록만 use_extra_feature_conv 지원
                if "use_extra_feature_conv" in inspect.signature(block_cls).parameters:
                    init_kwargs["use_extra_feature_conv"] = use_img_conv
                blocks.append(block_cls(**init_kwargs))
                cur_idx += 1

        if variant=="psf":
            # _add(PSFVarNetBlock,        cascade_counts[0], feature_processor=Unet2d(**kw))
            # _add(FeatureVarNetBlock,    cascade_counts[1], feature_processor=Unet2d(**kw))
            # _add(
            #     PSFVarNetBlock, cascade_counts[0],
            #     psf_K=self.psf_K, psf_radius=self.psf_radius,
            #     feature_processor=Unet2d(
            #         in_chans=enc.feature_chans,
            #         out_chans=enc.feature_chans,
            #         num_pool_layers=kw.get("pools", 4),
            #     ),
            # )
            # _add(
            #     FeatureVarNetBlock, cascade_counts[1],
            #     feature_processor=Unet2d(
            #         in_chans=enc.feature_chans,
            #         out_chans=enc.feature_chans,
            #         num_pool_layers=kw.get("pools", 4),
            #     ),
            # )
            _add(
                PSFVarNetBlock, cascade_counts[0],
                fp_cls=Unet2d,
                fp_kw=dict(
                    in_chans=enc.feature_chans,
                    out_chans=enc.feature_chans,
                    num_pool_layers=kw.get("pools", 4),
                    chans=unet_chans,
                ),
                psf_K=self.psf_K,
                psf_radius=self.psf_radius,
            )
            _add(
                FeatureVarNetBlock, cascade_counts[1],
                fp_cls=Unet2d,
                fp_kw=dict(
                    in_chans=enc.feature_chans,
                    out_chans=enc.feature_chans,
                    num_pool_layers=kw.get("pools", 4),
                    chans=unet_chans,
                ),
            )
        elif variant=="dlka":
            # _add(FeatureVarNetBlock,    cascade_counts[0],
            #      feature_processor=DLKAUnet2d(**kw))
            # _add(FeatureVarNetBlock,    cascade_counts[1], feature_processor=Unet2d(**kw))

            # _add(
            #     FeatureVarNetBlock, cascade_counts[0],
            #     feature_processor=DLKADeepUnet2d(
            #         in_chans=enc.feature_chans,
            #         out_chans=enc.feature_chans,
            #         num_pool_layers=kw.get("pools", 4),
            #     ),
            # )
            # _add(
            #     FeatureVarNetBlock, cascade_counts[1],
            #     feature_processor=Unet2d(
            #         in_chans=enc.feature_chans,
            #         out_chans=enc.feature_chans,
            #         num_pool_layers=kw.get("pools", 4),
            #     ),
            # )
            _add(
                FeatureVarNetBlock, cascade_counts[0],
                fp_cls=LKSADeepUnet2d,
                fp_kw=dict(
                    in_chans=enc.feature_chans,
                    out_chans=enc.feature_chans,
                    num_pool_layers=kw.get("pools", 4),
                    chans=unet_chans,
                ),
            )
            _add(
                FeatureVarNetBlock, cascade_counts[1],
                fp_cls=Unet2d,
                fp_kw=dict(
                    in_chans=enc.feature_chans,
                    out_chans=enc.feature_chans,
                    num_pool_layers=kw.get("pools", 4),
                    chans=unet_chans,
                ),
            )
        elif variant=="psf_dlka":
            # _add(PSFVarNetBlock,        cascade_counts[0],
            #      feature_processor=DLKAUnet2d(**kw))
            # _add(FeatureVarNetBlock,    cascade_counts[1],
            #      feature_processor=DLKAUnet2d(**kw))
            # _add(FeatureVarNetBlock,    cascade_counts[2], feature_processor=Unet2d(**kw))
            # _add(
            #     PSFVarNetBlock, cascade_counts[0],
            #     psf_K=self.psf_K, psf_radius=self.psf_radius,
            #     feature_processor=DLKADeepUnet2d(
            #         in_chans=enc.feature_chans,
            #         out_chans=enc.feature_chans,
            #         num_pool_layers=kw.get("pools", 4),
            #     ),
            # )
            # _add(
            #     FeatureVarNetBlock, cascade_counts[1],
            #     feature_processor=DLKADeepUnet2d(
            #         in_chans=enc.feature_chans,
            #         out_chans=enc.feature_chans,
            #         num_pool_layers=kw.get("pools", 4),
            #     ),
            # )
            # _add(
            #     FeatureVarNetBlock, cascade_counts[2],
            #     feature_processor=Unet2d(
            #         in_chans=enc.feature_chans,
            #         out_chans=enc.feature_chans,
            #         num_pool_layers=kw.get("pools", 4),
            #     ),
            # )

            _add(
                PSFVarNetBlock, cascade_counts[0],
                fp_cls=DLKADeepUnet2d,
                fp_kw=dict(
                    in_chans=enc.feature_chans,
                    out_chans=enc.feature_chans,
                    num_pool_layers=kw.get("pools", 4),
                    chans=unet_chans,
                ),
                psf_K=self.psf_K,
                psf_radius=self.psf_radius,
            )
            _add(
                FeatureVarNetBlock, cascade_counts[1],
                fp_cls=DLKADeepUnet2d,
                fp_kw=dict(
                    in_chans=enc.feature_chans,
                    out_chans=enc.feature_chans,
                    num_pool_layers=kw.get("pools", 4),
                    chans=unet_chans,
                ),
            )
            _add(
                FeatureVarNetBlock, cascade_counts[2],
                fp_cls=Unet2d,
                fp_kw=dict(
                    in_chans=enc.feature_chans,
                    out_chans=enc.feature_chans,
                    num_pool_layers=kw.get("pools", 4),
                    chans=unet_chans,
                ),
            )

        img_casc = cascade_counts[-1]
        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(kw.get("chans",18), kw.get("pools",4)))
             for _ in range(img_casc)]
        )
        self.feat_cascades = nn.ModuleList(blocks)
        
        
    # ------------------------------------------------------------
    #  helper ➊ : mask → 1-D mask 벡터 추출
    # ------------------------------------------------------------
    @staticmethod
    def _extract_mask_1d(mask: torch.Tensor) -> torch.Tensor:
        """
        fastMRI mask shape 들을 모두 지원한다.
         • (B,1,1,W,1)  • (B,1,W,1)  • (B,W)  • (W,)
        반환 : (B, W) 실수 float32
        """
        if mask.ndim == 5:            # (B,1,1,W,1)
            mask_1d = mask[:, 0, 0, :, 0]
        elif mask.ndim == 4:          # (B,1,W,1)
            mask_1d = mask[:, 0, :, 0]
        elif mask.ndim == 2:          # (B,W)
            mask_1d = mask
        elif mask.ndim == 1:          # (W,)  → batch 1 로 확장
            mask_1d = mask.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported mask shape {mask.shape}")
        return mask_1d.to(torch.float32)

    # ------------------------------------------------------------
    #  helper ➋ : 1-D mask → top-K alias offset 리스트
    # ------------------------------------------------------------
    def _alias_offsets(self, mask_1d: torch.Tensor) -> torch.Tensor:
        """
        mask_1d : (B,W) → (B,K) 정수 Δx (pixel offset, -W/2 ~ W/2)
        """
        B, W = mask_1d.shape
        K    = self.psf_K
        # centered IFFT ➜ PSF
        psf = torch.fft.ifft(torch.fft.ifftshift(mask_1d, dim=-1).to(torch.complex64), dim=-1)
        mag = psf.abs()
        if self.psf_exclude_center:
            mag[..., W // 2] = 0.0
        # top-K indices (가장 큰 |PSF|)
        topk = torch.topk(mag, k=min(K, W - 1), dim=-1).indices      # (B,K)
        # offset = idx - center
        offs = topk.to(torch.int64) - (W // 2)
        # 만약 K < self.psf_K 이면 나머지는 0 으로 패딩
        if offs.shape[-1] < K:
            pad = torch.zeros(B, K - offs.shape[-1], dtype=torch.int64, device=offs.device)
            offs = torch.cat([offs, pad], dim=-1)
        return offs    # (B,K)

    # ------------------------------------------------------------
    #  helper ➌ : (B,H,W) → (B,H,W,K,2) absolute anchor coords
    # ------------------------------------------------------------
    def _make_psf_table_from_mask(
        self,
        mask:  torch.Tensor,
        H:     int,
        W:     int,
    ) -> torch.Tensor:
        """
        alias offset Δxₖ 를 각 픽셀(x,y)에 적용해 절대 좌표 테이블 생성
        dy=0 이므로 세로 alias 는 없다고 가정한다.
        """
        device = mask.device
        mask_1d = self._extract_mask_1d(mask)         # (B,W)
        dx_list = self._alias_offsets(mask_1d)        # (B,K)

        # print("mask_1d.shape : ", mask_1d.shape, "dx_list.shape : ", dx_list.shape)
        B, K = dx_list.shape
        # ys = torch.arange(H, device=device).view(1, H, 1, 1, 1)      # (1,H,1,1,1)
        # xs = torch.arange(W, device=device).view(1, 1, W, 1, 1)      # (1,1,W,1,1)
        # ── ① base grid 만들기 ──────────────────────────────────────────
        ys = torch.arange(H, device=device).view(1, H, 1, 1, 1) \
                                            .expand(B, H, W, K, 1)  # (B,H,W,K,1)
        xs = torch.arange(W, device=device).view(1, 1, W, 1, 1) \
                                            .expand(B, H, W, K, 1)  # (B,H,W,K,1)

        # (B,1,1,K,1)
        dx = dx_list.view(B, 1, 1, K, 1).expand(B, H, W, K, 1)      # (B,H,W,K,1)
        dy = torch.zeros_like(dx)

        ycoords = (ys + dy).clamp(0, H - 1)
        xcoords = (xs + dx).clamp(0, W - 1)

        tbl = torch.cat([ycoords, xcoords], dim=-1)   # (B,H,W,K,2)
        # print("tbl.shape : ", tbl.shape)                 # → torch.Size([B, H, W, K, 2])

        # b_id = 0                                         # batch index
        # y, x = H // 2, W // 2                            # 가운데 픽셀, 필요하면 변경
        # offsets = tbl[b_id, y, x]                    # (K, 2) = (y_i, x_i)
        # print(f"[debug] psf_tbl[{b_id},{y},{x}] =")
        # print(f"{offsets.cpu().numpy()}")
        # # 예)  [[384 200]  [384 196]  [384 204] ...]  ← (y, x) 좌표 K 개
        # # 필요하면 전체 테이블 shape, 일부 slice 도 추가로 확인
        # print(f"[debug] psf_tbl shape = {tbl.shape}")

        return tbl

    # ============================================================
    #  helper : k-space ↔ feature 변환
    # ============================================================
    def _decode_output(self, fi: FeatureImage) -> Tensor:
        if self.use_checkpoint:
            img = checkpoint(
                lambda f, m, v: self.decoder(f, means=m, variances=v),
                self.decode_norm(fi.features), fi.means, fi.variances,
                use_reentrant=False,
            )
        else:
            img = self.decoder(
                self.decode_norm(fi.features), means=fi.means, variances=fi.variances
            )
        return sens_expand(img, fi.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
        psf_tbl: Optional[Tensor] = None,
    ) -> FeatureImage:

        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image     = sens_reduce(masked_kspace, sens_maps)

        if crop_size is not None:
            h, w     = image.shape[-2:]
            crop_size = (min(h, crop_size[0]), min(w, crop_size[1]))

        means, vars_ = self.norm_fn(image)
        if self.use_checkpoint:
            feats = checkpoint(
                lambda img, m, v: self.encoder(img, means=m, variances=v),
                image, means, vars_, use_reentrant=False,
            )
        else:
            feats = self.encoder(image, means=means, variances=vars_)

        # ----------------------------------------------------------
        #  β(PSF table) 자동 생성  – mask 기반 alias 분석
        # ----------------------------------------------------------
        if psf_tbl is None and "psf" in self.variant:
            _, _, H, W, _ = masked_kspace.shape
            psf_tbl = self._make_psf_table_from_mask(mask, H, W)

        return FeatureImage(
            features   = feats,
            sens_maps  = sens_maps,
            crop_size  = crop_size,
            means      = means,
            variances  = vars_,
            ref_kspace = masked_kspace,
            mask       = mask,
            beta       = psf_tbl,
        )


    # ============================================================
    #  forward
    # ============================================================
    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        psf_tbl: Optional[Tensor] = None,
    ) -> Tensor:

        masked_kspace = masked_kspace * self.kspace_mult_factor

        # -------- encode ----------
        fi = self._encode_input(
            masked_kspace, mask,
            crop_size=self.crop_size,
            num_low_frequencies=num_low_frequencies,
            psf_tbl=psf_tbl,
        )

        # # -------- feature cascades ----------
        # for blk in self.feat_cascades:
        #     if self.use_checkpoint:
        #         feats = checkpoint(
        #             lambda x: blk(fi._replace(features=x)).features,
        #             fi.features, use_reentrant=False,
        #         )
        #         fi = fi._replace(features=feats)
        #     else:
        #         fi = blk(fi)

        # -------- feature cascades (checkpoint용 closure에 blk, fi 고정) ----------
        for blk in self.feat_cascades:
            if self.use_checkpoint:
                # checkpoint는 오직 x(features) 텐서만 추적 → 나머지 인자들은 default-arg로 캡처
                feats = checkpoint(
                    # x, 현재 blk, 현재 fi 를 default 인자로 바인딩
                    lambda x, _blk=blk, _fi=fi: _blk(_fi._replace(features=x)).features,
                    fi.features,
                    use_reentrant=False,
                )
                fi = fi._replace(features=feats)
            else:
                fi = blk(fi)

        # -------- decode ----------
        kspace_pred = self._decode_output(fi)

        # -------- image-space DC ----------
        for blk in self.image_cascades:
            if self.use_checkpoint:
                kspace_pred = checkpoint(
                    blk, kspace_pred, fi.ref_kspace, mask, fi.sens_maps
                )
            else:
                kspace_pred = blk(kspace_pred, fi.ref_kspace, mask, fi.sens_maps)

        # -------- final image ----------
        kspace_pred = kspace_pred / self.kspace_mult_factor
        img = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        return center_crop(img, 384, 384)
