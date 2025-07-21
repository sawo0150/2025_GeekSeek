"""
VarNet 구성에 MambaUnrolled(패치–기반 recon 네트워크) 를
레귤러라이저로 꽂은 구현.
GPU 1080 = sm_61 환경에서도 CUDA 커널 없이 동작.
"""
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
import fastmri, fastmri.data.transforms as T
from .mamba_unrolled import MambaUnrolled                     # ↖ 리팩터링 결과
from utils.common.utils import center_crop                          # fastMRI util
from .layers.unet import SensitivityModel

class MambaVarNetBlock(nn.Module):
    def __init__(self, regularizer: nn.Module):
        super().__init__()
        self.model = regularizer
        self.dc_weight = nn.Parameter(torch.ones(1))
    def sens_expand(self, x, s):  return fastmri.fft2c(fastmri.complex_mul(x,s))
    def sens_reduce(self, x, s):  return fastmri.complex_mul(fastmri.ifft2c(x),
                                  fastmri.complex_conj(s)).sum(1,keepdim=True)
    def forward(self, cur_k, ref_k, mask, sens):
        
        """
        cur_k:  (B, Nc, H, W, 2) - current k-space estimate
        ref_k:  (B, Nc, H, W, 2) - undersampled k-space
        mask:   (B, 1, 1, W, 1)  - sampling mask
        sens:   (B, Nc, H, W, 2) - coil sensitivities
        """
        zero = torch.zeros_like(cur_k)
        # uint8 → bool 로 캐스팅해 경고 제거
        soft_dc = torch.where(mask.bool(), cur_k - ref_k, zero) * self.dc_weight

        # 1) k-space → coil-combined complex 이미지  (B, 1, H, W, 2)
        img_complex = self.sens_reduce(cur_k, sens)

        # 2) (real, imag) 채널을 Conv2d 입력 형태(B, 2, H, W)로 변환
        b, _, h, w, _ = img_complex.shape
        img = img_complex.permute(0, 4, 1, 2, 3).reshape(b, 2, h, w).contiguous()

        # 3) Mamba regularizer 실행  → (B, 2, H, W)
        reg_output = self.model(img, mask, sens)

        # 4) 다시 complex 텐서(B, 1, H, W, 2)로 변환
        reg_output_complex = (
            reg_output.permute(0, 2, 3, 1)    # (B, H, W, 2)
                      .unsqueeze(1)           # coil dim 추가 → (B, 1, H, W, 2)
                      .contiguous()
        )

        # 5) coil sensitivity 확장 후 최종 업데이트
        model_term = self.sens_expand(reg_output_complex, sens)
        return cur_k - soft_dc - model_term

class MambaVarNet(nn.Module):
    def __init__(self,
                 num_cascades: int = 12,
                 use_checkpoint: bool = False,
                 mamba_kw: dict = None):
        super().__init__()
        self.sens_net = SensitivityModel(chans=8, num_pools=4)
        reg_net = lambda: MambaUnrolled(**mamba_kw)
        self.cascades = nn.ModuleList([MambaVarNetBlock(reg_net()) for _ in range(num_cascades)])

        if mamba_kw is None:
            raise ValueError("mamba_kw must be provided")
        self.sens_net = SensitivityModel(chans=8, num_pools=4)
        reg_net = lambda: MambaUnrolled(**mamba_kw)
        self.cascades = nn.ModuleList([MambaVarNetBlock(reg_net()) for _ in range(num_cascades)])

        self.use_checkpoint = use_checkpoint
    def forward(self, masked_k, mask):
        sens = self.sens_net(masked_k, mask)
        k = masked_k.clone()
        for c in self.cascades:
            k = checkpoint(c, k, masked_k, mask, sens, use_reentrant=False) if self.use_checkpoint else \
                c(k, masked_k, mask, sens)
        img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(k)), dim=1)
        return center_crop(img, 384, 384)
