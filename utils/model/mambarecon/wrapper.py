from torch import nn
from torch.utils.checkpoint import checkpoint
from fastmri.fftc import ifft2c_new as ifft2c
from fastmri.math import complex_abs
from fastmri.coil_combine import rss
from utils.common.utils import center_crop
from utils.model.mambarecon_core.mamba_unrolled import MambaUNet  # 경로 확인

class MambaRecon(nn.Module):
    def __init__(self,
                 num_cascades: int = 6,
                 kspace_mult_factor: float = 1e6,
                 use_checkpoint: bool = True,
                 **net_kwargs):
        super().__init__()
        self.net = MambaUNet(cascades=num_cascades, **net_kwargs)
        self.k_mult = kspace_mult_factor
        self.use_ckpt = use_checkpoint

    def forward(self, masked_kspace, mask,
                num_low_frequencies=None, crop_size=None):
        k = masked_kspace * self.k_mult
        if self.use_ckpt:
            k = checkpoint(self.net, k, mask)
        else:
            k = self.net(k, mask)
        img = rss(complex_abs(ifft2c(k / self.k_mult)), dim=1)
        return center_crop(img, 384, 384)
