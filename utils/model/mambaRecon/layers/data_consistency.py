# mambaRecon/layers/data_consistency.py
from ..common.fft import fft2c, ifft2c
from einops import rearrange
from .patch import PatchEmbed2D
import torch, torch.nn as nn
import torch.nn.functional as F
# DataConsistency · Unpatchify 정의 그대로 옮기되, relative import만 수정



class Unpatchify(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.layer = nn.Linear(dim, 2 * dim_scale**2, bias=False)

    def forward(self, x):

        x = self.layer(x)
        _, _, _, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        return x.permute(0, 3, 1, 2)


class DataConsistency(nn.Module):
    def __init__(self, num_of_feat_maps, patchify, patch_size=2):
        super(DataConsistency, self).__init__()
        self.unpatchify = Unpatchify(num_of_feat_maps, dim_scale=patch_size)
        self.activation = nn.SiLU()
        self.patchify = patchify
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=2, embed_dim=num_of_feat_maps, norm_layer=nn.LayerNorm)

    def data_cons_layer(self, im, mask, zero_fill, coil_map):
        # im_complex = im[:,0,:,:] + 1j * im[:,1,:,:]
        # zero_fill_complex = zero_fill[:,0,:,:] + 1j * zero_fill[:,1,:,:]
        # zero_fill_complex_coil_sep = torch.tile(zero_fill_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        # im_complex_coil_sep = torch.tile(im_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        
        # -- 1. 실·허수 → 복소수 ------------------------------------------
        im_complex        = im[:, 0] + 1j * im[:, 1]          # (B, H, W)
        zero_fill_complex = zero_fill[:, 0] + 1j * zero_fill[:, 1]

        # coil_map: (B, Nc, H, W, 2) → (B, Nc, H, W) complex
        if coil_map.dim() == 5 and coil_map.size(-1) == 2:
            coil_map = coil_map[..., 0] + 1j * coil_map[..., 1]

        # mask: (B, 1, 1, W, 1) → (B, H, W)   (H = im.shape[-2])
        if mask.dim() == 5:
            mask = mask[:, 0, 0, :, 0]                     # (B, W)
            mask = mask.unsqueeze(1).expand(-1, im.shape[-2], -1)  # (B, H, W)

        # -- 2. 데이터 일관성 --------------------------------------------
        zero_fill_complex_coil_sep = (
            zero_fill_complex.unsqueeze(1).expand(-1, coil_map.shape[1], -1, -1)
            * coil_map
        )
        im_complex_coil_sep = (
            im_complex.unsqueeze(1).expand_as(coil_map) * coil_map
        )
      
        
        actual_kspace = fft2c(zero_fill_complex_coil_sep)
        gen_kspace = fft2c(im_complex_coil_sep)
        # mask_bool = mask>0
        # mask_coil_sep = torch.tile(mask_bool, dims=[1,coil_map.shape[1],1,1])

        mask_bool = mask.bool()
        mask_coil_sep = mask_bool.unsqueeze(1).expand_as(actual_kspace)

        gen_kspace_dc = torch.where(mask_coil_sep, actual_kspace, gen_kspace)
        gen_im = torch.sum(ifft2c(gen_kspace_dc) * torch.conj(coil_map), dim=1)
        gen_im_return = torch.stack([torch.real(gen_im), torch.imag(gen_im)], dim=1)
        return gen_im_return.type(im.dtype)
    
    def forward(self, x, zero_fill, mask, coil_map):
        h = self.unpatchify(x)  
        h = self.data_cons_layer(h, mask, zero_fill, coil_map)
        if self.patchify:
            h = self.activation(h)
            h = self.patch_embed(h)
            return x + h
        else:
            return h


