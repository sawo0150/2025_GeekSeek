

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2)*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)

        return x
    



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
        im_complex = im[:,0,:,:] + 1j * im[:,1,:,:]
        zero_fill_complex = zero_fill[:,0,:,:] + 1j * zero_fill[:,1,:,:]
        zero_fill_complex_coil_sep = torch.tile(zero_fill_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        im_complex_coil_sep = torch.tile(im_complex.unsqueeze(1), dims=[1,coil_map.shape[1],1,1]) * coil_map
        actual_kspace = fft2c(zero_fill_complex_coil_sep)
        gen_kspace = fft2c(im_complex_coil_sep)
        mask_bool = mask>0
        mask_coil_sep = torch.tile(mask_bool, dims=[1,coil_map.shape[1],1,1])
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
