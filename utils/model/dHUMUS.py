# utils/model/dHUMUS.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import fastmri
from fastmri.data import transforms
from einops import rearrange

from utils.model.varnet import SensitivityModel
from utils.common.loss_function import SSIMLoss

# ✅ [FINAL FIX] 어떤 크기의 입력에도 대응할 수 있는 crop 또는 pad 함수
def center_crop_or_pad(data: torch.Tensor, shape: tuple[int, int]):
    """
    Applies a center crop or zero pad to obtain shape.
    Args:
        data: The input tensor to be cropped or padded.
        shape: The desired output shape.
    """
    if not (0 < shape[0] and 0 < shape[1]):
        raise ValueError("Invalid output shape.")

    # Get original shape
    h, w = data.shape[-2:]

    # Get desired shape
    target_h, target_w = shape

    # Pad or crop height
    if h < target_h:
        h_pad_top = (target_h - h) // 2
        h_pad_bottom = target_h - h - h_pad_top
        pad_h = (h_pad_top, h_pad_bottom)
    else:
        h_crop_start = (h - target_h) // 2
        data = data[..., h_crop_start:h_crop_start + target_h, :]
        pad_h = (0, 0)

    # Pad or crop width
    if w < target_w:
        w_pad_left = (target_w - w) // 2
        w_pad_right = target_w - w - w_pad_left
        pad_w = (w_pad_left, w_pad_right)
    else:
        w_crop_start = (w - target_w) // 2
        data = data[..., w_crop_start:w_crop_start + target_w]
        pad_w = (0, 0)
        
    # Apply padding
    padding = pad_w + pad_h
    if any(p > 0 for p in padding):
        data = F.pad(data, padding, "constant", 0)

    return data


# --- Helper Modules ---
class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )
    def forward(self, x): return self.layers(x)

# --- Swin Transformer Components ---
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]; attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0); attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.window_size = window_size

    def forward(self, x, H, W, shift_size):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0: x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, H_pad, W_pad, _ = x.shape
        if shift_size > 0: x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        x_windows = rearrange(x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        attn_windows = self.attn(x_windows)
        x = rearrange(attn_windows, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', h=(H_pad // self.window_size), w=(W_pad // self.window_size), p1=self.window_size, p2=self.window_size)
        if shift_size > 0: x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        if pad_h > 0 or pad_w > 0: x = x[:, :H, :W, :].contiguous()
        x = x.view(B, L, C)
        return shortcut + x + self.mlp(self.norm2(x))

# --- MUST: Multi-scale Swin Transformer ---
class MUST(nn.Module):
    def __init__(self, dim, pools, num_heads, window_size):
        super().__init__()
        self.depth = pools
        self.layers = nn.ModuleList([SwinTransformerBlock(dim=dim*(2**i), num_heads=num_heads, window_size=window_size) for i in range(pools)])
        self.downsamples = nn.ModuleList([nn.Conv2d(dim*(2**i), dim*(2**(i+1)), kernel_size=2, stride=2) for i in range(pools - 1)])
        self.upsamples = nn.ModuleList([nn.ConvTranspose2d(dim*(2**i), dim*(2**(i-1)), kernel_size=2, stride=2) for i in range(pools - 1, 0, -1)])
        self.convs = nn.ModuleList([ConvBlock(dim*(2**i), dim*(2**(i-1))) for i in range(pools - 1, 0, -1)])

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            B, C, H, W = x.shape
            x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
            shift_size = self.layers[i].window_size // 2 if i % 2 == 1 else 0
            x_reshaped = self.layers[i](x_reshaped, H, W, shift_size)
            x = rearrange(x_reshaped, 'b (h w) c -> b c h w', h=H, w=W)
            if i < self.depth - 1: skips.append(x); x = self.downsamples[i](x)
        for i in range(self.depth - 1):
            x = self.upsamples[i](x)
            skip_connection = skips[self.depth - 2 - i]
            if x.shape != skip_connection.shape: x = F.pad(x, (0, skip_connection.shape[3] - x.shape[3], 0, skip_connection.shape[2] - x.shape[2]))
            x = torch.cat([x, skip_connection], dim=1)
            x = self.convs[i](x)
        return x

# --- HMUST Block ---
class HMUST(nn.Module):
    def __init__(self, scale, chans, pools, num_heads, window_size):
        super().__init__()
        self.scale, self.chans = scale, chans
        must_pools = max(1, int(torch.log2(torch.tensor(float(scale))).item()) + 1) if scale > 1 else 1
        self.H = ConvBlock(1, chans)
        if self.scale > 1:
            self.L = nn.Conv2d(chans, chans, kernel_size=2, stride=scale)
            self.MUST = MUST(dim=chans, pools=must_pools, num_heads=num_heads, window_size=window_size)
            self.R = nn.ConvTranspose2d(chans * 2, 1, kernel_size=2, stride=scale)
        else:
            self.res_block = ConvBlock(chans, 1)

    def forward(self, x):
        h_feat = self.H(x)
        if self.scale > 1:
            l_feat = self.L(h_feat)
            d_feat = self.MUST(l_feat)
            h_downsampled = F.interpolate(h_feat, size=d_feat.shape[2:], mode='bilinear', align_corners=False)
            combined_feat = torch.cat([d_feat, h_downsampled], dim=1)
            residual = self.R(combined_feat)
        else:
            residual = self.res_block(h_feat)
        if residual.shape != x.shape:
            residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x + residual

# --- OSPN: Optimal Scale Prediction Network ---
class OSPN(nn.Module):
    def __init__(self, pu_factors, rnn_hidden_size, scale_options):
        super().__init__()
        self.pu_factors, self.scale_options = pu_factors, scale_options
        self.ssim_metric = SSIMLoss()
        self.U_layers = nn.ModuleList([nn.Linear(f*f - 1, rnn_hidden_size) for f in pu_factors if f*f-1 > 0])
        self.W_layer = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, len(scale_options))

    def forward(self, x):
        B, _, H, W = x.shape
        with torch.no_grad():
            x_norm = (x - x.amin(dim=(2,3), keepdim=True)) / (x.amax(dim=(2,3), keepdim=True) - x.amin(dim=(2,3), keepdim=True) + 1e-6)
            hidden_state = torch.zeros(B, self.W_layer.in_features, device=x.device)
            u_layer_idx = 0
            for factor in self.pu_factors:
                if factor*factor - 1 <= 0: continue
                pad_h, pad_w = (factor - H % factor) % factor, (factor - W % factor) % factor
                padded_x = F.pad(x_norm, (0, pad_w, 0, pad_h))
                unshuffled = F.pixel_unshuffle(padded_x, factor)
                ref, others = unshuffled[:, 0:1, :, :].squeeze(1), unshuffled[:, 1:, :, :]
                B_comp, N_comp, H_comp, W_comp = others.shape
                loss_vals = self.ssim_metric(ref.unsqueeze(1).repeat(1, N_comp, 1, 1).reshape(-1, H_comp, W_comp), others.reshape(-1, H_comp, W_comp), data_range=torch.ones(B_comp * N_comp, device=x.device))
                vs = (1.0 - loss_vals).reshape(B_comp, N_comp)
                U_vs = self.U_layers[u_layer_idx](vs)
                W_h = self.W_layer(hidden_state)
                hidden_state = F.relu(U_vs + W_h)
                u_layer_idx += 1
            logits = self.fc(hidden_state)
            pred_indices = torch.argmax(logits, dim=1)
            pred_indices_list = pred_indices.cpu().tolist()
            pred_scales = torch.tensor([self.scale_options[i] for i in pred_indices_list], device=x.device, dtype=torch.long)
        return pred_scales

# --- dHUMUS-Net: Main Model ---
class dHUMUSNet(nn.Module):
    def __init__(
        self, num_cascades: int, chans: int, pools: int, num_heads: int,
        window_size: int, scale_options: list, pu_factors: list,
        rnn_hidden_size: int, sens_chans: int, sens_pools: int, use_checkpoint: bool,
    ):
        super().__init__()
        print("\n--- Initializing dHUMUS-Net (Final Patched v7 - Stable) ---")
        self.use_checkpoint, self.num_cascades = use_checkpoint, num_cascades
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.ospn = OSPN(pu_factors, rnn_hidden_size, scale_options)
        self.cascades = nn.ModuleList([nn.ModuleDict({f'scale_{s}': HMUST(s, chans, pools, num_heads, window_size) for s in scale_options}) for _ in range(num_cascades)])
        self.dc_weights = nn.Parameter(torch.ones(num_cascades))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))
    
    def _cascade_forward(self, cascade_block, ospn_block, dc_weight, kspace_pred, ref_kspace, mask, sens_maps):
        image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1).unsqueeze(1)
        predicted_scales = ospn_block(image)
        output_residuals = torch.zeros_like(image)
        for scale_val in torch.unique(predicted_scales):
            indices = (predicted_scales == scale_val).nonzero(as_tuple=True)[0]
            hmust_module = cascade_block[f'scale_{scale_val.item()}']
            output_residuals[indices] = hmust_module(image[indices])
        complex_residuals = torch.stack([output_residuals.squeeze(1), torch.zeros_like(output_residuals).squeeze(1)], dim=-1)
        model_term = self.sens_expand(complex_residuals, sens_maps)
        soft_dc = torch.where(mask.to(torch.bool), kspace_pred - ref_kspace, torch.tensor(0.0, device=kspace_pred.device))
        return kspace_pred - soft_dc * dc_weight - model_term

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        for i in range(self.num_cascades):
            if self.use_checkpoint:
                kspace_pred = checkpoint(self._cascade_forward, self.cascades[i], self.ospn, self.dc_weights[i], kspace_pred, masked_kspace, mask, sens_maps, use_reentrant=False)
            else:
                kspace_pred = self._cascade_forward(self.cascades[i], self.ospn, self.dc_weights[i], kspace_pred, masked_kspace, mask, sens_maps)
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        
        # ✅ [FINAL FIX] `center_crop`을 `center_crop_or_pad`로 교체
        return center_crop_or_pad(result, (384, 384))