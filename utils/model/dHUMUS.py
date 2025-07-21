import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import fastmri
from einops import rearrange
# import logging  # DEBUG: logging 모듈 임포트

from utils.model.varnet import SensitivityModel
from utils.common.loss_function import SSIMLoss

# # DEBUG: 로깅 기본 설정. INFO 레벨 이상의 로그를 터미널에 출력합니다.
# # 파일로 저장하고 싶다면 filename='debug.log' 옵션을 추가하세요.
# logging.basicConfig(
#     level=logging.WARNING,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# # DEBUG: 텐서 정보를 로깅하기 위한 헬퍼 함수
# def log_tensor_preview(name: str, tensor: torch.Tensor):
#     """Logs key tensor information without breaking tqdm."""
#     if tensor is None:
#         logging.info(f"[DEBUG] {name}: is None")
#         return
#     try:
#         # ✅ FIX: .view() -> .reshape()으로 변경하여 non-contiguous 에러 방지
#         preview = tensor.reshape(-1)[:8].detach().cpu().numpy()
#         log_message = (
#             f"[DEBUG] {name}:\n"
#             f"  shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}\n"
#             f"  preview(first 8)={preview}"
#         )
#         logging.info(log_message)
#     except Exception as e:
#         logging.error(f"Error logging tensor '{name}': {e}")


def center_crop_or_pad(data: torch.Tensor, shape: tuple[int, int]):
    h, w = data.shape[-2:]
    target_h, target_w = shape
    if h > target_h: h_start, h_end = (h - target_h) // 2, (h - target_h) // 2 + target_h
    else: h_start, h_end = 0, h
    if w > target_w: w_start, w_end = (w - target_w) // 2, (w - target_w) // 2 + target_w
    else: w_start, w_end = 0, w
    cropped_data = data[..., h_start:h_end, w_start:w_end]
    pad_h_top = (target_h - cropped_data.shape[-2]) // 2
    pad_h_bottom = target_h - cropped_data.shape[-2] - pad_h_top
    pad_w_left = (target_w - cropped_data.shape[-1]) // 2
    pad_w_right = target_w - cropped_data.shape[-1] - pad_w_left
    padding = (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)
    return F.pad(cropped_data, padding, "constant", 0) if any(p > 0 for p in padding) else cropped_data

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )
    def forward(self, x): return self.layers(x)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim, self.window_size, self.num_heads = dim, window_size, num_heads
        # Ensure dim is divisible by num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
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
        # assert L == H * W, f"[Swin-Assert] L={L} but H*W={H*W}"
        
        shortcut = x
        
        # ✅ FIX: .view() -> .reshape()으로 변경하여 non-contiguous 텐서 에러를 근본적으로 해결
        x = self.norm1(x).reshape(B, H, W, C)
        
        pad_h, pad_w = (self.window_size - H % self.window_size) % self.window_size, (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        _, H_pad, W_pad, _ = x.shape
        
        if shift_size > 0: x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        
        x_windows = rearrange(x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        attn_windows = self.attn(x_windows)
        x = rearrange(attn_windows, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', h=(H_pad // self.window_size), w=(W_pad // self.window_size), p1=self.window_size, p2=self.window_size)
        
        if shift_size > 0: x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.reshape(B, H * W, C) # 여기도 .reshape()으로 변경하여 일관성 유지
        
        return shortcut + x + self.mlp(self.norm2(x))



class MUST(nn.Module):
    def __init__(self, dim, pools, num_heads, window_size):
        super().__init__()
        self.depth = pools
        # FIX: Ensure num_heads is adapted for each level if desired, or ensure dim is always divisible
        self.layers = nn.ModuleList([
            SwinTransformerBlock(dim=dim * (2**i), num_heads=num_heads, window_size=window_size) 
            for i in range(pools)
        ])
        self.downsamples = nn.ModuleList([nn.Conv2d(dim * (2**i), dim * (2**(i+1)), 2, 2) for i in range(pools - 1)])
        self.upsamples = nn.ModuleList([nn.ConvTranspose2d(dim * (2**i), dim * (2**(i-1)), 2, 2) for i in range(pools - 1, 0, -1)])
        self.convs = nn.ModuleList([ConvBlock(dim * (2**i), dim * (2**(i-1))) for i in range(pools - 1, 0, -1)])
    
    def forward(self, x):
        # logging.info("\n===== MUST Module Start =====")
        # log_tensor_preview("MUST Initial Input", x)
        skips = []
        
        for i in range(self.depth):
            # logging.info(f"\n--- MUST Down-Path, Depth: {i} ---")
            B, C, H, W = x.shape
            # logging.info(f"  Input x shape for this depth: {(B, C, H, W)}")
            
            x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
            # log_tensor_preview(f"  [{i}] After Flatten", x_reshaped)
            
            shift_size = self.layers[i].window_size // 2 if i % 2 == 1 else 0
            x_reshaped = self.layers[i](x_reshaped, H, W, shift_size)
            # log_tensor_preview(f"  [{i}] After SwinBlock", x_reshaped)

            x = rearrange(x_reshaped, 'b (h w) c -> b c h w', h=H, w=W)
            # log_tensor_preview(f"  [{i}] After Un-flatten", x)
            
            if i < self.depth - 1:
                skips.append(x)
                x = self.downsamples[i](x)
                # log_tensor_preview(f"  [{i}] After Downsample", x)
                
        for i in range(self.depth - 1):
            # logging.info(f"\n--- MUST Up-Path, Depth: {i} ---")
            x = self.upsamples[i](x)
            skip_connection = skips[self.depth - 2 - i]
            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = F.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip_connection], dim=1)
            x = self.convs[i](x)
            # log_tensor_preview(f"  [{i}] After Up-Conv", x)
            
        # logging.info("===== MUST Module End =====")
        return x

# ... (HMUST, OSPN, and dHUMUSNet classes remain mostly the same, but they will benefit from the logging setup)
class HMUST(nn.Module):
    def __init__(self, scale, chans, pools, num_heads, window_size):
        super().__init__()
        self.scale, self.chans = scale, chans
        must_pools = max(1, int(torch.log2(torch.tensor(float(scale))).item()) + 1) if scale > 1 else 1
        self.H = ConvBlock(1, chans)
        if self.scale > 1:
            self.L = nn.Conv2d(chans, chans, 2, 2)
            self.MUST = MUST(dim=chans, pools=must_pools, num_heads=num_heads, window_size=window_size)
            self.R = nn.ConvTranspose2d(chans * 2, 1, 2, 2)
        else:
            self.res_block = ConvBlock(chans, 1)
            
    def forward(self, x):
        # 입력 x의 평균과 표준편차를 저장하여 스케일 정보를 보존
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x_norm = (x - mean) / (std + 1e-6)

        # 모델 내부는 정규화된 x_norm으로 처리
        h_feat = self.H(x_norm)
        if self.scale > 1:
            l_feat = self.L(h_feat)
            d_feat = self.MUST(l_feat)
            h_downsampled = F.interpolate(h_feat, size=d_feat.shape[2:], mode='bilinear', align_corners=False)
            combined_feat = torch.cat([d_feat, h_downsampled], dim=1)
            residual_norm = self.R(combined_feat)
        else:
            residual_norm = self.res_block(h_feat)
            
        if residual_norm.shape[-2:] != x_norm.shape[-2:]:
            residual_norm = F.interpolate(residual_norm, size=x_norm.shape[-2:], mode='bilinear', align_corners=False)

        # 정규화된 입력에 정규화된 residual을 더한 후, 원래 스케일로 복원
        output_norm = x_norm + residual_norm
        return output_norm * std + mean

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
                if factor * factor - 1 <= 0: continue
                pad_h, pad_w = (factor - H % factor) % factor, (factor - W % factor) % factor
                padded_x = F.pad(x_norm, (0, pad_w, 0, pad_h))
                unshuffled = F.pixel_unshuffle(padded_x, factor)
                ref, others = unshuffled[:, 0:1], unshuffled[:, 1:]

                B_comp, N_comp, H_comp, W_comp = others.shape
                ref_expanded = ref.expand(-1, N_comp, -1, -1)

                # ✅ FIX 1: SSIM 입력을 위해 4D -> 3D로 변경 (채널 차원 제거)
                # (B*N, 1, H, W) -> (B*N, H, W)
                ref_flat = ref_expanded.reshape(-1, H_comp, W_comp)
                others_flat = others.reshape(-1, H_comp, W_comp)

                # ✅ FIX 1.1: SSIMLoss의 반환값이 튜플이 아닐 수 있으므로 안전하게 받음
                ssim_val = self.ssim_metric(ref_flat, others_flat)
                if isinstance(ssim_val, tuple):
                    ssim_val = ssim_val[0]
                vs = (1.0 - ssim_val).reshape(B_comp, N_comp)

                U_vs = self.U_layers[u_layer_idx](vs)
                W_h = self.W_layer(hidden_state)
                hidden_state = F.relu(U_vs + W_h)
                u_layer_idx += 1

            logits = self.fc(hidden_state)
            pred_indices = torch.argmax(logits, dim=1)
            pred_scales = torch.tensor([self.scale_options[i] for i in pred_indices.tolist()], device=x.device, dtype=torch.long)
        return pred_scales


class dHUMUSNet(nn.Module):
    def __init__(
        self, num_cascades: int, chans: int, pools: int, num_heads: int,
        window_size: int, scale_options: list, pu_factors: list,
        rnn_hidden_size: int, sens_chans: int, sens_pools: int, use_checkpoint: bool,
    ):
        super().__init__()
        self.use_checkpoint, self.num_cascades = use_checkpoint, num_cascades
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.ospn = OSPN(pu_factors, rnn_hidden_size, scale_options)
        self.cascades = nn.ModuleList([nn.ModuleDict({f'scale_{s}': HMUST(s, chans, pools, num_heads, window_size) for s in scale_options}) for _ in range(num_cascades)])
        self.dc_weights = nn.Parameter(torch.ones(num_cascades))



    def sens_expand(self, x_real: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        # ✅ FIX 2: 실수(real) 입력을 복소수 형태로 변환 후, 올바른 복소수 곱셈 수행
        x_complex = torch.stack([x_real, torch.zeros_like(x_real)], dim=-1)
        x_complex_coils = fastmri.complex_mul(x_complex.unsqueeze(1), sens_maps)
        return fastmri.fft2c(x_complex_coils)

    def sens_reduce(self, kspace_coils: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        images_coils = fastmri.ifft2c(kspace_coils)
        # ✅ FIX 2: 올바른 복소수 곱셈 수행
        images_coils_reduced = fastmri.complex_mul(images_coils, fastmri.complex_conj(sens_maps))
        return torch.sum(images_coils_reduced, dim=1)

    def _cascade_forward(self, i, kspace_pred_complex, ref_kspace_complex, mask, sens_maps_complex):
        cascade_block = self.cascades[i]
        dc_weight = self.dc_weights[i]

        image_space_pred = self.sens_reduce(kspace_pred_complex, sens_maps_complex)
        image = fastmri.complex_abs(image_space_pred).unsqueeze(1)

        predicted_scales = self.ospn(image)

        # ✅ FIX 3: Residual을 담을 텐서를 복소수 형태가 아닌 실수(real) 형태로 초기화
        output_residuals = torch.zeros_like(image.squeeze(1))
        for scale_val in torch.unique(predicted_scales):
            indices = (predicted_scales == scale_val).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                hmust_module = cascade_block[f'scale_{scale_val.item()}']
                image_batch = image[indices]
                # HMUST의 출력은 (B,1,H,W) 이므로 squeeze(1)로 (B,H,W)로 만듦
                residual_pred = hmust_module(image_batch).squeeze(1)
                output_residuals[indices] = residual_pred

        # output_residuals는 실수(real) 텐서이므로 sens_expand에서 처리
        model_term = self.sens_expand(output_residuals, sens_maps_complex)

        # Soft DC
        soft_dc_term = torch.where(mask, kspace_pred_complex - ref_kspace_complex, torch.tensor(0.0, device=kspace_pred_complex.device))
        
        return kspace_pred_complex - soft_dc_term * dc_weight - model_term

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred_complex = masked_kspace.clone()

        for i in range(self.num_cascades):
            if self.use_checkpoint:
                # checkpoint는 튜플을 반환하지 않으므로, 인자 전달 방식 수정
                kspace_pred_complex = checkpoint(
                    self._cascade_forward, i, kspace_pred_complex, masked_kspace, mask, sens_maps,
                    use_reentrant=False # PyTorch 1.10+ 권장
                )
            else:
                kspace_pred_complex = self._cascade_forward(i, kspace_pred_complex, masked_kspace, mask, sens_maps)
        
        # 최종 결과 이미지 생성
        final_image_space = self.sens_reduce(kspace_pred_complex, sens_maps)
        result = fastmri.complex_abs(final_image_space)
        
        return center_crop_or_pad(result.unsqueeze(1), (384, 384)).squeeze(1)