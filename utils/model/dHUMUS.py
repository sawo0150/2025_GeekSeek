# utils/model/dHUMUS.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import fastmri
from fastmri.data import transforms
from einops import rearrange

from utils.model.varnet import SensitivityModel, VarNetBlock # 기존 모듈 재사용
from torchmetrics.image import StructuralSimilarityIndexMeasure

# --- Helper Modules ---
class ConvBlock(nn.Module):
    """단순한 Conv-ReLU-Conv-ReLU 블록"""
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

    def forward(self, x):
        return self.layers(x)

class PixelUnshuffle(nn.Module):
    """Pixel Unshuffle 연산. 이미지를 여러 개의 서브 이미지로 분해"""
    def __init__(self, downscale_factor):
        super().__init__()
        self.factor = downscale_factor

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C * factor^2, H/factor, W/factor)
        return F.pixel_unshuffle(x, self.factor)

# --- Swin Transformer Components ---
# HUMUS-Net의 MUST 블록을 구현하기 위한 핵심 요소들
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x, H, W, shift_size, window_size):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # Partition windows
        x_windows = rearrange(x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c', p1=window_size, p2=window_size)
        
        attn_windows = self.attn(x_windows) # mask=attn_mask
        
        # Merge windows
        x = rearrange(attn_windows, '(b h w) (p1 p2) c -> b (h p1) (w p2) c', h=(H // window_size), w=(W // window_size), p1=window_size, p2=window_size)

        # Reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

# --- MUST: Multi-scale Swin Transformer ---
class MUST(nn.Module):
    def __init__(self, dim, pools, num_heads, window_size):
        super().__init__()
        self.depth = pools
        self.layers = nn.ModuleList()
        for i in range(pools):
            self.layers.append(SwinTransformerBlock(dim=dim * (2**i), num_heads=num_heads, window_size=window_size))
        
        self.downsamples = nn.ModuleList()
        for i in range(pools - 1):
            self.downsamples.append(nn.Conv2d(dim * (2**i), dim * (2**(i+1)), kernel_size=2, stride=2))

        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(pools - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose2d(dim * (2**i), dim * (2**(i-1)), kernel_size=2, stride=2))
            self.convs.append(ConvBlock(dim * (2**i), dim * (2**(i-1))))

    def forward(self, x):
        # x: (B, C, H, W)
        skips = []
        
        # Encoder
        for i in range(self.depth):
            B, C, H, W = x.shape
            x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
            shift_size = self.layers[i].attn.window_size // 2 if i % 2 == 1 else 0
            x_reshaped = self.layers[i](x_reshaped, H, W, shift_size, self.layers[i].attn.window_size)
            x = rearrange(x_reshaped, 'b (h w) c -> b c h w', h=H, w=W)

            if i < self.depth - 1:
                skips.append(x)
                x = self.downsamples[i](x)

        # Decoder
        for i in range(self.depth - 1):
            x = self.upsamples[i](x)
            skip_connection = skips[self.depth - 2 - i]
            x = torch.cat([x, skip_connection], dim=1)
            x = self.convs[i](x)
        
        return x

# --- HMUST: Hybrid Multi-scale Unrolled Swin Transformer Block ---
class HMUST(nn.Module):
    def __init__(self, scale, chans, pools, num_heads, window_size):
        super().__init__()
        self.scale = scale # 1, 2, 4, 8...
        must_pools = max(1, int(torch.log2(torch.tensor(scale)).item()) + 1)
        
        # H: High-dim feature extractor
        self.H = ConvBlock(1, chans)
        # L: Low-dim feature extractor
        self.L = nn.Conv2d(chans, chans, kernel_size=2, stride=scale)
        # MUST: Deep feature extractor
        self.MUST = MUST(dim=chans, pools=must_pools, num_heads=num_heads, window_size=window_size)
        # R: Reconstruction operator
        self.R = nn.ConvTranspose2d(chans * 2, 1, kernel_size=2, stride=scale)

    def forward(self, x):
        # x: (B, 1, H, W)
        h_feat = self.H(x)
        l_feat = self.L(h_feat)
        d_feat = self.MUST(l_feat)
        
        # R combines h_feat and d_feat
        h_downsampled = F.interpolate(h_feat, scale_factor=1/self.scale, mode='bilinear', align_corners=False)
        combined_feat = torch.cat([d_feat, h_downsampled], dim=1)
        
        out = self.R(combined_feat)
        return x + out # Residual connection

# --- OSPN: Optimal Scale Prediction Network ---
class OSPN(nn.Module):
    def __init__(self, pu_factors, rnn_hidden_size, scale_options):
        super().__init__()
        self.pu_factors = pu_factors
        self.scale_options = scale_options
        self.num_scales = len(scale_options)
        
        self.pixel_unshuffles = nn.ModuleList([PixelUnshuffle(f) for f in pu_factors])
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # RNN to process similarity vectors
        self.rnn = nn.RNN(input_size=1, hidden_size=rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, self.num_scales)

    def forward(self, x): # x: (B, 1, H, W)
        B = x.shape[0]
        # Normalize image for SSIM
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)
        
        sim_vectors = []
        for pu in self.pixel_unshuffles:
            unshuffled = pu(x_norm) # (B, C, H', W')
            unshuffled = rearrange(unshuffled, 'b c h w -> b c (h w)')
            
            # Simplified SSIM calculation: compare first channel with others
            s_vec = []
            for i in range(1, unshuffled.shape[1]):
                ssim_val = self.ssim_metric(unshuffled[:, 0:1], unshuffled[:, i:i+1])
                s_vec.append(ssim_val)
            sim_vectors.append(torch.stack(s_vec, dim=1))

        # Concatenate and process with RNN
        rnn_input = torch.cat(sim_vectors, dim=1).unsqueeze(-1) # (B, SeqLen, 1)
        _, h_n = self.rnn(rnn_input)
        
        # Predict scale probabilities
        logits = self.fc(h_n.squeeze(0))
        
        # Get the index of the best scale
        pred_scale_idx = torch.argmax(logits, dim=1)
        
        # Map index to actual scale value
        pred_scales = torch.tensor([self.scale_options[i] for i in pred_scale_idx], device=x.device)
        return pred_scales

# --- dHUMUS-Net: Main Model ---
class dHUMUSNet(nn.Module):
    def __init__(
        self,
        num_cascades: int,
        chans: int,
        pools: int,
        num_heads: int,
        window_size: int,
        scale_options: list,
        pu_factors: list,
        rnn_hidden_size: int,
        sens_chans: int,
        sens_pools: int,
        use_checkpoint: bool,
    ):
        super().__init__()
        
        # --- 디버깅용 하이퍼파라미터 출력 ---
        print("\n--- Initializing dHUMUS-Net ---")
        print(f"  > num_cascades: {num_cascades}")
        print(f"  > chans: {chans}, pools: {pools}, num_heads: {num_heads}")
        print(f"  > scale_options: {scale_options}")
        print(f"  > OSPN (pu_factors: {pu_factors}, rnn_hidden_size: {rnn_hidden_size})")
        print(f"  > use_checkpoint: {use_checkpoint}")
        print("-------------------------------\n")

        self.use_checkpoint = use_checkpoint
        self.num_cascades = num_cascades

        # 1. 민감도 맵 추정기 (VarNet에서 재사용)
        self.sens_net = SensitivityModel(sens_chans, sens_pools)

        # 2. OSPN (Optimal Scale Prediction Network)
        self.ospn = OSPN(pu_factors, rnn_hidden_size, scale_options)

        # 3. Cascades: 각 cascade는 스케일별 HMUST 블록을 가짐
        self.cascades = nn.ModuleList()
        for _ in range(num_cascades):
            cascade_dict = nn.ModuleDict()
            for scale in scale_options:
                cascade_dict[f'scale_{scale}'] = HMUST(
                    scale=scale, chans=chans, pools=pools, num_heads=num_heads, window_size=window_size
                )
            self.cascades.append(cascade_dict)
            
        # 데이터 일관성(DC)을 위한 가중치
        self.dc_weights = nn.Parameter(torch.ones(num_cascades))

    # --- VarNetBlock에서 가져온 유틸리티 함수들 ---
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) # Add coil dimension
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
    
    # --- Custom forward function for checkpointing ---
    def _cascade_forward(self, cascade_block, ospn_block, dc_weight, kspace_pred, ref_kspace, mask, sens_maps):
        # 1. 현재 k-space에서 이미지 복원
        image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1).unsqueeze(1)
        
        # 2. OSPN으로 최적 스케일 예측
        with torch.no_grad(): # OSPN은 학습시키지 않고 추론만 사용
            predicted_scales = ospn_block(image)
        
        # 3. Dynamic HMUST: 스케일에 맞는 HMUST 블록 선택 및 실행
        # 참고: 배치 내 모든 샘플에 동일 스케일을 적용하여 병렬 처리
        # 논문에서는 미니배치 구성 전략을 제안했지만, 여기서는 단순화를 위해 첫 샘플의 스케일을 따름
        scale = predicted_scales[0].item()
        hmust_module = cascade_block[f'scale_{scale}']
        image_residual = hmust_module(image)

        # 4. 모델 결과(regularizer term) 계산
        model_term = self.sens_expand(image_residual, sens_maps)

        # 5. 데이터 일관성(Data Consistency) 적용
        soft_dc = torch.where(mask, kspace_pred - ref_kspace, torch.tensor(0.0, device=kspace_pred.device))
        kspace_pred = kspace_pred - soft_dc * dc_weight - model_term
        
        return kspace_pred

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. 민감도 맵 추정
        sens_maps = self.sens_net(masked_kspace, mask)
        
        # 초기 k-space 예측값
        kspace_pred = masked_kspace.clone()

        # 2. Cascade 반복
        for i in range(self.num_cascades):
            if self.use_checkpoint:
                kspace_pred = checkpoint(
                    self._cascade_forward,
                    self.cascades[i],
                    self.ospn,
                    self.dc_weights[i],
                    kspace_pred,
                    masked_kspace,
                    mask,
                    sens_maps,
                    use_reentrant=False
                )
            else:
                kspace_pred = self._cascade_forward(
                    self.cascades[i], self.ospn, self.dc_weights[i],
                    kspace_pred, masked_kspace, mask, sens_maps
                )

        # 3. 최종 이미지 복원
        # k-space -> image-space -> complex_abs -> RSS
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        
        # 기존 VarNet과 출력 크기 맞추기
        result = transforms.center_crop(result, (384, 384))
        
        return result