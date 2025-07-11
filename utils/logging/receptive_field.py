# 파일 경로: utils/logging/receptive_field.py

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

def _create_erf_figure(grad_map: np.ndarray, epoch: int):
    """NumPy 그래디언트 맵을 Matplotlib Figure 객체로 변환합니다."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 그래디언트 값의 편차가 매우 크므로, Log 스케일로 시각화하여 디테일을 살립니다.
    im = ax.imshow(
        grad_map,
        cmap='viridis',
        norm=LogNorm(vmin=np.min(grad_map[grad_map > 0]), vmax=np.max(grad_map))
    )
    
    ax.set_title(f'Effective Receptive Field (Avg. over Val Set)\nEpoch: {epoch}')
    ax.set_xlabel('k-space (x-axis)')
    ax.set_ylabel('k-space (y-axis)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    
    return fig


# ┕ [수정] @torch.no_grad() 데코레이터를 삭제합니다.
def log_receptive_field(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, epoch: int, device: torch.device):
    """
    모델의 유효 수용 영역(ERF)을 계산하고 W&B에 로깅합니다.
    """
    if not (wandb and wandb.run):
        return

    model.eval()
    
    _, sample_kspace, _, _, _, _, _ = next(iter(data_loader))
    H, W = sample_kspace.shape[2:4]
    total_grad_map = torch.zeros((H, W), dtype=torch.float32, device='cpu')
    num_slices = 0

    pbar = tqdm(data_loader, desc='Calculating ERF', leave=False, ncols=70)
    for mask, kspace, _, _, _, _, _ in pbar:
        kspace = kspace.to(device)
        mask = mask.to(device)
        
        # ┕ [수정] with torch.enable_grad() 블록으로 그래디언트 계산이 필요한 부분을 명시적으로 감쌉니다.
        with torch.enable_grad():
            kspace.requires_grad_(True)
            output = model(kspace, mask)
            
            B, H_out, W_out = output.shape
            center_pixel_values = output[:, H_out // 2, W_out // 2]
            scalar_objective = center_pixel_values.sum()
            
            scalar_objective.backward()
        
        # ┕ [수정] 그래디언트 처리 부분은 no_grad 컨텍스트에서 안전하게 수행합니다.
        with torch.no_grad():
            if kspace.grad is not None:
                grad_magnitude = torch.sqrt(kspace.grad.pow(2).sum(dim=-1)).sum(dim=(0, 1))
                total_grad_map += grad_magnitude.cpu()
                num_slices += B

    # 나머지 로직은 동일
    if num_slices > 0:
        avg_grad_map = total_grad_map / num_slices
        avg_grad_map_np = avg_grad_map.numpy()
        fig = _create_erf_figure(avg_grad_map_np, epoch)
        wandb.log({"validation/effective_receptive_field": wandb.Image(fig)}, step=epoch)
        plt.close(fig)
