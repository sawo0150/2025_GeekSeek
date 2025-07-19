# 파일 경로: utils/logging/receptive_field.py

import torch
import numpy as np
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
    # 그래디언트가 0인 경우를 고려해 vmin에 작은 값을 더해줍니다.
    vmin = np.min(grad_map[grad_map > 0]) if np.any(grad_map > 0) else 1e-9
    im = ax.imshow(
        grad_map,
        cmap='viridis',
        norm=LogNorm(vmin=vmin, vmax=np.max(grad_map))
    )
    
    ax.set_title(f'Effective Receptive Field (from one batch)\nEpoch: {epoch}')
    ax.set_xlabel('k-space (x-axis)')
    ax.set_ylabel('k-space (y-axis)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    
    return fig

def log_receptive_field(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, epoch: int, device: torch.device):
    """
    모델의 유효 수용 영역(ERF)을 계산하고 W&B에 로깅합니다.
    데이터셋 전체를 순회하는 대신 단일 배치만 사용하여, crop이 비활성화된 경우에도 작동합니다.
    """
    if not (wandb and wandb.run):
        return

    model.eval()
    
    try:
        # 데이터로더에서 대표적인 첫 번째 배치를 가져옵니다.
        mask, kspace, _, _, _, _, _ = next(iter(data_loader))
    except StopIteration:
        print("ERF calculation skipped: Data loader is empty.")
        return

    kspace = kspace.to(device)
    mask = mask.to(device)
    
    # 그래디언트 계산이 필요한 부분을 명시적으로 감쌉니다.
    with torch.enable_grad():
        kspace.requires_grad_(True)
        # 순전파
        output = model(kspace, mask)
        
        # 목표 함수 정의: 출력 이미지 중앙 픽셀 값의 합
        B, H_out, W_out = output.shape
        center_pixel_values = output[:, H_out // 2, W_out // 2]
        scalar_objective = center_pixel_values.sum()
        
        # 입력에 대한 그래디언트 계산
        scalar_objective.backward()
    
    # 그래디언트 처리는 no_grad 컨텍스트에서 안전하게 수행합니다.
    with torch.no_grad():
        if kspace.grad is not None:
            # 그래디언트 크기를 계산합니다. (실수/허수 채널과 코일 채널에 대해 합산)
            # kspace.grad shape: [B, C, H, W, 2]
            grad_magnitude = torch.sqrt(kspace.grad.pow(2).sum(dim=-1)).sum(dim=(0, 1))
            
            # NumPy 배열로 변환
            grad_map_np = grad_magnitude.cpu().numpy()
            
            # Matplotlib Figure 생성 및 W&B 로깅
            fig = _create_erf_figure(grad_map_np, epoch)
            wandb.log({"validation/effective_receptive_field": wandb.Image(fig)}, step=epoch)
            plt.close(fig)
        else:
            print("ERF calculation failed: kspace.grad is None.")