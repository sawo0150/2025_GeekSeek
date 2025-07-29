"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from skimage.metrics import structural_similarity
from torchmetrics.functional import structural_similarity_index_measure as ssim_f

import h5py
import numpy as np
import torch
import random
import os

def save_reconstructions(reconstructions, out_dir, targets=None, inputs=None):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])

def ssim_loss(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)
       ssim_loss is defined as (1 - ssim)
    """
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]
    return 1 - ssim

def ssim_loss_gpu(output, target, maximum):
    """
    output, target : (B, H, W) float32 CUDA tensors
    maximum        : (B,)         - 각 샘플의 max 값
    반환            : (B,)         - slice 평균 후 1-SSIM
    """
    # ① validate 로직과 맞추기 ─ 정규화
    output = output / maximum[:, None, None]   # (B,H,W)
    target = target / maximum[:, None, None]

    # ② 토치메트릭 입력 규격 맞추기
    output = output.unsqueeze(1)               # (B,1,H,W)
    target = target.unsqueeze(1)

    # ③ GPU SSIM (배치 평균)
    ssim_val = ssim_f(output, target, data_range=1.0)  # tensor shape (B,)
    return 1 - ssim_val.mean()                         # validate 와 동일 의미

def seed_fix(n):
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(n)   # 파이썬 객체 해시 순서를 고정
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False  # TF32 (가변 정밀) 끄기
    torch.backends.cudnn.allow_tf32      = False
    np.random.seed(n)
    random.seed(n)

def center_crop(data, height, width):
    """
    Center crop the data to the specified shape and move it to the specified device.
    
    Parameters:
    - data: numpy array of shape (C, H, W)
    - height: desired output height
    - width: desired output width
    - device: 'cpu' or 'cuda'
    
    Returns:
    - cropped tensor on the specified device
    """
    # Convert to PyTorch tensor and move to device
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.to(device)

    _, h, w = data.shape

    # Height padding
    if h < height:
        pad_h1 = (height - h) // 2
        pad_h2 = (height - h) - pad_h1
        data = torch.nn.functional.pad(data, (0, 0, pad_h1, pad_h2), mode='constant', value=0)
        h = height

    # Width padding
    if w < width:
        pad_w1 = (width - w) // 2
        pad_w2 = (width - w) - pad_w1
        data = torch.nn.functional.pad(data, (pad_w1, pad_w2, 0, 0), mode='constant', value=0)
        w = width

    start_h = (h - height) // 2
    start_w = (w - width) // 2
    return data[:, start_h:start_h + height, start_w:start_w + width]