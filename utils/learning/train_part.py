import shutil
import numpy as np
import torch
import torch.nn as nn
import time, math
from pathlib import Path
import copy
from typing import Optional

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from tqdm import tqdm
from hydra.utils import instantiate          # ★ NEW
from omegaconf import OmegaConf              # ★ NEW
from torchvision.utils import make_grid
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure  # ★ NEW
from torch.cuda.amp import GradScaler, autocast          # ★---
from torch.nn import Module

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.logging.metric_accumulator import MetricAccumulator
from utils.logging.vis_logger import log_epoch_samples
from utils.common.utils import save_reconstructions, ssim_loss, ssim_loss_gpu
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

# def train_epoch(args, epoch, model, data_loader, optimizer, loss_type,ssim_metric, metricLog_train):
def train_epoch(args, epoch, model, data_loader, optimizer,
                loss_type, metricLog_train,
                scaler, amp_enabled, accum_steps):
    model.train()
    # start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    pbar = tqdm(enumerate(data_loader),
                total=len_loader,
                ncols=70,
                leave=False,
                desc=f"Epoch[{epoch:2d}/{args.num_epochs}]/")

    start_iter = time.perf_counter()
    for iter, data in pbar:

    # for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, fnames, _, cats = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True) # 슬라이스 max가 아니라 볼륨 전체 max임... (한 환자에 대해서..)

        with autocast(enabled=amp_enabled):
            output = model(kspace, mask)
        current_loss   = loss_type(output, target, maximum)
        loss = current_loss/accum_steps
        # print("max alloc MB:", torch.cuda.max_memory_allocated() / 1024**2)
        # ─── Accumulation ──────────────────────────────────────────────
        if iter % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # step & update?
        if (iter + 1) % accum_steps == 0 or (iter + 1) == len_loader:
            if amp_enabled:
                # unscale / clip_grad 등 필요 시 여기서
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
        # -------- SSIM Metric (no grad) ----------------------------------
        with torch.no_grad():
            ssim_val = 1 - ssim_loss_gpu(output.detach(), target, maximum).item()

        loss_val = current_loss.item()
        total_loss += loss_val

        # --- tqdm & ETA ---------------------------------------------------
        avg = (time.perf_counter() - start_iter) / (iter + 1)
        pbar.set_postfix(loss=f"{current_loss.item():.4g}")

        # --- 카테고리별 누적 ---------------------------------------------
        metricLog_train.update(loss_val, ssim_val, cats)
    # total_loss = total_loss / len_loader
    # return total_loss, time.perf_counter() - start_epoch

    epoch_time = time.perf_counter() - start_iter
    return total_loss / len_loader, epoch_time


def validate(args, model, data_loader, acc_val, epoch):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()
    total_loss = 0.0
    n_slices   = 0
    
    len_loader = len(data_loader)                 # ← 전체 길이
    pbar = tqdm(enumerate(data_loader),           # ← tqdm 래퍼
                total=len_loader,
                ncols=70,
                leave=False,
                desc=f"Val  [{epoch:2d}/{args.num_epochs}]")
    
    with torch.no_grad():
        for idx, data in pbar:
            mask, kspace, target, _, fnames, slices, cats = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):    # validate Batch 개수 고려해서 for로 묶었는듯
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                
                # ---- 스칼라 누적 -----------------------------------------
                loss_i  = ssim_loss(target[i].numpy(),
                                    output[i].cpu().numpy())
                total_loss += loss_i
                n_slices   += 1
                acc_val.update(loss_i, 1 - loss_i, [cats[i]])

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    # 기존 validate 방식 : subject 별 평균값의 평균값 (leaderboard랑 평가 방식이 다름)
    # metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    # num_subjects = len(reconstructions)
    
    metric_loss = total_loss / n_slices        # ← leaderboard 방식 (Slice 별 평균)
    

    return metric_loss, n_slices, reconstructions, targets, None, time.perf_counter() - start

def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # ▸ 0. 옵션 파싱 (기본값 유지)
    accum_steps   = getattr(args, "training_accum_steps",   1)
    checkpointing = getattr(args, "training_checkpointing", False)
    amp_enabled   = getattr(args, "training_amp",           False)

    model = VarNet(num_cascades=args.cascade,
                   chans=args.chans,
                   sens_chans=args.sens_chans,
                   use_checkpoint=checkpointing).to(device)    # ← Hydra 플래그 연결)

    loss_type = SSIMLoss().to(device=device)
    loss_cfg = getattr(args, "LossFunction", {"_target_": "utils.common.loss_function.SSIMLoss"})
    loss_type = instantiate(OmegaConf.create(loss_cfg)).to(device=device)

    # ── 1. Optimizer 선택 (fallback: Adam) ─────────────────────────────
    optim_cfg = getattr(args, "optimizer", None)
    if optim_cfg is not None:
        optimizer = instantiate(
            OmegaConf.create(optim_cfg),        # cfg → OmegaConf 객체
            params=model.parameters()           # 추가 인자주입
        )
    else:                                       # 안전장치
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    print(f"[Hydra] Optimizer ▶ {optimizer.__class__.__name__}")

    # ▸ 2. AMP scaler (옵션)
    scaler = GradScaler(enabled=amp_enabled)

    print(f"[Hydra] loss_func ▶ {loss_type}")      # 디버그

    # ──────────────── LR Scheduler (옵션) ────────────────
    scheduler_cfg = getattr(args, "LRscheduler", None)   # flatten 단계에서 dict 로 들어옴
    if scheduler_cfg is not None:
        # dict → OmegaConf 로 감싸야 instantiate 가 제대로 동작
        scheduler = instantiate(OmegaConf.create(scheduler_cfg),
                                optimizer=optimizer)
        print(f"[Hydra] Scheduler ▶ {scheduler}")      # 디버그
    else:
        scheduler = None

    best_val_loss = 1.
    start_epoch = 0

    print(args.data_path_train)
    print(args.data_path_val)
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    val_loss_log = np.empty((0, 2))

    for epoch in range(start_epoch, args.num_epochs):
        MetricLog_train = MetricAccumulator("train")
        MetricLog_val   = MetricAccumulator("val")
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model,
                                             train_loader, optimizer,
                                             loss_type, MetricLog_train,
                                             scaler, amp_enabled, accum_steps)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader,
                                                                                            MetricLog_val, epoch)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        
        # ──────────── LR 스케줄러 업데이트 ────────────
        if scheduler is not None:
            # ReduceLROnPlateau 는 val_metric 이 필요함
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)      # 여기서는 “낮을수록 좋음” metric
            else:
                scheduler.step()

        # ---------------- W&B 에폭 로그 (카테고리별) ----------------
        if getattr(args, "use_wandb", False) and wandb:
            MetricLog_train.log(epoch)
            MetricLog_val.log(epoch)
            # 추가 전역 정보(learning-rate 등)만 개별로 저장
            log_epoch_samples(reconstructions, targets,
                            step=epoch,
                            max_per_cat=args.max_vis_per_cat)   # ← config 값 사용
            
            wandb.log({"epoch": epoch,
                       "lr": optimizer.param_groups[0]['lr']}, step=epoch)
            if is_new_best:
                wandb.save(str(args.exp_dir / "best_model.pt"))

                                
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
