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
from utils.common.loss_function import SSIMLoss # train loss & metric loss
from utils.model.varnet import VarNet
from utils.logging.receptive_field import log_receptive_field


import os

def train_epoch(args, epoch, model, data_loader, optimizer,
                loss_type, ssim_metric, metricLog_train,
                scaler, amp_enabled, accum_steps):
    model.train()
    # start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    total_slices = 0

    pbar = tqdm(enumerate(data_loader),
                total=len_loader,
                ncols=70,
                leave=False,
                desc=f"Epoch[{epoch:2d}/{args.num_epochs}]/")

    start_iter = time.perf_counter()
    for iter, data in pbar:
        mask, kspace, target, maximum, fnames, _, cats = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True) # 슬라이스 max가 아니라 볼륨 전체 max임... (한 환자에 대해서..)

        with autocast(enabled=amp_enabled):
            output = model(kspace, mask)
            
        # pass cats list so MaskedLoss can pick per-cat thresholds
        # 1) per-sample loss 텐서 [B]
        current_loss = loss_type(output, target, maximum, cats)
        # 2) backward 에는 스칼라 평균을 사용
        loss = current_loss.mean() / accum_steps

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
        loss_vals = current_loss.detach().cpu().tolist()                # list of floats, len=
        with torch.no_grad():
            ssim_loss_vals = ssim_metric(output.detach(), target, maximum, cats)
            ssim_vals = [1.0 - v for v in ssim_loss_vals]

        total_loss += sum(loss_vals)
        total_slices += len(loss_vals)

        # --- tqdm & ETA ---------------------------------------------------
        batch_mean = sum(loss_vals) / len(loss_vals)
        pbar.set_postfix(loss=f"{batch_mean:.4g}")

        # --- 카테고리별 & slice 별 누적 -------------------------------------
        # MetricAccumulator.update(loss: float, ssim: float, cats: list[str])
        # 여기서는 샘플별로 호출해서, 각 slice 로깅
        for lv, sv, cat in zip(loss_vals, ssim_vals, cats):
            metricLog_train.update(lv, sv, [cat])

    # total_loss = total_loss / len_loader
    # return total_loss, time.perf_counter() - start_epoch

    epoch_time = time.perf_counter() - start_iter
    return total_loss / total_slices, epoch_time


def validate(args, model, data_loader, acc_val, epoch, loss_type, ssim_metric):
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
            mask, kspace, target, maximum, fnames, slices, cats = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target  = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            output = model(kspace, mask)

            # for i in range(output.shape[0]):    # validate Batch 개수 고려해서 for로 묶었는듯
            #     reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
            #     targets[fnames[i]][int(slices[i])] = target[i].numpy()
                
            #     # ---- 스칼라 누적 -----------------------------------------
            #     loss_i  = ssim_loss(target[i].numpy(),
            #                         output[i].cpu().numpy())
            #     total_loss += loss_i
            #     n_slices   += 1
            #     acc_val.update(loss_i, 1 - loss_i, [cats[i]])

            
            for i in range(output.shape[0]):    # validate Batch 개수 고려해서 for로 묶었는듯
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])]       = target[i].cpu().numpy()

                # ------------------ loss 계산 -------------------
                out_slice = output[i]
                tgt_slice = target[i]
                max_i     = maximum[i] if maximum.ndim>0 else maximum
                loss_i    = loss_type(out_slice, tgt_slice, max_i, cats[i]).item()
                total_loss += loss_i
                n_slices  += 1

                # -------------- SSIM metric 계산 ---------------
                ssim_loss_i = ssim_metric(out_slice, tgt_slice, max_i, cats[i]).item()
                ssim_i = 1 - ssim_loss_i
                acc_val.update(loss_i, ssim_i, [cats[i]])

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
    
    checkpoint = {
        'epoch':         epoch,
        'args':          args,                              # ← SimpleNamespace 통째로
        'model':         model.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'exp_dir':       str(exp_dir),                      # Path는 문자열로 저장해도 OK
    }
    torch.save(checkpoint, exp_dir / 'model.pt')
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

    loss_cfg = getattr(args, "LossFunction", {"_target_": "utils.common.loss_function.SSIMLoss"})
    loss_type = instantiate(OmegaConf.create(loss_cfg)).to(device=device)
    # SSIM metric 계산용 (항상 SSIM 기반 로그를 위해 별도 생성)
    mask_th = {  'brain_x4': 5e-5,
                    'brain_x8': 5e-5,
                    'knee_x4':  2e-5,
                    'knee_x8':  2e-5}
    ssim_metric = SSIMLoss(mask_only = True, mask_threshold=mask_th).to(device=device)

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

    # ✨ Augmenter 객체 생성
    augmenter = None
    if getattr(args, "aug", None):
        print("[Hydra] Augmenter를 생성합니다.")
        # args.aug에 mraugment.yaml에서 읽은 설정이 들어있습니다.
        augmenter = instantiate(args.aug)



    # ── Resume logic (이제 model, optimizer가 정의된 이후) ──
    start_epoch   = 0
    best_val_loss = float('inf')
    val_loss_history = [] # val loss기록 for augmenter

    if getattr(args, 'resume_checkpoint', None):
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        start_epoch = ckpt.get('epoch', 0)
        val_loss_history = ckpt.get('val_loss_history', []) # 체크포인트에서 기록 복원
        print(f"[Resume] Loaded '{args.resume_checkpoint}' → epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        # 재개 시, augmenter의 상태도 복원
        if augmenter:
            augmenter.val_loss_history.clear()
            augmenter.val_loss_history.extend(val_loss_history)
            print(f"[Resume] Augmenter val_loss history 복원 완료 ({len(val_loss_history)}개 항목)")

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

    best_val_loss = 10000.

    print(args.data_path_train)
    print(args.data_path_val)


    # train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True) # 매 에폭마다 생성 예정
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, augmenter=None)
    
    val_loss_log = np.empty((0, 2))

    for epoch in range(start_epoch, args.num_epochs):
        MetricLog_train = MetricAccumulator("train")
        MetricLog_val   = MetricAccumulator("val")
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        if augmenter is not None:
            last_val_loss = val_loss_history[-1] if val_loss_history else None
            augmenter.update_state(current_epoch=epoch, val_loss=last_val_loss)

        train_loader = create_data_loaders(
            data_path=args.data_path_train, 
            args=args, 
            shuffle=True, 
            augmenter=augmenter # 업데이트된 augmenter 전달
        )

        train_loss, train_time = train_epoch(args, epoch, model,
                                             train_loader, optimizer,
                                             loss_type, ssim_metric, MetricLog_train,
                                             scaler, amp_enabled, accum_steps)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader,
                                                                                        MetricLog_val, epoch,
                                                                                        loss_type, ssim_metric)
        # ✨ val_loss 기록 (스케줄러 및 체크포인트용)
        val_loss_history.append(val_loss)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        # ✨ save_model에 val_loss_history 추가 (상태 저장을 위해)
        checkpoint = {
            'epoch': epoch + 1,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': str(args.exp_dir),
            'val_loss_history': val_loss_history 
        }
        torch.save(checkpoint, args.exp_dir / 'model.pt')

        if is_new_best:
            shutil.copyfile(args.exp_dir / 'model.pt', args.exp_dir / 'best_model.pt')
        
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
            log_epoch_samples(reconstructions, targets,
                            step=epoch,
                            max_per_cat=args.max_vis_per_cat)
            
            wandb.log({"epoch": epoch, "lr": optimizer.param_groups[0]['lr']}, step=epoch)
            
            # ┕ [추가] 매 에폭의 검증 단계 후, ERF를 계산하고 W&B에 로깅합니다.
            print("Calculating and logging effective receptive field...")
            log_receptive_field(
                model=model,
                data_loader=val_loader,
                epoch=epoch,
                device=device
            )
            print("Effective receptive field logged to W&B.")
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
