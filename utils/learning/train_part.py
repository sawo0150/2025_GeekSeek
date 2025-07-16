import shutil
import numpy as np
import torch
import torch.nn as nn
import time, math
from pathlib import Path
import copy
from typing import Optional
from importlib import import_module
import inspect, math

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.learning.leaderboard_eval_part import run_leaderboard_eval
from utils.logging.metric_accumulator import MetricAccumulator
from utils.logging.vis_logger import log_epoch_samples
from utils.common.utils import save_reconstructions, ssim_loss, ssim_loss_gpu
from utils.common.loss_function import SSIMLoss
from utils.logging.receptive_field import log_receptive_field

import os
torch.autograd.set_detect_anomaly(True)

def train_epoch(args, epoch, model, data_loader, optimizer, scheduler,
                loss_type, ssim_metric, metricLog_train,
                scaler, amp_enabled, accum_steps):
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.
    total_slices = 0

    pbar = tqdm(enumerate(data_loader), total=len_loader, ncols=70, leave=False, desc=f"Epoch[{epoch:2d}/{args.num_epochs}]/")

    start_iter = time.perf_counter()
    for iter, data in pbar:
        mask, kspace, target, maximum, fnames, _, cats = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        # # ✨ [디버깅] 입력 데이터 점검
        # print(f"Iter {iter}: kspace min={kspace.min().item():.4e}, max={kspace.max().item():.4e}")
        # print(f"Iter {iter}: target min={target.min().item():.4e}, max={target.max().item():.4e}")
        # print(f"Iter {iter}: maximum min={maximum.min().item():.4e}, max={maximum.max().item():.4e}")
        # print(f"Iter {iter}: mask min={mask.min().item():.4e}, max={mask.max().item():.4e}")
        
        if torch.isnan(kspace).any() or torch.isinf(kspace).any():
            print(f"NaN or Inf detected in kspace at iter {iter}!")
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"NaN or Inf detected in target at iter {iter}!")
        if torch.isnan(maximum).any() or torch.isinf(maximum).any():
            print(f"NaN or Inf detected in maximum at iter {iter}!")
        if maximum.min().item() <= 1e-10:  # maximum이 0에 가까운 경우
            print(f"Warning: maximum is too small at iter {iter}!")

        with autocast(enabled=amp_enabled):
            output = model(kspace, mask)
            # ✨ [디버깅] 모델 출력 점검
            # print(f"Iter {iter}: output min={output.min().item():.4e}, max={output.max().item():.4e}")
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"NaN or Inf detected in model output at iter {iter}!")

            current_loss = loss_type(output, target, maximum, cats)
            # ✨ [디버깅] 손실 값 점검
            if torch.isnan(current_loss).any() or torch.isinf(current_loss).any():
                print(f"NaN or Inf detected in loss calculation at iter {iter}!")

        # backward에 사용할 스칼라 평균 loss
        loss = current_loss.mean()
        
        # ✨ loss가 유효한 숫자인지 확인 후 역전파 수행
        if not torch.isfinite(loss):
            print(f"Iter {iter}: Invalid loss detected (NaN or Inf), skipping backward pass.")
            if (iter + 1) % accum_steps == 0 or (iter + 1) == len_loader:
                optimizer.zero_grad(set_to_none=True)
            continue

        loss = loss / accum_steps
        
        if iter % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✨ Gradient Clipping 추가
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✨ Gradient Clipping 추가
            
        if (iter + 1) % accum_steps == 0 or (iter + 1) == len_loader:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None and isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
                scheduler.step()
        
        # -------- SSIM Metric (no grad) ----------------------------------
        loss_vals = current_loss.detach().cpu().tolist()
        with torch.no_grad():
            ssim_loss_vals = ssim_metric(output.detach(), target, maximum, cats)
            ssim_vals = [1.0 - v for v in ssim_loss_vals]

        total_loss += sum(loss_vals)
        total_slices += len(loss_vals)

        # --- tqdm & ETA ---------------------------------------------------
        batch_mean = sum(loss_vals) / len(loss_vals)
        pbar.set_postfix(loss=f"{batch_mean:.4g}")

        # --- 카테고리별 & slice 별 누적 -------------------------------------
        for lv, sv, cat in zip(loss_vals, ssim_vals, cats):
            metricLog_train.update(lv, sv, [cat])

    epoch_time = time.perf_counter() - start_iter
    return total_loss / total_slices, epoch_time


def validate(args, model, data_loader, acc_val, epoch, loss_type, ssim_metric):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()
    total_loss = 0.0
    total_ssim = 0.0
    n_slices = 0
    
    len_loader = len(data_loader)
    pbar = tqdm(enumerate(data_loader), total=len_loader, ncols=70, leave=False, desc=f"Val  [{epoch:2d}/{args.num_epochs}]")
    
    with torch.no_grad():
        for idx, data in pbar:
            mask, kspace, target, maximum, fnames, slices, cats = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            
            # ✨ [디버깅] 검증 데이터 점검
            # print(f"Val Iter {idx}: kspace min={kspace.min().item():.4e}, max={kspace.max().item():.4e}")
            # print(f"Val Iter {idx}: target min={target.min().item():.4e}, max={target.max().item():.4e}")
            # print(f"Val Iter {idx}: maximum min={maximum.min().item():.4e}, max={maximum.max().item():.4e}")
            
            output = model(kspace, mask)
            print(f"Val Iter {idx}: output min={output.min().item():.4e}, max={output.max().item():.4e}")
            
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

                out_slice = output[i]
                tgt_slice = target[i]
                max_i = maximum[i] if maximum.ndim > 0 else maximum
                loss_i = loss_type(out_slice, tgt_slice, max_i, cats[i]).item()
                print(f"Val Iter {idx}, Slice {i}: loss={loss_i:.4e}")
                
                total_loss += loss_i
                n_slices += 1

                ssim_loss_i = ssim_metric(out_slice, tgt_slice, max_i, cats[i]).item()
                ssim_i = 1 - ssim_loss_i
                total_ssim += ssim_i
                acc_val.update(loss_i, ssim_i, [cats[i]])

    for fname in reconstructions:
        reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname].items())])
    for fname in targets:
        targets[fname] = np.stack([out for _, out in sorted(targets[fname].items())])
    
    metric_loss = total_loss / n_slices
    metric_ssim = total_ssim / n_slices
    
    return metric_loss, metric_ssim, n_slices, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, scheduler, best_val_loss, best_val_ssim, val_loss_history, is_new_best):
    checkpoint = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_val_loss': best_val_loss,
        'best_val_ssim': best_val_ssim,
        'exp_dir': str(exp_dir),
        'val_loss_history': val_loss_history
    }
    torch.save(checkpoint, exp_dir / 'model.pt')
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    print('Current cuda device: ', torch.cuda.current_device())

    dup_cfg = getattr(args, "maskDuplicate", {"enable": False})
    dup_mul = len(dup_cfg.get("accel_cfgs", [])) if dup_cfg.get("enable", False) else 1
    
    print("[Hydra-visLogging] ", getattr(args, "wandb_use_visLogging", False))
    print("[Hydra-receptiveField] ", getattr(args, "wandb_use_receptiveField", False))
    print(f"[Hydra-maskDuplicate] {dup_cfg}")

    accum_steps = getattr(args, "training_accum_steps", 1)
    checkpointing = getattr(args, "training_checkpointing", False)
    amp_enabled = getattr(args, "training_amp", False)  # ✨ AMP 비활성화 가능

    early_cfg = getattr(args, "early_stop", {})
    early_enabled = early_cfg.get("enable", False)
    stage_table = {s["epoch"]: s["ssim"] for s in early_cfg.get("stages", [])}
    print(f"[Hydra-eval] {early_cfg}")
    print(f"[Hydra-eval] early_enabled={early_enabled}, stage_table={stage_table}")

    model_cfg = getattr(args, "model", {"_target_": "utils.model.varnet.VarNet"})
    model = instantiate(OmegaConf.create(model_cfg), use_checkpoint=checkpointing)
    model.to(device)
    
    # ✨ 모델 가중치 초기화 (Xavier)
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)
    
    print(f"[Hydra-model] model_cfg={model_cfg}")

    # ✨ 손실 함수 선택 (디버깅용으로 MSELoss 옵션 추가)
    loss_cfg = getattr(args, "LossFunction", {"_target_": "utils.common.loss_function.SSIMLoss"})
    use_mse_loss = False  # ✨ 디버깅용: True로 설정하면 MSELoss 사용
    if use_mse_loss:
        loss_type = torch.nn.MSELoss().to(device=device)
    else:
        loss_type = instantiate(OmegaConf.create(loss_cfg)).to(device=device)
    
    mask_th = {'brain_x4': 5e-5, 'brain_x8': 5e-5, 'knee_x4': 2e-5, 'knee_x8': 2e-5}
    ssim_metric = SSIMLoss(mask_only=True, mask_threshold=mask_th).to(device=device)
    print(f"[Hydra] loss_func ▶ {loss_type}")

    optim_cfg = getattr(args, "optimizer", None)
    if optim_cfg is not None:
        optimizer = instantiate(OmegaConf.create(optim_cfg), params=model.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=getattr(args, "lr", 0.0001))  # ✨ 학습률 낮춤
    print(f"[Hydra] Optimizer ▶ {optimizer.__class__.__name__}")

    temp_loader = create_data_loaders(
        data_path=args.data_path_train,
        args=args,
        shuffle=True,
        augmenter=None,
        mask_augmenter=None
    )
    effective_steps = math.ceil(len(temp_loader) / accum_steps)
    del temp_loader

    sched_cfg_raw = getattr(args, "LRscheduler", None)
    scheduler = None
    if sched_cfg_raw is not None:
        sched_cfg = OmegaConf.create(sched_cfg_raw)
        temp_loader = create_data_loaders(
            data_path=args.data_path_train,
            args=args,
            shuffle=True,
            augmenter=None,
            mask_augmenter=None,
            is_train=True
        )
        effective_steps = math.ceil(len(temp_loader) / accum_steps)
        del temp_loader

        target_path = sched_cfg["_target_"]
        mod_name, cls_name = target_path.rsplit(".", 1)
        SchedulerCls = getattr(import_module(mod_name), cls_name)
        sig = inspect.signature(SchedulerCls.__init__)
        valid_keys = set(sig.parameters.keys())

        if "effective_steps" in sched_cfg and "effective_steps" not in valid_keys:
            del sched_cfg["effective_steps"]

        if cls_name == "CyclicLR":
            sched_cfg["effective_steps"] = effective_steps
            sched_cfg["step_size_up"] = sched_cfg.get("step_size_up", effective_steps * 2)
            sched_cfg["max_lr"] = sched_cfg.get("max_lr", args.lr * 2)  # ✨ max_lr 낮춤
        if cls_name == "OneCycleLR":
            sched_cfg["effective_steps"] = effective_steps
            sched_cfg["total_steps"] = sched_cfg.get("total_steps", effective_steps * args.num_epochs)
            sched_cfg["max_lr"] = sched_cfg.get("max_lr", args.lr * 2)  # ✨ max_lr 낮춤

        clean_dict = {k: v for k, v in sched_cfg.items() if k in valid_keys or k.startswith("_")}
        scheduler = instantiate(OmegaConf.create(clean_dict), optimizer=optimizer)
        print(f"[Hydra] Scheduler ▶ {scheduler}")

    augmenter = None
    if getattr(args, "aug", None):
        print("[Hydra] Augmenter를 생성합니다.")
        augmenter = instantiate(args.aug)
    print(getattr(args, "aug", None))

    mask_augmenter = None
    mask_aug_cfg = getattr(args, "maskAugment", {"enable": False})
    print("[Hydra] mask_augmenter : ", mask_aug_cfg.get("enable", False))
    if mask_aug_cfg.get("enable", False):
        cfg_clean = OmegaConf.create({k: v for k, v in mask_aug_cfg.items() if k != "enable"})
        mask_augmenter = instantiate(cfg_clean)
    print(getattr(args, "maskAugment", None))

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_ssim = 0.0
    val_loss_history = []

    if getattr(args, 'resume_checkpoint', None):
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_val_ssim = ckpt.get('best_val_ssim', 0.0)
        start_epoch = ckpt.get('epoch', 0)
        val_loss_history = ckpt.get('val_loss_history', [])
        print(f"[Resume] Loaded '{args.resume_checkpoint}' → epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if augmenter:
            augmenter.val_loss_history.clear()
            augmenter.val_loss_history.extend(val_loss_history)
            print(f"[Resume] Augmenter val_loss history 복원 완료 ({len(val_loss_history)}개 항목)")
        if mask_augmenter:
            mask_augmenter.val_hist.clear()
            mask_augmenter.val_hist.extend(val_loss_history)
            print(f"[Resume] MaskAugmenter val_loss history 복원 완료 ({len(val_loss_history)}개 항목)")

    scaler = GradScaler(enabled=amp_enabled)

    print(args.data_path_train)
    print(args.data_path_val)

    _raw_eval = getattr(args, "evaluation", {})
    eval_cfg = _raw_eval.get("evaluation", _raw_eval)
    lb_enable = eval_cfg.get("enable", False)
    lb_every = eval_cfg.get("every_n_epochs", 999_999)
    print(f"[Hydra-eval] {eval_cfg}")
    print(f"[Hydra-eval] lb_enable={lb_enable}, lb_every={lb_every}")

    val_loader = create_data_loaders(
        data_path=args.data_path_val,
        args=args,
        augmenter=None,
        mask_augmenter=None,
        is_train=False
    )
    
    val_loss_log = np.empty((0, 2))

    for epoch in range(start_epoch, args.num_epochs):
        MetricLog_train = MetricAccumulator("train")
        MetricLog_val = MetricAccumulator("val")
        print(f'Epoch #{epoch:2d} ............... {args.exp_name} ...............')

        if augmenter is not None:
            last_val_loss = val_loss_history[-1] if val_loss_history else None
            augmenter.update_state(current_epoch=epoch, val_loss=last_val_loss)
        if mask_augmenter is not None:
            last_val_loss = val_loss_history[-1] if val_loss_history else None
            mask_augmenter.update_state(current_epoch=epoch, val_loss=last_val_loss)
        
        train_loader = create_data_loaders(
            data_path=args.data_path_train,
            args=args,
            shuffle=True,
            augmenter=augmenter,
            mask_augmenter=mask_augmenter,
            is_train=True
        )

        train_loss, train_time = train_epoch(args, epoch, model,
                                             train_loader, optimizer, scheduler,
                                             loss_type, ssim_metric, MetricLog_train,
                                             scaler, amp_enabled, accum_steps)
        val_loss, val_ssim, num_subjects, reconstructions, targets, inputs, val_time = validate(
            args, model, val_loader, MetricLog_val, epoch, loss_type, ssim_metric)
        
        val_loss_history.append(val_loss)
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        is_new_best = val_ssim > best_val_ssim
        best_val_loss = min(best_val_loss, val_loss)
        best_val_ssim = max(best_val_ssim, val_ssim)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, scheduler,
                   best_val_loss, best_val_ssim, val_loss_history, is_new_best)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if getattr(args, "use_wandb", False) and wandb:
            MetricLog_train.log(epoch * dup_mul)
            MetricLog_val.log(epoch * dup_mul)
            if getattr(args, "wandb_use_visLogging", False):
                print("visual Logging...")
                log_epoch_samples(reconstructions, targets,
                                  step=epoch * dup_mul,
                                  max_per_cat=args.max_vis_per_cat)
            
            wandb.log({"epoch": epoch, "lr": optimizer.param_groups[0]['lr']}, step=epoch * dup_mul)
            
            if getattr(args, "wandb_use_receptiveField", False):
                print("Calculating and logging effective receptive field...")
                log_receptive_field(model=model, data_loader=val_loader, epoch=epoch, device=device)
                print("Effective receptive field logged to W&B.")

            if is_new_best:
                wandb.save(str(args.exp_dir / "best_model.pt"))

        if lb_enable and (epoch + 1) % lb_every == 0:
            print(f"[LeaderBoard] Epoch {epoch + 1}: reconstruct & eval 시작")
            t0 = time.perf_counter()
            ssim = run_leaderboard_eval(
                model_ckpt_dir=args.exp_dir,
                leaderboard_root=Path(eval_cfg["leaderboard_root"]),
                gpu=args.GPU_NUM,
                batch_size=eval_cfg["batch_size"],
                output_key=eval_cfg["output_key"],
            )
            dt = time.perf_counter() - t0
            print(f"[LeaderBoard] acc4={ssim['acc4']:.4f}  acc8={ssim['acc8']:.4f}  "
                  f"mean={ssim['mean']:.4f}  ({dt/60:.1f} min)")

            if getattr(args, "use_wandb", False) and wandb:
                wandb.log({
                    "leaderboard/ssim_acc4": ssim["acc4"],
                    "leaderboard/ssim_acc8": ssim["acc8"],
                    "leaderboard/ssim_mean": ssim["mean"],
                    "leaderboard/epoch": epoch,
                    "leaderboard/time_min": dt / 60,
                }, step=epoch * dup_mul)
                                
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} ValSSIM = {val_ssim:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(f'ForwardTime = {time.perf_counter() - start:.4f}s')

        current_epoch = epoch + 1
        if early_enabled and current_epoch in stage_table:
            req = stage_table[current_epoch]
            if val_ssim < req:
                print(f"[EarlyStop] Epoch {current_epoch}: "
                      f"val_ssim={val_ssim:.4f} < target={req:.4f}. 학습 중단!")
                break