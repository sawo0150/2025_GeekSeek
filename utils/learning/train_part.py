# utils/learning/train_part.py
import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import math
from pathlib import Path
from typing import Optional
from importlib import import_module
import inspect
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:64,garbage_collection_threshold:0.6"
)

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.logging.metric_accumulator import MetricAccumulator
from utils.logging.vis_logger import log_epoch_samples
from utils.common.utils import save_reconstructions
from utils.common.loss_function import SSIMLoss
from utils.logging.receptive_field import log_receptive_field

try:
    from utils.learning.leaderboard_eval_part import run_leaderboard_eval
except ImportError:
    run_leaderboard_eval = None


def train_epoch(args, epoch, model, data_loader, optimizer, scheduler,
                loss_type, ssim_metric, metricLog_train,
                scaler, amp_enabled, use_deepspeed, accum_steps):
    model.train()
    torch.cuda.reset_peak_memory_stats()

    len_loader = len(data_loader)
    total_loss, total_slices = 0., 0

    grad_clip_enabled  = getattr(args, "training_grad_clip_enable", False)
    grad_clip_max_norm = getattr(args, "training_grad_clip_max_norm", 1.0)
    grad_clip_norm_t   = getattr(args, "training_grad_clip_norm_type", 2)

    pbar = tqdm(enumerate(data_loader), total=len_loader, ncols=90, leave=False, desc=f"Epoch[{epoch:2d}/{args.num_epochs}]")

    start_iter = time.perf_counter()
    is_prompt_model = "Prompt" in model.__class__.__name__

    for iter_num, data in pbar:
        # [최종 수정] 9개, 7개 모든 케이스를 유연하게 처리
        if len(data) == 9:
            mask, kspace, target, maximum, fnames, _, cats, domain_indices, acc_indices = data
            domain_indices = domain_indices.cuda(non_blocking=True)
            acc_indices = acc_indices.cuda(non_blocking=True)
        elif len(data) == 7:
            mask, kspace, target, maximum, fnames, _, cats = data
            domain_indices = None
            acc_indices = None
        else:
            raise ValueError(f"Unexpected data batch length: {len(data)}")

        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        with autocast(enabled=amp_enabled):
            # [최종 수정] 모델 종류에 따라 필요한 인자를 정확히 전달
            if "PromptFIVarNet" in model.__class__.__name__:
                output = model(kspace, mask, domain_indices, acc_indices)
            elif is_prompt_model:
                output = model(kspace, mask, domain_indices)
            else:
                output = model(kspace, mask)

        if iter_num > 0 and iter_num % args.report_interval == 0:
            torch.cuda.synchronize()
            vram_alloc = torch.cuda.memory_allocated() / 1024**2
            vram_peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"\n  [VRAM at iter {iter_num}] Allocated: {vram_alloc:.2f} MB | Peak: {vram_peak:.2f} MB")
            
        current_loss = loss_type(output, target, maximum, cats)
        loss = current_loss.mean() / accum_steps

        if iter_num % accum_steps == 0:
            if use_deepspeed: model.zero_grad()
            else: optimizer.zero_grad(set_to_none=True)

        if use_deepspeed: model.backward(loss)
        elif amp_enabled: scaler.scale(loss).backward()
        else: loss.backward()

        if (iter_num + 1) % accum_steps == 0 or (iter_num + 1) == len_loader:
            if use_deepspeed:
                model.step()
                model.zero_grad()
            elif amp_enabled:
                scaler.unscale_(optimizer)
                if grad_clip_enabled: clip_grad_norm_(model.parameters(), grad_clip_max_norm, norm_type=grad_clip_norm_t)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                if grad_clip_enabled: clip_grad_norm_(model.parameters(), grad_clip_max_norm, norm_type=grad_clip_norm_t)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if (not use_deepspeed) and scheduler and isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)):
                scheduler.step()
        
        loss_vals = current_loss.detach().cpu().tolist()
        with torch.no_grad():
            ssim_loss_vals = ssim_metric(output.detach(), target, maximum, cats)
            ssim_vals = [1.0 - v for v in ssim_loss_vals]

        total_loss += sum(loss_vals)
        total_slices += len(loss_vals)

        batch_mean = sum(loss_vals) / len(loss_vals) if len(loss_vals) > 0 else 0
        pbar.set_postfix(loss=f"{batch_mean:.4g}")

        for lv, sv, cat in zip(loss_vals, ssim_vals, cats):
            metricLog_train.update(lv, sv, [cat])

        del output, current_loss, loss, loss_vals, ssim_loss_vals, ssim_vals, mask, kspace, target, maximum
        if domain_indices is not None: del domain_indices
        if 'acc_indices' in locals() and acc_indices is not None: del acc_indices
        torch.cuda.empty_cache()

    epoch_time = time.perf_counter() - start_iter
    return total_loss / total_slices if total_slices > 0 else 0, epoch_time


def validate(args, model, data_loader, acc_val, epoch, loss_type, ssim_metric):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()
    total_loss, total_ssim, n_slices = 0.0, 0.0, 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=90, leave=False, desc=f"Val  [{epoch:2d}/{args.num_epochs}]")
    is_prompt_model = "Prompt" in model.__class__.__name__

    with torch.no_grad():
        for idx, data in pbar:
            # [최종 수정] 9개, 7개 모든 케이스를 유연하게 처리
            if len(data) == 9:
                mask, kspace, target, maximum, fnames, slices, cats, domain_indices, acc_indices = data
                domain_indices = domain_indices.cuda(non_blocking=True)
                acc_indices = acc_indices.cuda(non_blocking=True)
            elif len(data) == 7:
                mask, kspace, target, maximum, fnames, slices, cats = data
                domain_indices = None
                acc_indices = None
            else:
                raise ValueError(f"Unexpected data batch length: {len(data)}")

            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)
            
            # [최종 수정] 모델 종류에 따라 필요한 인자를 정확히 전달
            if "PromptFIVarNet" in model.__class__.__name__:
                output = model(kspace, mask, domain_indices, acc_indices)
            elif is_prompt_model:
                output = model(kspace, mask, domain_indices)
            else:
                output = model(kspace, mask)
            
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

                out_slice, tgt_slice = output[i], target[i]
                max_i = maximum[i] if maximum.ndim > 0 else maximum
                loss_i = loss_type(out_slice, tgt_slice, max_i, cats[i]).item()
                ssim_loss_i = ssim_metric(out_slice, tgt_slice, max_i, cats[i]).item()
                ssim_i = 1 - ssim_loss_i
                
                total_loss += loss_i
                total_ssim += ssim_i
                n_slices += 1
                acc_val.update(loss_i, ssim_i, [cats[i]])

    for fname in reconstructions:
        reconstructions[fname] = np.stack([out for _, out in sorted(reconstructions[fname].items())])
    for fname in targets:
        targets[fname] = np.stack([out for _, out in sorted(targets[fname].items())])
    
    metric_loss = total_loss / n_slices if n_slices > 0 else 0
    metric_ssim = total_ssim / n_slices if n_slices > 0 else 0

    return metric_loss, metric_ssim, n_slices, reconstructions, targets, None, time.perf_counter() - start


def train(args, classifier=None):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    dup_cfg = getattr(args, "maskDuplicate", {"enable": False})
    dup_mul = (len(dup_cfg.get("accel_cfgs", [])) if dup_cfg.get("enable", False) else 1)
    
    accum_steps_default = getattr(args, "training_accum_steps", 1)
    ga_sched_cfg = getattr(args, "training_grad_accum_scheduler", {"enable": False})
    ga_sched_enable = ga_sched_cfg.get("enable", False)
    ga_milestones = sorted(ga_sched_cfg.get("milestones", []), key=lambda x: x.get("epoch", 0))

    def _accum_steps_for_epoch(ep: int) -> int:
        if not ga_sched_enable or not ga_milestones: return accum_steps_default
        curr = accum_steps_default
        for m in ga_milestones:
            if ep >= m.get("epoch", 0): curr = m.get("steps", curr)
            else: break
        return max(1, int(curr))

    accum_steps = _accum_steps_for_epoch(0)
    checkpointing = getattr(args, "training_checkpointing", False)
    amp_enabled = getattr(args, "training_amp", False)
    
    model_cfg = getattr(args, "model", {"_target_": "utils.model.varnet.VarNet"})
    model = instantiate(OmegaConf.create(model_cfg), use_checkpoint=checkpointing)
    model.to(device)
    
    is_prompt_model = "Prompt" in model.__class__.__name__
    print(f"[Hydra-model] Instantiated: {model.__class__.__name__}")
    print(f"[Mode] Running in {'Prompt-MR' if is_prompt_model else 'Standard'} mode.")

    loss_cfg = getattr(args, "LossFunction", {"_target_": "utils.common.loss_function.SSIMLoss"})
    loss_type = instantiate(OmegaConf.create(loss_cfg)).to(device=device)
    mask_th = {'brain_x4': 5e-5, 'brain_x8': 5e-5, 'knee_x4': 2e-5, 'knee_x8': 2e-5}
    ssim_metric = SSIMLoss(mask_only=True, mask_threshold=mask_th).to(device=device)

    ds_cfg = getattr(args, "deepspeed", None)
    use_deepspeed = ds_cfg is not None and ds_cfg.get("enable", False)

    if use_deepspeed:
        optimizer = None
    else:
        optim_cfg = getattr(args, "optimizer", None)
        optimizer = instantiate(OmegaConf.create(optim_cfg), params=model.parameters()) if optim_cfg else torch.optim.Adam(model.parameters(), args.lr)

    temp_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True, is_train=True)
    effective_steps = math.ceil(len(temp_loader) / accum_steps)
    del temp_loader

    if ds_cfg and ds_cfg["config"].get("scheduler"):
        sched_p = ds_cfg["config"]["scheduler"]["params"]
        sched_p["warmup_num_steps"] = args.warmup_epochs * effective_steps
        sched_p["total_num_steps"] = args.num_epochs * effective_steps

    scheduler = None
    if use_deepspeed:
        pass
    else:
        sched_cfg_raw = getattr(args, "LRscheduler", None)
        if sched_cfg_raw is not None:
            scheduler = instantiate(OmegaConf.create(sched_cfg_raw), optimizer=optimizer)

    if use_deepspeed:
        import deepspeed
        model, optimizer, scheduler, _ = deepspeed.initialize(model=model, optimizer=optimizer, model_parameters=model.parameters(), config=ds_cfg["config"])

    augmenter = instantiate(args.aug) if getattr(args, "aug", None) else None
    mask_aug_cfg = getattr(args, "maskAugment", {"enable": False})
    mask_augmenter = None
    if mask_aug_cfg.get("enable", False):
        cfg_clean = OmegaConf.create({k: v for k, v in mask_aug_cfg.items() if k != "enable"})
        mask_augmenter = instantiate(cfg_clean)

    start_epoch, best_val_loss, best_val_ssim, val_loss_history = 0, float('inf'), 0.0, []
    scaler = GradScaler(enabled=amp_enabled)

    if getattr(args, 'resume_checkpoint', None):
        ckpt_path = getattr(args, 'resume_checkpoint')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            best_val_ssim = ckpt.get('best_val_ssim', 0.0)
            start_epoch = ckpt.get('epoch', 0)
            val_loss_history = ckpt.get('val_loss_history', [])
            if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
            if scaler and ckpt.get('scaler'): scaler.load_state_dict(ckpt['scaler'])
            if augmenter and hasattr(augmenter, 'val_loss_history'): augmenter.val_loss_history.extend(val_loss_history)
            if mask_augmenter and hasattr(mask_augmenter, 'val_hist'): mask_augmenter.val_hist.extend(val_loss_history)
            print(f"[Resume] Loaded '{ckpt_path}'")

    _raw_eval = getattr(args, "evaluation", {})
    eval_cfg = _raw_eval.get("evaluation", _raw_eval)
    lb_enable = eval_cfg.get("enable", False)
    lb_every = eval_cfg.get("every_n_epochs", 999_999)

    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, is_train=False, classifier=classifier)
    
    for epoch in range(start_epoch, args.num_epochs):
        MetricLog_train = MetricAccumulator("train")
        MetricLog_val = MetricAccumulator("val")
        print(f'\nEpoch #{epoch:2d} ............... {args.exp_name} ...............')
        torch.cuda.empty_cache()
        
        last_val_loss = val_loss_history[-1] if val_loss_history else None
        if augmenter: augmenter.update_state(current_epoch=epoch, val_loss=last_val_loss)
        if mask_augmenter: mask_augmenter.update_state(current_epoch=epoch, val_loss=last_val_loss)

        train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True, augmenter=augmenter, mask_augmenter=mask_augmenter, is_train=True, classifier=classifier if is_prompt_model else None)

        accum_steps_epoch = _accum_steps_for_epoch(epoch)
        if accum_steps_epoch != accum_steps:
            accum_steps = accum_steps_epoch
            print(f"[GradAccum] Epoch {epoch}: accum_steps set to {accum_steps}")

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, scheduler, loss_type, ssim_metric, MetricLog_train, scaler, amp_enabled, use_deepspeed, accum_steps)
        val_loss, val_ssim, num_slices, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, MetricLog_val, epoch, loss_type, ssim_metric)
        val_loss_history.append(val_loss)

        is_new_best = val_ssim > best_val_ssim
        best_val_loss = min(best_val_loss, val_loss)
        best_val_ssim = max(best_val_ssim, val_ssim)
        
        checkpoint = {
            'epoch': epoch + 1, 'args': args, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'scaler': scaler.state_dict(),
            'best_val_ssim': best_val_ssim, 'best_val_loss': best_val_loss,
            'exp_dir': str(args.exp_dir), 'val_loss_history': val_loss_history 
        }
        torch.save(checkpoint, args.exp_dir / 'model.pt')
        if is_new_best: shutil.copyfile(args.exp_dir / 'model.pt', args.exp_dir / 'best_model.pt')

        if scheduler and not use_deepspeed:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(val_loss)
            elif not isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR)): pass
            else: scheduler.step()

        if getattr(args, "use_wandb", False) and wandb:
            MetricLog_train.log(epoch * dup_mul)
            MetricLog_val.log(epoch * dup_mul)
            if getattr(args, "wandb_use_visLogging", False):
                log_epoch_samples(reconstructions, targets, step=epoch * dup_mul, max_per_cat=args.max_vis_per_cat)
            wandb.log({"epoch": epoch, "lr": optimizer.param_groups[0]['lr']}, step=epoch * dup_mul)
            if is_new_best: wandb.save(str(args.exp_dir / "best_model.pt"))
        
        if run_leaderboard_eval and lb_enable and (epoch + 1) % lb_every == 0:
            print(f"[LeaderBoard] Epoch {epoch+1}: reconstruct & eval 시작")
            t0 = time.perf_counter()
            ssim = run_leaderboard_eval(args, model, classifier)
            dt = time.perf_counter() - t0
            print(f"[LeaderBoard] acc4={ssim['acc4']:.4f}  acc8={ssim['acc8']:.4f}  mean={ssim['mean']:.4f}  ({dt/60:.1f} min)")
            if getattr(args, "use_wandb", False) and wandb:
                wandb.log({"leaderboard/ssim_acc4": ssim["acc4"], "leaderboard/ssim_acc8": ssim["acc8"], "leaderboard/ssim_mean": ssim["mean"]}, step=epoch * dup_mul)
        
        print(f'Epoch=[{epoch+1:4d}/{args.num_epochs:4d}] TrainLoss={train_loss:.4g} ValLoss={val_loss:.4g} ValSSIM={val_ssim:.4g} TrainTime={train_time:.4f}s ValTime={val_time:.4f}s')
        if is_new_best: print("@@@@@@@@@@@@@@@@@@@@@@@@@ NewRecord @@@@@@@@@@@@@@@@@@@@@@@@@")

        del train_loader
        torch.cuda.empty_cache()