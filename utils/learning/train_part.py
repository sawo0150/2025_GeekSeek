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
from hydra.utils import instantiate          # ★ NEW
from omegaconf import OmegaConf              # ★ NEW
from torchvision.utils import make_grid
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure  # ★ NEW
from torch.cuda.amp import GradScaler, autocast          # ★---
from torch.nn import Module

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.learning.leaderboard_eval_part import run_leaderboard_eval
from utils.logging.metric_accumulator import MetricAccumulator
from utils.logging.vis_logger import log_epoch_samples
from utils.common.utils import save_reconstructions, ssim_loss, ssim_loss_gpu
from utils.common.loss_function import SSIMLoss # train loss & metric loss
from utils.model.varnet import VarNet
from utils.logging.receptive_field import log_receptive_field


import os

def train_epoch(args, epoch, model, data_loader, optimizer, scheduler,
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

            if scheduler is not None:
                # OneCycleLR·CyclicLR 등은 매 iteration 호출이 권장
                if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR,
                                        torch.optim.lr_scheduler.CyclicLR)):
                    scheduler.step()
        
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
    total_ssim = 0.0
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
            # print("max alloc MB:", torch.cuda.max_memory_allocated() / 1024**2)
            
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
                total_ssim += ssim_i
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
    metric_ssim = total_ssim / n_slices        # ← leaderboard 방식 (Slice 별 평균)
    

    return metric_loss, metric_ssim, n_slices, reconstructions, targets, None, time.perf_counter() - start

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

    # 파일 맨 앞, train() 안
    dup_cfg   = getattr(args, "maskDuplicate", {"enable": False})
    dup_mul   = (len(dup_cfg.get("accel_cfgs", []))
                if dup_cfg.get("enable", False) else 1)
    
    print("[Hydra-visLogging] ", getattr(args, "wandb_use_visLogging", False))
    print("[Hydra-receptiveField] ", getattr(args, "wandb_use_receptiveField", False))
    
    print(f"[Hydra-maskDuplicate] {dup_cfg}")

    # ▸ 0. 옵션 파싱 (기본값 유지)
    accum_steps   = getattr(args, "training_accum_steps",   1)
    checkpointing = getattr(args, "training_checkpointing", False)
    amp_enabled   = getattr(args, "training_amp",           False)

    early_cfg = getattr(args, "early_stop", {})
    early_enabled = early_cfg.get("enable", False)
    stage_table = {s["epoch"]: s["ssim"] for s in early_cfg.get("stages", [])}
    # ex) {10:0.90, 20:0.95, 25:0.96}
    print(f"[Hydra-eval] {early_cfg}")
    print(f"[Hydra-eval] early_enabled={early_enabled}, stage_table={stage_table}")


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

    print(f"[Hydra] loss_func ▶ {loss_type}")      # 디버그


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

    # ─ train() 맨 앞쪽: train_loader 길이 한 번만 미리 계산 ─
    temp_loader = create_data_loaders(
        data_path=args.data_path_train,
        args=args,
        shuffle=True,
        augmenter=None,            # 길이만 알면 되므로 Augment 미적용
        mask_augmenter=None
    )
    effective_steps = math.ceil(len(temp_loader) / accum_steps)
    del temp_loader                # 메모리 바로 반환

    # ──────────────── LR Scheduler (옵션) ────────────────
    # 0) Config → OmegaConf
    sched_cfg_raw = getattr(args, "LRscheduler", None)
    scheduler = None
    if sched_cfg_raw is not None:
        sched_cfg = OmegaConf.create(sched_cfg_raw)   # dict → OmegaConf

        # 1) 공통 계산
        temp_loader = create_data_loaders(
            data_path=args.data_path_train,
            args=args,
            shuffle=True,
            augmenter=None,
            mask_augmenter=None,
            is_train=True,      # ★ train만 True
        )
        effective_steps = math.ceil(len(temp_loader) / accum_steps)
        del temp_loader

        # 2) target class 로드 & 시그니처 분석
        target_path = sched_cfg["_target_"]
        mod_name, cls_name = target_path.rsplit(".", 1)
        SchedulerCls = getattr(import_module(mod_name), cls_name)
        sig = inspect.signature(SchedulerCls.__init__)
        valid_keys = set(sig.parameters.keys())        # 허용 인수 목록

        # 3) 필요한 key만 conditionally 추가
        if "effective_steps" in sched_cfg and "effective_steps" not in valid_keys:
            del sched_cfg["effective_steps"]

        # CyclicLR
        if cls_name == "CyclicLR":
            sched_cfg["effective_steps"] = effective_steps
            sched_cfg["step_size_up"] = sched_cfg.get(
                "step_size_up", effective_steps * 2
            )
            sched_cfg["max_lr"] = sched_cfg.get("max_lr", args.lr * 6)

        # OneCycleLR
        if cls_name == "OneCycleLR":
            sched_cfg["effective_steps"] = effective_steps
            sched_cfg["total_steps"] = sched_cfg.get(
                "total_steps", effective_steps * args.num_epochs
            )
            sched_cfg["max_lr"] = sched_cfg.get("max_lr", args.lr * 10)

        # 4) instantiate (불필요 인수 제거 후)
        clean_dict = {k: v for k, v in sched_cfg.items() if k in valid_keys or k.startswith("_")}
        scheduler = instantiate(OmegaConf.create(clean_dict), optimizer=optimizer)
        print(f"[Hydra] Scheduler ▶ {scheduler}")


    # ✨ Augmenter 객체 생성
    augmenter = None
    print(getattr(args, "aug", None))
    if getattr(args, "aug", None):
        print("[Hydra] Augmenter를 생성합니다.")
        # args.aug에 mraugment.yaml에서 읽은 설정이 들어있습니다.
        augmenter = instantiate(args.aug)

    # ① MaskAugmenter : 한 번만 생성
    mask_augmenter = None
    mask_aug_cfg = getattr(args, "maskAugment", {"enable": False})
    print("[Hydra] mask_augmenter : ", mask_aug_cfg.get("enable", False))
    if mask_aug_cfg.get("enable", False):
        # 'enable' 키만 제거한 새 OmegaConf 객체를 만들어야 합니다.
        cfg_clean = OmegaConf.create({k: v for k, v in mask_aug_cfg.items()
                                    if k != "enable"})
        mask_augmenter = instantiate(cfg_clean)

    # ── Resume logic (이제 model, optimizer가 정의된 이후) ──
    start_epoch   = 0
    best_val_loss = float('inf')
    best_val_ssim = 0.0
    val_loss_history = [] # val loss기록 for augmenter

    if getattr(args, 'resume_checkpoint', None):
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_val_ssim = ckpt.get('best_val_ssim', 0.0)
        start_epoch = ckpt.get('epoch', 0)
        val_loss_history = ckpt.get('val_loss_history', []) # 체크포인트에서 기록 복원
        print(f"[Resume] Loaded '{args.resume_checkpoint}' → epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        # 재개 시, augmenter의 상태도 복원
        # --- resume 영역 ---
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


    # ▸ 2. AMP scaler (옵션)
    scaler = GradScaler(enabled=amp_enabled)

    print(args.data_path_train)
    print(args.data_path_val)

    # ───────────────── evaluation 서브트리 언랩 ─────────────────
    # ①  args.evaluation 이 이미 {"enable": …} 형태라면 그대로 사용
    # ②  args.evaluation 이 {"evaluation": {...}} 처럼 한 번 더
    #     래핑돼 있으면 내부 dict 를 꺼낸다.
    _raw_eval = getattr(args, "evaluation", {})
    eval_cfg  = _raw_eval.get("evaluation", _raw_eval)

    lb_enable = eval_cfg.get("enable", False)
    lb_every  = eval_cfg.get("every_n_epochs", 999_999)   # 기본 매우 크게

    print(f"[Hydra-eval] {eval_cfg}")
    print(f"[Hydra-eval] lb_enable={lb_enable}, lb_every={lb_every}")

    # train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True) # 매 에폭마다 생성 예정
    val_loader = create_data_loaders(data_path=args.data_path_val, 
                                     args=args, 
                                     augmenter=None,
                                     mask_augmenter=None,
                                     is_train=False,     # ★ val/test는 False)
    )
    
    val_loss_log = np.empty((0, 2))

    for epoch in range(start_epoch, args.num_epochs):
        MetricLog_train = MetricAccumulator("train")
        MetricLog_val   = MetricAccumulator("val")
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        if augmenter is not None:
            last_val_loss = val_loss_history[-1] if val_loss_history else None
            augmenter.update_state(current_epoch=epoch, val_loss=last_val_loss)
        # ---- MaskAugmenter 스케줄 업데이트 ----
        if mask_augmenter is not None:
            last_val_loss = val_loss_history[-1] if val_loss_history else None
            mask_augmenter.update_state(current_epoch=epoch, val_loss=last_val_loss)
        
        train_loader = create_data_loaders(
            data_path=args.data_path_train, 
            args=args, 
            shuffle=True, 
            augmenter=augmenter, # 업데이트된 augmenter 전달
            mask_augmenter=mask_augmenter,
            is_train=True      # ★ train만 True
        )

        train_loss, train_time = train_epoch(args, epoch, model,
                                             train_loader, optimizer, scheduler,
                                             loss_type, ssim_metric, MetricLog_train,
                                             scaler, amp_enabled, accum_steps)
        val_loss,val_ssim, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader,
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

        # is_new_best = val_loss < best_val_loss
        is_new_best = val_ssim > best_val_ssim
        best_val_loss = min(best_val_loss, val_loss)
        best_val_ssim = max(best_val_ssim, val_ssim)

        # ✨ save_model에 val_loss_history 추가 (상태 저장을 위해)
        checkpoint = {
            'epoch': epoch + 1,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            'best_val_ssim': best_val_ssim,
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

            MetricLog_train.log(epoch*dup_mul)
            MetricLog_val.log(epoch*dup_mul)
            # 추가 전역 정보(learning-rate 등)만 개별로 저장
            if getattr(args, "wandb_use_visLogging", False) and wandb:
                print("visual Logging...")
                log_epoch_samples(reconstructions, targets,
                                step=epoch*dup_mul,
                                max_per_cat=args.max_vis_per_cat)   # ← config 값 사용
            
            wandb.log({"epoch": epoch,
                       "lr": optimizer.param_groups[0]['lr']}, step=epoch*dup_mul)
            
            if getattr(args, "wandb_use_receptiveField", False) and wandb:
                # ┕ [추가] 매 에폭의 검증 단계 후, ERF를 계산하고 W&B에 로깅합니다.
                # crop 안할시 receptive Field 계산 불가능 -> 이거 해결용..
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

        # ───────── Leaderboard 평가 트리거 ─────────
        if lb_enable and (epoch + 1) % lb_every == 0:
            print(f"[LeaderBoard] Epoch {epoch+1}: reconstruct & eval 시작")
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

            # ─ W&B 로깅 ─
            if getattr(args, "use_wandb", False) and wandb:
                wandb.log({
                    "leaderboard/ssim_acc4": ssim["acc4"],
                    "leaderboard/ssim_acc8": ssim["acc8"],
                    "leaderboard/ssim_mean": ssim["mean"],
                    "leaderboard/epoch":     epoch,
                    "leaderboard/time_min":  dt/60,
                }, step=epoch*dup_mul)
                                
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} ValSSIM = {val_ssim:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

        # ────────── epoch 루프 내부, val 계산·로그 이후 ──────────
        current_epoch = epoch + 1         # 사람 눈금 1-base
        if early_enabled and current_epoch in stage_table:
            req = stage_table[current_epoch]
            if val_ssim < req:
                print(f"[EarlyStop] Epoch {current_epoch}: "
                    f"val_ssim={val_ssim:.4f} < target={req:.4f}. 학습 중단!")
                break                     # for epoch 루프 탈출