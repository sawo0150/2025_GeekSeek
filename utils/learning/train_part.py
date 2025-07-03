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
from torchvision.utils import make_grid

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

import os

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    # start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    pbar = tqdm(enumerate(data_loader),
                total=len_loader,
                ncols=120,
                leave=False,
                desc=f"Epoch[{epoch:2d}/{args.num_epochs}]/")

    start_iter = time.perf_counter()
    for iter, data in pbar:

    # for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # if iter % args.report_interval == 0:
        #     print(
        #         f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
        #         f'Iter = [{iter:4d}/{len(data_loader):4d}] '
        #         f'Loss = {loss.item():.4g} '
        #         f'Time = {time.perf_counter() - start_iter:.4f}s',
        #     )
        #     start_iter = time.perf_counter()

        # --- tqdm & ETA ---------------------------------------------------
        avg = (time.perf_counter() - start_iter) / (iter + 1)
        pbar.set_postfix(loss=f"{loss.item():.4g}")
                        #  eta=f"{(len_loader - iter - 1) * avg/60:5.1f}m")

        # --- 배치 스칼라 W&B 로깅 -----------------------------------------
        if getattr(args, "use_wandb", False) and wandb:
            wandb.log({"train_loss": loss.item(),
                       "lr": optimizer.param_groups[0]['lr']},
                      step=epoch * len_loader + iter)
    # total_loss = total_loss / len_loader
    # return total_loss, time.perf_counter() - start_epoch

    epoch_time = time.perf_counter() - start_iter
    return total_loss / len_loader, epoch_time


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    # return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start
    # --- 샘플 이미지 (첫 파일 첫 슬라이스) ------------------------------
    sample_grid: Optional[torch.Tensor] = None
    if reconstructions:
        first = next(iter(reconstructions))
        recon = torch.from_numpy(reconstructions[first][0]).unsqueeze(0)
        target = torch.from_numpy(targets[first][0]).unsqueeze(0)
        sample_grid = make_grid(torch.cat([recon, target], 0),
                                nrow=2, normalize=True)
    return (metric_loss, num_subjects,
            reconstructions, targets, None, sample_grid,
            time.perf_counter() - start)

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

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    print(args.data_path_train)
    print(args.data_path_val)
    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)
    
    val_loss_log = np.empty((0, 2))

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        # val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        (val_loss_raw, num_subjects, reconstructions, targets,inputs, sample_grid, val_time) = validate(args, model, val_loader)

        # SSIMLoss→SSIM 으로 변환 (1-loss)
        val_ssim = 1 - (val_loss_raw / num_subjects)
        val_loss_scalar = (val_loss_raw / num_subjects).item()
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss_scalar]]), axis=0)

        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        # train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss_scalar).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        
        # ---------------- W&B 에폭 로그 ----------------
        if getattr(args, "use_wandb", False) and wandb:
            wandb.log({"epoch": epoch,
                    #    "train_loss": train_loss,
                       "val_loss": val_loss.item(),
                       "val_ssim": val_ssim.item(),
                       "lr": optimizer.param_groups[0]['lr']})
            # if sample_grid is not None:
            #     wandb.log({"recon_sample":
            #                wandb.Image(sample_grid,
            #                            caption=f"epoch{epoch}")})
            if is_new_best:
                # best 모델 아티팩트 업로드
                wandb.save(str(args.exp_dir / "best_model.pt"))
                                
        print(
            # f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            # f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
