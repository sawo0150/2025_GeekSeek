# utils/learning/classifier_train_part.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.data.load_data import create_data_loaders

DOMAIN_MAP = {"knee": 0, "brain": 1}
INV_DOMAIN_MAP = {v: k for k, v in DOMAIN_MAP.items()}


def train_classifier_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct_preds, total_samples = 0, 0, 0
    
    pbar = tqdm(loader, desc="Classifier_Train", ncols=90, leave=False)
    for mask, kspace, target, maximum, fnames, slices, cats, domain_indices in pbar:
        kspace = kspace.to(device)
        labels = torch.tensor([DOMAIN_MAP[c.split('_')[0]] for c in cats], dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = model(kspace)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * kspace.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += kspace.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = correct_preds / total_samples if total_samples > 0 else 0
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    return total_loss / total_samples if total_samples > 0 else 0, correct_preds / total_samples if total_samples > 0 else 0

def validate_classifier(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct_preds, total_samples = 0, 0, 0
    
    pbar = tqdm(loader, desc="Classifier_Val", ncols=90, leave=False)
    with torch.no_grad():
        for mask, kspace, target, maximum, fnames, slices, cats, domain_indices in pbar:
            # [FIX] 디버깅 코드는 제거합니다.
            kspace = kspace.to(device)
            labels = torch.tensor([DOMAIN_MAP[c.split('_')[0]] for c in cats], dtype=torch.long).to(device)
            
            logits = model(kspace)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * kspace.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += kspace.size(0)
            
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            avg_acc = correct_preds / total_samples if total_samples > 0 else 0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    return total_loss / total_samples if total_samples > 0 else 0, correct_preds / total_samples if total_samples > 0 else 0

def train_classifier(args):
    """ Main function to train the domain classifier """
    print("\n" + "="*80)
    print("PHASE 1: Training Domain Classifier")
    print("="*80)
    
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    classifier_cfg_dict = getattr(args, "classifier", None)
    if not classifier_cfg_dict:
        raise ValueError("Classifier configuration ('classifier') not found in args.")
    
    classifier_model = instantiate(OmegaConf.create(classifier_cfg_dict))
    classifier_model.to(device)
    
    cfg = args.classifier_training
    optimizer = Adam(classifier_model.parameters(), lr=cfg['lr'])
    loss_fn = nn.CrossEntropyLoss()

    train_loader = create_data_loaders(
        data_path=args.data_path_train, args=args, is_train=True, 
        batch_size_override=cfg['batch_size'], isforward=False,
        for_classifier=True,
        shuffle=True  # [FIX] 학습 데이터는 반드시 셔플해야 합니다.
    )
    # 검증 데이터는 순서대로 평가하는 것이 일반적이므로 shuffle=False를 유지합니다.
    val_loader = create_data_loaders(
        data_path=args.data_path_val, args=args, is_train=False, 
        batch_size_override=cfg['batch_size'], isforward=False,
        for_classifier=True,
        shuffle=False
    )

    best_val_acc = 0.0
    exp_dir = Path(args.exp_dir)
    
    for epoch in range(cfg['epochs']):
        t0 = time.time()
        train_loss, train_acc = train_classifier_epoch(classifier_model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate_classifier(classifier_model, val_loader, loss_fn, device)
        dt = time.time() - t0

        print(
            f"Classifier Epoch {epoch+1:02d}/{cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"Time: {dt:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> New best validation accuracy: {best_val_acc:.4f}. Saving model...")
            torch.save(classifier_model.state_dict(), exp_dir / "best_classifier.pt")

    print("\nClassifier training finished. Best validation accuracy:", best_val_acc)
    classifier_model.load_state_dict(torch.load(exp_dir / "best_classifier.pt"))
    return classifier_model
