# utils/learning/classifier_train_part.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path

# hydra 유틸리티 임포트
from hydra.utils import instantiate
from omegaconf import OmegaConf

from utils.data.load_data import create_data_loaders

# 도메인 이름과 인덱스를 매핑
DOMAIN_MAP = {
    "knee_x4": 0, "knee_x8": 1,
    "brain_x4": 2, "brain_x8": 3,
}
# 역방향 매핑도 생성
INV_DOMAIN_MAP = {v: k for k, v in DOMAIN_MAP.items()}


def train_classifier_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Classifier_Train", ncols=90, leave=False)
    # [FIX] 데이터로더가 8개 항목을 반환하므로, 8번째 변수(domain_indices)를 추가로 받아줍니다.
    # 이 함수에서는 사용하지 않으므로, _ (언더스코어)로 받아 무시해도 됩니다.
    for mask, kspace, target, maximum, fnames, slices, cats, domain_indices in pbar:
        kspace = kspace.to(device)
        labels = torch.tensor([DOMAIN_MAP[c] for c in cats], dtype=torch.long).to(device)

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
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Classifier_Val", ncols=90, leave=False)
    with torch.no_grad():
        # [FIX] train_classifier_epoch와 마찬가지로 8개 항목을 받도록 수정합니다.
        for mask, kspace, target, maximum, fnames, slices, cats, domain_indices in pbar:
            kspace = kspace.to(device)
            labels = torch.tensor([DOMAIN_MAP[c] for c in cats], dtype=torch.long).to(device)
            
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
        batch_size_override=cfg['batch_size'],
        isforward=False
    )
    val_loader = create_data_loaders(
        data_path=args.data_path_val, args=args, is_train=False, 
        batch_size_override=cfg['batch_size'],
        isforward=False
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
