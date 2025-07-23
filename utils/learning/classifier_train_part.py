# utils/learning/classifier_train_part.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from pathlib import Path

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
    for mask, kspace, _, _, _, _, cats in pbar:
        kspace = kspace.to(device)
        # 'cats' 리스트를 인덱스 텐서로 변환
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
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_preds/total_samples:.4f}")

    return total_loss / total_samples, correct_preds / total_samples

def validate_classifier(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc="Classifier_Val", ncols=90, leave=False)
    with torch.no_grad():
        for mask, kspace, _, _, _, _, cats in pbar:
            kspace = kspace.to(device)
            labels = torch.tensor([DOMAIN_MAP[c] for c in cats], dtype=torch.long).to(device)
            
            logits = model(kspace)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * kspace.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += kspace.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_preds/total_samples:.4f}")

    return total_loss / total_samples, correct_preds / total_samples

def train_classifier(args, classifier_model):
    """ Main function to train the domain classifier """
    print("\n" + "="*80)
    print("PHASE 1: Training Domain Classifier")
    print("="*80)
    
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    classifier_model.to(device)
    
    cfg = args.classifier_training
    optimizer = Adam(classifier_model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    # 데이터 로더 생성 (분류기 학습용이므로 augment 비활성화)
    # create_data_loaders는 내부적으로 is_train=True일 때 shuffle을 활성화함
    train_loader = create_data_loaders(
        data_path=args.data_path_train, args=args, is_train=True, batch_size_override=cfg.batch_size
    )
    val_loader = create_data_loaders(
        data_path=args.data_path_val, args=args, is_train=False, batch_size_override=cfg.batch_size
    )

    best_val_acc = 0.0
    exp_dir = Path(args.exp_dir) # main.py에서 생성된 경로 활용
    
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, train_acc = train_classifier_epoch(classifier_model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate_classifier(classifier_model, val_loader, loss_fn, device)
        dt = time.time() - t0

        print(
            f"Classifier Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"Time: {dt:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> New best validation accuracy: {best_val_acc:.4f}. Saving model...")
            torch.save(classifier_model.state_dict(), exp_dir / "best_classifier.pt")

    print("\nClassifier training finished. Best validation accuracy:", best_val_acc)
    # 가장 성능 좋았던 모델 state를 로드하여 반환
    classifier_model.load_state_dict(torch.load(exp_dir / "best_classifier.pt"))
    return classifier_model
