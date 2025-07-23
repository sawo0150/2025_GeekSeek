# utils/data/load_data.py

import h5py, re
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate, BatchSampler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path

# [PROMPT-MR] utils.learning.classifier_train_part가 먼저 생성/실행된다고 가정
try:
    from utils.learning.classifier_train_part import DOMAIN_MAP, INV_DOMAIN_MAP
except ImportError:
    # 임시 정의 (classifier_train_part가 없을 경우 대비)
    DOMAIN_MAP = {
        "knee_x4": 0, "knee_x8": 1,
        "brain_x4": 2, "brain_x8": 3,
    }
    INV_DOMAIN_MAP = {v: k for k, v in DOMAIN_MAP.items()}

from utils.data.transforms import DataTransform
from utils.data.transform_wrapper import TransformWrapper
from utils.data.sampler import GroupByCoilBatchSampler


class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        for tr in self.transforms:
            mask, kspace, target, attrs, fname, slice_idx = tr(
                mask, kspace, target, attrs, fname, slice_idx
            )
        return mask, kspace, target, attrs, fname, slice_idx

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, is_test_or_leaderboard=False, classifier=None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.is_test_or_leaderboard = is_test_or_leaderboard
        self.classifier = classifier # [PROMPT-MR] 분류기 모델
        self.device = next(classifier.parameters()).device if classifier and hasattr(classifier, 'parameters') and list(classifier.parameters()) else torch.device('cpu')

        self.image_examples = []
        self.kspace_examples = []

        def _cat_from_fname(fname: Path):
            organ = "brain" if "brain" in fname.name.lower() else "knee"
            acc = "x4" if re.search(r"_acc4_|x4|r04", fname.name, re.I) else "x8"
            return f"{organ}_{acc}"
        
        image_files = sorted(list(Path(root / "image").iterdir()))
        kspace_files = sorted(list(Path(root / "kspace").iterdir()))

        # 학습/검증 시에는 파일명에서 카테고리를 가져옴
        if not self.is_test_or_leaderboard:
            for fname in image_files:
                num_slices = self._get_metadata(fname)
                cat = _cat_from_fname(fname)
                self.image_examples += [(fname, i, cat) for i in range(num_slices)]
            for fname in kspace_files:
                num_slices = self._get_metadata(fname)
                cat = _cat_from_fname(fname)
                self.kspace_examples += [(fname, i, cat) for i in range(num_slices)]
        else: # 테스트/리더보드 시에는 카테고리 정보 없음
            for fname in kspace_files:
                num_slices = self._get_metadata(fname)
                # 카테고리 자리에 None 플레이스홀더
                self.kspace_examples += [(fname, i, None) for i in range(num_slices)]
            # image_examples는 kspace_examples와 길이를 맞추기 위해 복사
            self.image_examples = self.kspace_examples
        
        shape_map = {}
        for entry in self.kspace_examples:
            fname = entry[0]
            key = str(fname)
            if key not in shape_map:
                with h5py.File(fname, 'r') as hf:
                    try:
                        arr = hf[self.input_key]
                        shape_map[key] = tuple(arr.shape[1:])
                    except KeyError:
                        print(f"Warning: input_key '{self.input_key}' not found in {fname}. Skipping.")
                        continue
        
        self.coil_counts = [shape[0] for shape in (shape_map[str(e[0])] for e in self.kspace_examples if str(e[0]) in shape_map)]
        self.sample_shapes = [shape_map[str(entry[0])] for entry in self.kspace_examples if str(entry[0]) in shape_map]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf:
                return hf[self.input_key].shape[0]
            if isinstance(self.target_key, str) and self.target_key in hf:
                return hf[self.target_key].shape[0]
            first_ds = next(iter(hf.keys()))
            return hf[first_ds].shape[0]

    def __len__(self):
        return len(self.kspace_examples)

    @torch.no_grad()
    def _get_cat_with_classifier(self, kspace_data):
        if self.classifier is None:
            return "unknown", -1
            
        self.classifier.eval()
        # kspace_data (C, H, W, 2) numpy array
        # numpy -> tensor, add batch dim, move to device
        kspace_tensor = torch.from_numpy(kspace_data).unsqueeze(0).to(self.device)
        
        logits = self.classifier(kspace_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
        return INV_DOMAIN_MAP.get(pred_idx, "unknown"), pred_idx

    def __getitem__(self, i):
        kspace_fname, dataslice, cat = self.kspace_examples[i]
        
        with h5py.File(kspace_fname, "r") as hf:
            input_data = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])
        
        domain_idx = -1 # 기본값

        if self.is_test_or_leaderboard and self.classifier:
            cat, domain_idx = self._get_cat_with_classifier(input_data)
        elif cat is not None:
            domain_idx = DOMAIN_MAP.get(cat, -1)
        else:
            cat = "unknown"

        if self.is_test_or_leaderboard:
            target = -1
            attrs = {}
        else:
            image_fname, _, _ = self.image_examples[i]
            if image_fname.name != kspace_fname.name:
                raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        attrs['cat'] = cat
        sample = self.transform(mask, input_data, target, attrs, kspace_fname.name, dataslice)
        
        return (*sample, domain_idx)

def create_data_loaders(data_path, args, shuffle=False, isforward=False, 
                        augmenter=None, mask_augmenter=None, is_train=False,
                        domain_filter=None, 
                        classifier=None,
                        batch_size_override=None):
    
    is_test_or_leaderboard = isforward

    if not isforward:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = None

    transforms_list = []
    if augmenter is not None and not isforward:
        transforms_list.append(augmenter)
    if mask_augmenter is not None and not isforward:
        transforms_list.append(mask_augmenter)
    
    # MaskApplyTransform을 찾기 위해 임포트 경로 확인
    try:
        from utils.data.transforms import MaskApplyTransform
        transforms_list.append(MaskApplyTransform())
    except ImportError:
        print("Warning: MaskApplyTransform not found.")

    if getattr(args, 'use_crop', False):
        transforms_list.append(instantiate(args.centerCropPadding))
    transforms_list.append(DataTransform(isforward, max_key_))
    transform_chain = MultiCompose(transforms_list)

    raw_ds = SliceData(
        root=data_path, transform=lambda *x: x,
        input_key=args.input_key, target_key=target_key_, 
        is_test_or_leaderboard=is_test_or_leaderboard,
        classifier=classifier
    )
    
    if domain_filter:
        from utils.data.domain_subset import DomainSubset
        raw_ds = DomainSubset(raw_ds, domain_filter)

    dup_cfg = getattr(args, "maskDuplicate", {"enable": False})
    if dup_cfg.get("enable", False) and not isforward and is_train:
        dup_cfg_clean = OmegaConf.create({k:v for k,v in dup_cfg.items() if k!="enable"})
        duped_ds = instantiate(dup_cfg_clean, base_ds=raw_ds, _recursive_=False)
    else:
        duped_ds = raw_ds

    data_storage = TransformWrapper(duped_ds, transform_chain)
    
    if isforward:
        # 테스트/리더보드 모드: domain_idx 포함 8개 튜플을 받지만, default_collate는 튜플을 그대로 배치로 만듦
        collate_fn = default_collate
    else:
        base_collate_fn = instantiate(args.collator, _recursive_=False)
        def collate_with_domain_idx(batch):
            base_items = [item[:-1] for item in batch] # 마지막 domain_idx 제외
            domain_indices = [item[-1] for item in batch]
            collated_base = base_collate_fn(base_items)
            return (*collated_base, torch.tensor(domain_indices, dtype=torch.long))
        collate_fn = collate_with_domain_idx

    batch_size = batch_size_override if batch_size_override else (args.val_batch_size if not is_train else args.batch_size)
    num_workers = args.num_workers

    if isforward and isinstance(data_storage.data, SliceData):
        sampler = GroupByCoilBatchSampler(
            data_source=data_storage, sample_shapes=getattr(data_storage.data, "sample_shapes", None),
            batch_size=batch_size, shuffle=False)
    elif not isforward and hasattr(args, 'sampler'):
         sampler = instantiate(
            args.sampler, data_source=data_storage,
            sample_shapes=getattr(data_storage.data, "sample_shapes", None),
            batch_size=batch_size, shuffle=shuffle, _recursive_=False)
    else:
        sampler = None # fallback to default sampler

    if sampler and isinstance(sampler, BatchSampler):
        # BatchSampler를 사용하면 batch_size와 shuffle은 sampler가 관리하므로 None으로 설정
        return DataLoader(
            dataset=data_storage, batch_sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        # 일반 Sampler나 기본 Sampler 사용
        return DataLoader(
            dataset=data_storage, batch_size=batch_size, sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
