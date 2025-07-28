# utils/data/load_data.py

import h5py, re
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate, BatchSampler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path

try:
    from utils.learning.classifier_train_part import DOMAIN_MAP, INV_DOMAIN_MAP
except ImportError:
    DOMAIN_MAP = {"knee": 0, "brain": 1}
    INV_DOMAIN_MAP = {v: k for k, v in DOMAIN_MAP.items()}

from utils.data.transforms import DataTransform
from utils.data.transform_wrapper import TransformWrapper
from utils.data.sampler import GroupByCoilBatchSampler

def padding_collate_fn(batch):
    items = list(zip(*batch))
    processed_items = []
    for item_list in items:
        elem = item_list[0]
        if isinstance(elem, torch.Tensor) and elem.dim() > 1:
            max_h = max(t.shape[-2] for t in item_list)
            max_w = max(t.shape[-1] for t in item_list)
            padded_tensors = []
            for tensor in item_list:
                h, w = tensor.shape[-2:]
                pad_h, pad_w = max_h - h, max_w - w
                padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
                padded_tensors.append(F.pad(tensor, padding, "constant", 0))
            processed_items.append(torch.stack(padded_tensors, 0))
        else:
            try:
                processed_items.append(default_collate(item_list))
            except (TypeError, RuntimeError):
                processed_items.append(item_list)
    return tuple(processed_items)

class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        for tr in self.transforms:
            mask, kspace, target, attrs, fname, slice_idx = tr(mask, kspace, target, attrs, fname, slice_idx)
        return mask, kspace, target, attrs, fname, slice_idx

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, is_test_or_leaderboard=False, classifier=None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.is_test_or_leaderboard = is_test_or_leaderboard
        self.classifier = classifier
        self.device = next(classifier.parameters()).device if classifier and hasattr(classifier, 'parameters') and list(classifier.parameters()) else torch.device('cpu')
        self.image_examples, self.kspace_examples = [], []
        def _cat_from_fname(fname: Path):
            organ = "brain" if "brain" in fname.name.lower() else "knee"
            acc = "x4" if re.search(r"_acc4_|x4|r04", fname.name, re.I) else "x8"
            return f"{organ}_{acc}"
        image_files, kspace_files = sorted(list(Path(root / "image").iterdir())), sorted(list(Path(root / "kspace").iterdir()))
        if not self.is_test_or_leaderboard:
            for fname in image_files:
                num_slices, cat = self._get_metadata(fname), _cat_from_fname(fname)
                self.image_examples += [(fname, i, cat) for i in range(num_slices)]
            for fname in kspace_files:
                num_slices, cat = self._get_metadata(fname), _cat_from_fname(fname)
                self.kspace_examples += [(fname, i, cat) for i in range(num_slices)]
        else:
            for fname in kspace_files:
                num_slices = self._get_metadata(fname)
                self.kspace_examples += [(fname, i, None) for i in range(num_slices)]
            self.image_examples = self.kspace_examples
        shape_map = {}
        for entry in self.kspace_examples:
            fname = entry[0]
            key = str(fname)
            if key not in shape_map:
                with h5py.File(fname, 'r') as hf:
                    try: shape_map[key] = tuple(hf[self.input_key].shape[1:])
                    except KeyError: continue
        self.coil_counts = [shape[0] for shape in (shape_map.get(str(e[0])) for e in self.kspace_examples) if shape]
        self.sample_shapes = [shape for shape in (shape_map.get(str(e[0])) for e in self.kspace_examples) if shape]
    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf: return hf[self.input_key].shape[0]
            if isinstance(self.target_key, str) and self.target_key in hf: return hf[self.target_key].shape[0]
            return hf[next(iter(hf.keys()))].shape[0]
    def __len__(self): return len(self.kspace_examples)
    @torch.no_grad()
    def _get_cat_with_classifier(self, kspace_data):
        if self.classifier is None: return "unknown", -1
        self.classifier.eval()
        kspace_complex = torch.from_numpy(kspace_data)
        kspace_real = torch.view_as_real(kspace_complex)
        kspace_tensor = kspace_real.unsqueeze(0).to(self.device)
        logits = self.classifier(kspace_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
        return INV_DOMAIN_MAP.get(pred_idx, "unknown"), pred_idx
    def __getitem__(self, i):
        kspace_fname, dataslice, cat = self.kspace_examples[i]
        with h5py.File(kspace_fname, "r") as hf:
            input_data, mask = hf[self.input_key][dataslice], np.array(hf["mask"])
        mask_density = np.mean(mask)
        acc_idx = 1 if mask_density < 0.16 else 0
        domain_idx = -1
        if self.is_test_or_leaderboard and self.classifier:
            organ, domain_idx = self._get_cat_with_classifier(input_data)
            acc_str = "x8" if acc_idx == 1 else "x4"
            cat = f"{organ}_{acc_str}"
        elif cat is not None:
            organ = cat.split('_')[0]
            domain_idx = DOMAIN_MAP.get(organ, -1)
        else:
            cat = "unknown"
        if self.is_test_or_leaderboard:
            target, attrs = -1, {}
        else:
            image_fname, _, _ = self.image_examples[i]
            if image_fname.name != kspace_fname.name: raise ValueError(f"Filename mismatch")
            with h5py.File(image_fname, "r") as hf:
                target, attrs = hf[self.target_key][dataslice], dict(hf.attrs)
        return (mask, input_data, target, attrs, kspace_fname.name, dataslice, cat, domain_idx, acc_idx)

def create_data_loaders(data_path, args, shuffle=False, isforward=False, augmenter=None, mask_augmenter=None, is_train=False,
                        domain_filter=None, classifier=None, batch_size_override=None, for_classifier=False):
    max_key_ = args.max_key if not isforward else -1
    target_key_ = args.target_key if not isforward else None
    transforms_list = []
    if augmenter and not isforward and is_train and not for_classifier:
        transforms_list.append(augmenter)
    if mask_augmenter and not isforward and is_train and not for_classifier:
        transforms_list.append(mask_augmenter)
    try:
        from utils.data.transforms import MaskApplyTransform
        transforms_list.append(MaskApplyTransform())
    except ImportError: print("Warning: MaskApplyTransform not found.")
    if getattr(args, 'use_crop', False):
        transforms_list.append(instantiate(args.centerCropPadding))
    transforms_list.append(DataTransform(isforward, max_key_))
    transform_chain = MultiCompose(transforms_list)
    raw_ds = SliceData(root=data_path, transform=lambda *x: x, input_key=args.input_key, target_key=target_key_, 
                       is_test_or_leaderboard=isforward, classifier=classifier)
    if domain_filter:
        from utils.data.domain_subset import DomainSubset
        raw_ds = DomainSubset(raw_ds, domain_filter)
    dup_cfg = getattr(args, "maskDuplicate", {"enable": False})
    if dup_cfg.get("enable", False) and not isforward and is_train and not for_classifier:
        cfg_clean = {k: v for k, v in dup_cfg.items() if k != "enable"}
        duped_ds = instantiate(OmegaConf.create(cfg_clean), base_ds=raw_ds, _recursive_=False)
    else:
        duped_ds = raw_ds
    data_storage = TransformWrapper(duped_ds, transform_chain)
    
    # [최종 수정] '상황인지형' collate 함수 로직
    is_prompt_model = 'prompt' in getattr(args, 'model', {}).get('_target_', '').lower()
    
    if isforward:
        collate_fn = padding_collate_fn
    elif for_classifier:
        # 분류기 학습: 9개 항목을 받아서 8개 항목(acc_idx 제외)을 반환
        base_collate_fn = instantiate(args.collator, _recursive_=False)
        def classifier_collate_fn(batch):
            domain_indices = [item[-2] for item in batch]
            base_items = [item[:-2] for item in batch] # 마지막 2개(domain_idx, acc_idx) 제외
            collated_base = base_collate_fn(base_items)
            return (*collated_base, torch.tensor(domain_indices, dtype=torch.long))
        collate_fn = classifier_collate_fn
    elif is_prompt_model:
        # Prompt 모델 학습: 9개 항목을 모두 반환
        base_collate_fn = instantiate(args.collator, _recursive_=False)
        def new_prompt_collate_fn(batch):
            acc_indices = [item[-1] for item in batch]
            domain_indices = [item[-2] for item in batch]
            base_items = [item[:-2] for item in batch]
            collated_base = base_collate_fn(base_items)
            return (*collated_base, torch.tensor(domain_indices, dtype=torch.long), torch.tensor(acc_indices, dtype=torch.long))
        collate_fn = new_prompt_collate_fn
    else:
        # 일반 모델(main.py) 학습: 9개 항목을 받아서 원래대로 7개 항목만 반환
        base_collate_fn = instantiate(args.collator, _recursive_=False)
        def original_collate_fn(batch):
            base_items = [item[:-2] for item in batch] # 마지막 2개(domain_idx, acc_idx) 버림
            return base_collate_fn(base_items)
        collate_fn = original_collate_fn

    batch_size = batch_size_override if batch_size_override else (args.val_batch_size if not is_train else args.batch_size)
    num_workers = args.num_workers

    if not isforward and hasattr(args, 'sampler'):
        sampler = instantiate(args.sampler, data_source=data_storage, sample_shapes=getattr(data_storage, "sample_shapes", None),
                              batch_size=batch_size, shuffle=shuffle, _recursive_=False)
    else:
        sampler = None

    if sampler and isinstance(sampler, BatchSampler):
        return DataLoader(dataset=data_storage, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        return DataLoader(dataset=data_storage, batch_size=batch_size, sampler=sampler, shuffle=shuffle if sampler is None else False,
                          num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)