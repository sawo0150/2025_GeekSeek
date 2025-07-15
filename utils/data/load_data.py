# utils/data/load_data.py

import h5py, re
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from torch.utils.data import default_collate
from torch.utils.data.sampler import BatchSampler
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
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        def _cat(fname: Path):
            organ = "brain" if "brain" in fname.name.lower() else "knee"
            acc   = "x4"   if re.search(r"_acc4_|x4|r04", fname.name, re.I) else "x8"
            return f"{organ}_{acc}"
        
        image_files = list(Path(root / "image").iterdir())
        kspace_files = list(Path(root / "kspace").iterdir())
        if not forward:
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                cat = _cat(fname)
                self.image_examples += [(fname, i, cat) for i in range(num_slices)]
            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)
                cat = _cat(fname)
                self.kspace_examples += [(fname, i, cat) for i in range(num_slices)]
        else:
            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)
                self.kspace_examples += [(fname, i) for i in range(num_slices)]
            self.image_examples = self.kspace_examples

        # coil_map = {}
        shape_map = {}
        for entry in self.kspace_examples:
            fname = entry[0]
            key = str(fname)
            # if key not in coil_map:
            if key not in shape_map:
                with h5py.File(fname, 'r') as hf:
                    arr = hf[self.input_key]
                    # coil_map[key] = arr.shape[1]
                    # arr.shape == (num_slices, C, H, W)
                    shape_map[key] = tuple(arr.shape[1:])

        # self.coil_counts = [coil_map[str(entry[0])] for entry in self.kspace_examples]
        # 기존 coil_counts 유지
        self.coil_counts = [shape[0] for shape in (shape_map[str(e[0])] for e in self.kspace_examples)]
        # 새로운 sample_shapes 속성 추가
        self.sample_shapes = [shape_map[str(entry[0])] for entry in self.kspace_examples]

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

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _, cat = self.image_examples[i]
            kspace_fname, dataslice, cat = self.kspace_examples[i]
        else:
            image_fname, _ = self.image_examples[i]
            kspace_fname, dataslice = self.kspace_examples[i]

        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input_data = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = {}
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            attrs['cat'] = cat
        sample = self.transform(mask, input_data, target, attrs, kspace_fname.name, dataslice)
        return (*sample, cat) if not self.forward else sample


def create_data_loaders(data_path, args, shuffle=False, isforward=False, 
                        augmenter=None, mask_augmenter=None, is_train=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = None

    transforms = []

    if augmenter is not None and not isforward:
        transforms.append(augmenter)

    if mask_augmenter is not None and not isforward:
        transforms.append(mask_augmenter)

    from utils.data.transforms import MaskApplyTransform
    transforms.append(MaskApplyTransform())

    # ✨ [수정] `use_noise_padding` 하이퍼파라미터를 읽어 transform에 전달
    if getattr(args, 'use_crop', False):
        # transforms.append(CenterCropOrPad(target_size=tuple(args.crop_size)))
        transforms.append(instantiate(args.centerCropPadding))

    # # (3) Coil compression (토글)
    # if getattr(args, "compressor", None):
    #     comp_tr = instantiate(args.compressor)
    #     transforms.append(comp_tr)

    # (4) Tensor 변환 및 real/imag 스택
    transforms.append(DataTransform(isforward, max_key_))

    transform_chain = MultiCompose(transforms)

    raw_ds = SliceData(
        root=data_path, transform=lambda *x: x,
        input_key=args.input_key, target_key=target_key_, forward=isforward
    )

    # 2)  *** Duplicate 적용 (crop 前) ***
    dup_cfg = getattr(args,"maskDuplicate",{"enable":False})
    if dup_cfg.get("enable",False) and not isforward and is_train: 
        dup_cfg_clean = OmegaConf.create({k:v for k,v in dup_cfg.items()
                                          if k!="enable"})
        duped_ds = instantiate(dup_cfg_clean, base_ds=raw_ds,
                               _recursive_=False)
    else:
        duped_ds = raw_ds

    data_storage = TransformWrapper(duped_ds, transform_chain)
    
    collate_fn = default_collate if isforward else instantiate(args.collator, _recursive_=False)

    # # Crop OFF시 batch_size 강제 1
    # use_crop = getattr(args, 'use_crop', False)
    batch_size = args.batch_size
    if not is_train:
        batch_size = args.val_batch_size
    # if not use_crop:
    #     if batch_size != 1:
    #         print("[WARN] use_crop=False 이므로 batch_size=1로 강제합니다.")
    #     batch_size = 1

    # 2) sampler 인스턴스 하나만 만들기
    if isforward:
        sampler = GroupByCoilBatchSampler(
            data_source=data_storage,
            sample_shapes=getattr(data_storage, "sample_shapes", None),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        sampler = instantiate(
            args.sampler, data_source=data_storage,
            sample_shapes=getattr(data_storage, "sample_shapes", None),
            batch_size=batch_size,
            shuffle=shuffle,
            _recursive_=False
        )

    num_workers = args.num_workers
    # if not use_crop and isforward:
    #     if num_workers != 0:
    #         print("[WARN] use_crop=False + isforward=True => num_workers=0으로 강제!")
    #     num_workers = 0

    # 3) DataLoader 에 넘겨줄 인자 결정
    #    sampler 가 BatchSampler 계열이면 batch_sampler=, 아니면 sampler= 로
    if isinstance(sampler, BatchSampler):
        return DataLoader(
            dataset=data_storage,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        return DataLoader(
            dataset=data_storage,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )