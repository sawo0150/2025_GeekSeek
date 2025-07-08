import h5py, re
import random
from utils.data.transforms import DataTransform, CenterCropOrPad
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate, get_class
from pathlib import Path
import numpy as np
from torch.utils.data import default_collate
from torch.utils.data.sampler import BatchSampler

# +++ custom multi-input Compose +++
class MultiCompose:
    """
    mask, kspace, target, attrs, fname, slice 6-tuple을
    self.transforms 리스트에 있는 transform들에
    차례대로 넘겨주고 최종 결과를 리턴합니다..
    """
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
        self.forward = forward          # test/submit 모드
        self.image_examples = []        # [(fname, slice, cat)]
        self.kspace_examples = []       # [(fname, slice, cat)]

        # ❶ organ/acc 추출 함수 ---------------------------------------------
        def _cat(fname: Path):
            organ = "brain" if "brain" in fname.name.lower() else "knee"
            acc   = "x4"   if re.search(r"_acc4_|x4|r04", fname.name, re.I) else "x8"
            return f"{organ}_{acc}"
        
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                cat = _cat(fname)
                self.image_examples += [
                    (fname, slice_ind, cat) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)
            cat = _cat(fname)
            self.kspace_examples += [
                (fname, slice_ind, cat) for slice_ind in range(num_slices)
            ]

        # --- coil_counts 미리 계산 ---
        # 파일 단위로 한 번씩만 열어서, self.input_key의 두 번째 차원이 C(coils)인 걸 이용
        coil_map = {}
        for fname, _, _ in self.kspace_examples:
            key = str(fname)
            if key not in coil_map:
                with h5py.File(fname, 'r') as hf:
                    arr = hf[self.input_key]
                    # arr.shape == (num_slices, C, H, W) 라고 가정
                    coil_map[key] = arr.shape[1]
        # 인덱스 순서대로 coil count 리스트
        self.coil_counts = [coil_map[str(f)] for f, _, _ in self.kspace_examples]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _, cat = self.image_examples[i]
        kspace_fname, dataslice, cat = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        sample = self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)
        return (*sample, cat)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    transforms = []

    # (0) Dynamic augmentation (추후 통합 예정)
    # # aug_tr = instantiate(args.aug)
    #     # aug_tr,   # TODO: MRaugment 통합 시 여기에 추가  현재 epoch를 받아야 해서 train_epoch 내부에서 처리

    # (1) Mask 적용 및 k-space numpy 반환
    from utils.data.transforms import MaskApplyTransform
    transforms.append(MaskApplyTransform())

    # (2) Spatial crop (토글)
    if getattr(args, 'use_crop', False):
        transforms.append(CenterCropOrPad(target_size=tuple(args.crop_size)))

    # # (3) Coil compression (토글)
    # if getattr(args, "compress", None):
    #     comp_tr = instantiate(args.compress)
    #     transforms.append(comp_tr)

    # (4) Tensor 변환 및 real/imag 스택
    transforms.append(DataTransform(isforward, max_key_))

    transform = MultiCompose(transforms)


    data_storage = SliceData(
        root=data_path,
        transform=transform,
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    # 1) collate_fn
    collate_fn = instantiate(args.collator, _recursive_=False)

    # 2) sampler 인스턴스 하나만 만들기
    sampler = instantiate(
        args.sampler,
        data_source=data_storage,
        coil_counts=getattr(data_storage, "coil_counts", None),
        batch_size=args.batch_size,
        shuffle=shuffle,
        _recursive_=False
    )

    # 3) DataLoader 에 넘겨줄 인자 결정
    #    sampler 가 BatchSampler 계열이면 batch_sampler=, 아니면 sampler= 로
    if isinstance(sampler, BatchSampler):
        return DataLoader(
            dataset=data_storage,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    else:
        return DataLoader(
            dataset=data_storage,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )