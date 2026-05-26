"""ADE20K scene parsing dataset (Zhou et al., CVPR 2017 / IJCV 2019).

We use the SceneParse150 split: 20,210 train + 2,000 val images, 150 named
classes. Pixel labels are PNGs where pixel value v ∈ [1, 150] indexes a class
(0 = background / ignore).

`prepare.py` writes manifest entries of the form:

    {"image": "datasets/ade20k/images/train/ADE_train_00001234.jpg",
     "label_png": "datasets/ade20k/annotations/train/ADE_train_00001234.png",
     "classes_present": [3, 47, 81, 120],
     "class_names_present": ["sky", "road", "person", "bicycle"],
     "dataset": "ade20k",
     "split": "train"}

For stage 2 we sample K class names per image (mix of present + absent).
The single-class __getitem__ used for stage 1 (if PhraseCut is unavailable)
picks one random present class and returns its binary mask.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import SAMPreprocess, build_joint_transform, to_tensor_chw


class ADE20KDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        class_names_file: str | Path,
        image_size: int = 512,
        split: str = "train",
        queries_per_image: int = 8,
        negative_query_ratio: float = 0.25,
        augmentations: dict | None = None,
        return_multi: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.split = split
        self.queries_per_image = queries_per_image
        self.negative_query_ratio = negative_query_ratio
        self.return_multi = return_multi

        with self.manifest_path.open() as f:
            self.entries = [
                e for e in (json.loads(l) for l in f)
                if e.get("dataset") == "ade20k" and e.get("split") == split
            ]
        self.class_names: List[str] = Path(class_names_file).read_text().splitlines()
        # class_names is the 150-entry index-aligned name list (line i = class i+1).

        self.augment = build_joint_transform(image_size, augmentations) if (split == "train" and augmentations) else None
        self.sam_pre = SAMPreprocess(target_size=image_size)

    def __len__(self):
        return len(self.entries)

    def _sample_queries(self, present_ids: List[int]) -> Tuple[List[int], List[str]]:
        """Pick K class names: mix of present and absent."""
        K = self.queries_per_image
        n_neg = int(round(self.negative_query_ratio * K))
        n_pos = K - n_neg
        if not present_ids:
            chosen_pos: List[int] = []
            n_neg = K
        else:
            chosen_pos = random.choices(present_ids, k=n_pos) if n_pos > 0 else []
        all_ids = list(range(1, len(self.class_names) + 1))
        absent = [c for c in all_ids if c not in present_ids]
        chosen_neg = random.sample(absent, min(n_neg, len(absent))) if absent else []
        # pad with random absent classes if we still need more
        while len(chosen_pos) + len(chosen_neg) < K:
            chosen_neg.append(random.choice(absent if absent else all_ids))
        ids = chosen_pos + chosen_neg
        random.shuffle(ids)
        names = [self.class_names[i - 1] for i in ids]
        return ids, names

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        img = np.array(Image.open(e["image"]).convert("RGB"))
        label = np.array(Image.open(e["label_png"]))  # (H, W) uint8

        # joint augmentation (apply to label as mask)
        if self.augment is not None:
            img, label = self.augment(img, label)

        present = sorted(int(c) for c in np.unique(label) if int(c) != 0)

        if not self.return_multi:
            cls = random.choice(present) if present else random.randint(1, len(self.class_names))
            mask = (label == cls).astype(np.uint8)
            image_t = to_tensor_chw(img)
            mask_t = torch.from_numpy(mask).float()
            image_t, mask_t, meta = self.sam_pre(image_t, mask_t)
            return {"image": image_t, "mask": mask_t, "text": self.class_names[cls - 1],
                    "meta": meta, "dataset": "ade20k"}

        # multi-query path (stage 2)
        ids, texts = self._sample_queries(present)
        image_t = to_tensor_chw(img)
        image_t, _, meta = self.sam_pre(image_t, None)
        masks = []
        # build per-query masks at the original (augmented) size then resize via sam_pre
        for cls_id in ids:
            m = (label == cls_id).astype(np.uint8)
            mt = torch.from_numpy(m).float()
            _, mt, _ = self.sam_pre(to_tensor_chw(img), mt)  # piggyback on sam_pre's resize/pad
            masks.append(mt)
        masks_t = torch.stack(masks, dim=0)                # (K, H, W)
        return {"image": image_t, "masks": masks_t, "texts": texts,
                "meta": meta, "dataset": "ade20k"}
