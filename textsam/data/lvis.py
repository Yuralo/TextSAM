"""LVIS dataset (Gupta, Dollár, Girshick, CVPR 2019).

LVIS = "Large Vocabulary Instance Segmentation" — 1,203 named classes on the
COCO 2017 image set. We reuse the user's existing COCO 2017 train/val images;
LVIS only adds annotation JSONs.

`prepare.py` writes manifest entries of the form:

    {"image":  "datasets/coco/train2017/000000123456.jpg",
     "annotations": [
        {"category_id": 481, "category_name": "tabby cat", "segmentation": <RLE or polygons>}, ...
     ],
     "image_h": 480, "image_w": 640,
     "dataset": "lvis",
     "split":   "train"}

We rasterize masks on the fly for the K queries we sample per image.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import SAMPreprocess, build_joint_transform, to_tensor_chw


def _decode_seg(seg, h: int, w: int) -> np.ndarray:
    from pycocotools import mask as coco_mask
    if isinstance(seg, list):  # polygons
        rles = coco_mask.frPyObjects(seg, h, w)
        rle = coco_mask.merge(rles)
    elif isinstance(seg, dict) and "counts" in seg:
        rle = seg
        if isinstance(rle["counts"], list):
            rle = coco_mask.frPyObjects(rle, h, w)
    else:
        return np.zeros((h, w), dtype=np.uint8)
    return coco_mask.decode(rle).astype(np.uint8)


class LVISDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 512,
        split: str = "train",
        queries_per_image: int = 8,
        negative_query_ratio: float = 0.25,
        category_pool: List[str] | None = None,
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
                if e.get("dataset") == "lvis" and e.get("split") == split
            ]
        # category_pool is the *full* LVIS class-name vocabulary used to sample
        # negative queries (classes not present in this image).
        self.category_pool = category_pool or self._collect_category_pool()
        self.augment = build_joint_transform(image_size, augmentations) if (split == "train" and augmentations) else None
        self.sam_pre = SAMPreprocess(target_size=image_size)

    def _collect_category_pool(self) -> List[str]:
        names = set()
        for e in self.entries:
            for a in e["annotations"]:
                names.add(a["category_name"])
        return sorted(names)

    def __len__(self):
        return len(self.entries)

    def _sample_queries(self, anns: List[dict]) -> tuple[List[dict], List[str]]:
        K = self.queries_per_image
        n_neg = int(round(self.negative_query_ratio * K))
        n_pos = K - n_neg
        if not anns:
            chosen_pos: List[dict] = []
            n_neg = K
        else:
            chosen_pos = random.choices(anns, k=n_pos) if n_pos > 0 else []
        present_names = {a["category_name"] for a in anns}
        absent_names = [n for n in self.category_pool if n not in present_names]
        chosen_neg = random.sample(absent_names, min(n_neg, len(absent_names))) if absent_names else []
        while len(chosen_pos) + len(chosen_neg) < K:
            chosen_neg.append(random.choice(absent_names if absent_names else self.category_pool))
        # Wrap negatives as "ann" dicts with no segmentation so __getitem__ produces zero masks.
        chosen_neg_anns = [{"category_name": n, "segmentation": None} for n in chosen_neg]
        all_anns = chosen_pos + chosen_neg_anns
        random.shuffle(all_anns)
        texts = [a["category_name"] for a in all_anns]
        return all_anns, texts

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        img = np.array(Image.open(e["image"]).convert("RGB"))
        H, W = img.shape[:2]
        # Build a combined per-class semantic mask before augmentation, so flips/crops stay in sync.
        anns = e["annotations"]

        if not self.return_multi:
            ann = random.choice(anns) if anns else None
            mask = _decode_seg(ann["segmentation"], H, W) if ann else np.zeros((H, W), np.uint8)
            text = ann["category_name"] if ann else random.choice(self.category_pool)
            if self.augment is not None:
                img, mask = self.augment(img, mask)
            image_t = to_tensor_chw(img)
            mask_t = torch.from_numpy(mask).float()
            image_t, mask_t, meta = self.sam_pre(image_t, mask_t)
            return {"image": image_t, "mask": mask_t, "text": text, "meta": meta, "dataset": "lvis"}

        sampled_anns, texts = self._sample_queries(anns)
        # rasterize each query's mask at original size, then run a single joint augment that
        # stacks them into channels. We expand the mask to a multi-channel image via albumentations.
        per_q_masks = []
        for a in sampled_anns:
            seg = a.get("segmentation")
            if seg is None:
                per_q_masks.append(np.zeros((H, W), dtype=np.uint8))
            else:
                per_q_masks.append(_decode_seg(seg, H, W))
        stacked = np.stack(per_q_masks, axis=-1)  # (H, W, K) so albumentations treats it as masks

        if self.augment is not None:
            try:
                import albumentations as A  # noqa
                # albumentations Compose `masks=` accepts list of single-channel masks
                # but we passed `mask=` earlier. Switch interface here:
                img_aug, masks_aug = self._augment_multi(img, per_q_masks)
                img = img_aug
                stacked = np.stack(masks_aug, axis=-1)
            except ImportError:
                pass

        image_t = to_tensor_chw(img)
        image_t, _, meta = self.sam_pre(image_t, None)
        out_masks = []
        for k in range(stacked.shape[-1]):
            mt = torch.from_numpy(stacked[..., k]).float()
            _, mt, _ = self.sam_pre(to_tensor_chw(img), mt)
            out_masks.append(mt)
        masks_t = torch.stack(out_masks, dim=0)
        return {"image": image_t, "masks": masks_t, "texts": texts, "meta": meta, "dataset": "lvis"}

    def _augment_multi(self, image: np.ndarray, masks: List[np.ndarray]):
        import albumentations as A
        # Rebuild a Compose that supports multi-mask via additional_targets.
        # Inexpensive; keep tiny per-call to stay simple.
        size = self.image_size
        ops = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        ]
        additional = {f"mask_{i}": "mask" for i in range(len(masks))}
        pipe = A.Compose(ops, additional_targets=additional)
        kwargs = {f"mask_{i}": m for i, m in enumerate(masks)}
        out = pipe(image=image, **kwargs)
        out_masks = [out[f"mask_{i}"] for i in range(len(masks))]
        return out["image"], out_masks
