"""PhraseCut dataset (Wu et al., CVPR 2020).

PhraseCut provides ~108K (image, referring-phrase, polygon mask) triples on
Visual Genome images. Stage 1 trains the single-mask head on this.

We expect `prepare.py` to have produced a JSONL manifest with one entry per
sample:

    {"image": "datasets/phrasecut/images/2367890.jpg",
     "polygons": [[x1,y1,x2,y2,...], ...],
     "text": "the red mug on the table",
     "dataset": "phrasecut",
     "split": "train"}

Mask is rasterized on-the-fly from the polygons (saves disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import SAMPreprocess, build_joint_transform, to_tensor_chw


def _rasterize_polygons(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Polygons in flat [x1,y1,x2,y2,...] format -> binary mask (H, W) uint8."""
    try:
        from pycocotools import mask as coco_mask
    except ImportError as e:
        raise ImportError("pip install pycocotools") from e
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    rles = coco_mask.frPyObjects(polygons, height, width)
    rle = coco_mask.merge(rles)
    m = coco_mask.decode(rle)
    return m.astype(np.uint8)


class PhraseCutDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 1024,
        split: str = "train",
        augmentations: dict | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.split = split
        rows = []
        with self.manifest_path.open() as f:
            for line in f:
                e = json.loads(line)
                if e.get("dataset") == "phrasecut" and e.get("split") == split:
                    rows.append(e)
        self.entries = rows
        self.augment = build_joint_transform(image_size, augmentations) if (split == "train" and augmentations) else None
        self.sam_pre = SAMPreprocess(target_size=image_size)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        img = np.array(Image.open(e["image"]).convert("RGB"))
        H, W = img.shape[:2]
        mask = _rasterize_polygons(e["polygons"], H, W)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        image_t = to_tensor_chw(img)
        mask_t = torch.from_numpy(mask).float()
        image_t, mask_t, meta = self.sam_pre(image_t, mask_t)
        return {
            "image": image_t,
            "mask": mask_t,
            "text": e["text"],
            "meta": meta,
            "dataset": "phrasecut",
        }
