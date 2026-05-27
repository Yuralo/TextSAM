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

import base64
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import SAMPreprocess, build_joint_transform, to_tensor_chw


def _rasterize_polygons(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Polygons -> binary mask (H, W) uint8.

    Accepts either flat `[x1,y1,x2,y2,...]` or nested `[[x,y],[x,y],...]` per
    polygon; pycocotools requires the flat form.
    """
    from pycocotools import mask as coco_mask
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    flat = []
    for poly in polygons:
        if poly and isinstance(poly[0], (list, tuple)):
            poly = [c for xy in poly for c in xy]
        if len(poly) >= 6:
            flat.append(poly)
    if not flat:
        return np.zeros((height, width), dtype=np.uint8)
    rles = coco_mask.frPyObjects(flat, height, width)
    rle = coco_mask.merge(rles)
    m = coco_mask.decode(rle)
    return m.astype(np.uint8)


def _decode_rle(rle_entry: dict) -> np.ndarray:
    """Decode the compact RLE we write in `prepare.py` (size + base64 counts)."""
    from pycocotools import mask as coco_mask
    counts = rle_entry["counts"]
    if isinstance(counts, str):
        counts = base64.b64decode(counts)
    rle = {"size": rle_entry["size"], "counts": counts}
    return coco_mask.decode(rle).astype(np.uint8)


class PhraseCutDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 1024,
        split: str = "train",
        augmentations: dict | None = None,
        subsample_per_image: int = 0,  # 0 = use all entries; 1 = one random phrase per image
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
        self.subsample_per_image = subsample_per_image if split == "train" else 0
        if self.subsample_per_image:
            # Group entry indices by image. One "epoch" then iterates over images,
            # sampling a fresh phrase per image each step.
            groups: dict[str, list[int]] = defaultdict(list)
            for i, e in enumerate(rows):
                groups[e["image"]].append(i)
            self.image_groups: list[list[int]] | None = list(groups.values())
        else:
            self.image_groups = None
        self.augment = build_joint_transform(image_size, augmentations) if (split == "train" and augmentations) else None
        self.sam_pre = SAMPreprocess(target_size=image_size)

    def __len__(self):
        return len(self.image_groups) if self.image_groups is not None else len(self.entries)

    def _pick_entry(self, idx: int) -> dict:
        if self.image_groups is None:
            return self.entries[idx]
        group = self.image_groups[idx]
        # Train: random sample. Val/test path bypasses this branch entirely.
        return self.entries[random.choice(group)]

    def __getitem__(self, idx: int):
        e = self._pick_entry(idx)
        img = np.array(Image.open(e["image"]).convert("RGB"))
        H, W = img.shape[:2]
        if "rle" in e:
            mask = _decode_rle(e["rle"])
        else:
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
