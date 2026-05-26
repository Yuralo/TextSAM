"""Build unified JSONL manifests for stage-1 and stage-2 training.

Each manifest line is a self-contained JSON object describing one trainable
sample (for PhraseCut) or one trainable image with its annotations (for
ADE20K / LVIS).

Outputs:
    datasets/manifest_stage1.jsonl   <- PhraseCut only (single-mask samples)
    datasets/manifest_stage2.jsonl   <- ADE20K + LVIS (multi-query images)

Usage:
    python -m textsam.data.prepare --stage 1
    python -m textsam.data.prepare --stage 2 --coco-root /path/to/coco
    python -m textsam.data.prepare --stage all --coco-root /path/to/coco
    python -m textsam.data.prepare --check       # verify referenced files exist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from tqdm import tqdm


DATASETS_DIR = Path("datasets")


# -------------------- PhraseCut -> stage 1 manifest --------------------

def iter_phrasecut(split: str) -> Iterator[dict]:
    refer_path = DATASETS_DIR / "phrasecut" / f"refer_{split}.json"
    img_dir = DATASETS_DIR / "phrasecut" / "images"
    if not refer_path.exists():
        print(f"[warn] {refer_path} missing — run `textsam-download --dataset phrasecut` first")
        return
    data = json.loads(refer_path.read_text())
    # Map "train"/"val"/"test" to our split names; PhraseCut's test we treat as val.
    out_split = {"train": "train", "val": "val", "test": "val"}[split]
    for r in data:
        img_id = r["image_id"]
        image_path = img_dir / f"{img_id}.jpg"
        if not image_path.exists():
            continue
        # PhraseCut stores polygons as list[list[float]] per instance; one phrase per instance.
        polygons = r.get("Polygons") or r.get("polygons")
        if not polygons:
            continue
        # PhraseCut polygon format is sometimes nested per-instance; flatten conservatively.
        flat_polygons = []
        for poly in polygons:
            if isinstance(poly[0], list):
                # list of [x,y,...] coords
                flat_polygons.extend([p for p in poly if len(p) >= 6])
            else:
                if len(poly) >= 6:
                    flat_polygons.append(poly)
        if not flat_polygons:
            continue
        yield {
            "image": str(image_path),
            "polygons": flat_polygons,
            "text": r.get("phrase") or r.get("Phrase"),
            "dataset": "phrasecut",
            "split": out_split,
        }


def build_stage1_manifest():
    out = DATASETS_DIR / "manifest_stage1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with out.open("w") as f:
        for split in ("train", "val", "test"):
            for entry in tqdm(iter_phrasecut(split), desc=f"PhraseCut {split}"):
                f.write(json.dumps(entry) + "\n")
                n_total += 1
    print(f"Stage-1 manifest: {n_total} entries -> {out}")


# -------------------- ADE20K -> stage 2 manifest entries --------------------

def iter_ade20k() -> Iterator[dict]:
    root = DATASETS_DIR / "ade20k" / "ADEChallengeData2016"
    names_file = DATASETS_DIR / "ade20k" / "class_names.txt"
    if not root.exists() or not names_file.exists():
        print("[warn] ADE20K not prepared — run `textsam-download --dataset ade20k`")
        return
    class_names = names_file.read_text().splitlines()
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise

    for split_dir, out_split in (("training", "train"), ("validation", "val")):
        img_dir = root / "images" / split_dir
        ann_dir = root / "annotations" / split_dir
        for img_path in tqdm(sorted(img_dir.glob("*.jpg")), desc=f"ADE20K {split_dir}"):
            ann_path = ann_dir / (img_path.stem + ".png")
            if not ann_path.exists():
                continue
            arr = np.array(Image.open(ann_path))
            present = sorted(int(v) for v in np.unique(arr) if 0 < int(v) <= len(class_names))
            yield {
                "image": str(img_path),
                "label_png": str(ann_path),
                "classes_present": present,
                "class_names_present": [class_names[c - 1] for c in present],
                "dataset": "ade20k",
                "split": out_split,
            }


# -------------------- LVIS -> stage 2 manifest entries --------------------

def iter_lvis(coco_root: Path) -> Iterator[dict]:
    lroot = DATASETS_DIR / "lvis"
    splits = [("lvis_v1_train.json", "train", "train2017"),
              ("lvis_v1_val.json",   "val",   "val2017")]
    for ann_name, out_split, coco_split in splits:
        ann_path = lroot / ann_name
        if not ann_path.exists():
            print(f"[warn] {ann_path} missing — run `textsam-download --dataset lvis`")
            continue
        data = json.loads(ann_path.read_text())
        cats = {c["id"]: c.get("synonyms", [c["name"]])[0].replace("_", " ") for c in data["categories"]}
        anns_by_image: dict[int, list] = {}
        for a in data["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)
        for img in tqdm(data["images"], desc=f"LVIS {out_split}"):
            file_name = img["coco_url"].rsplit("/", 1)[-1] if "coco_url" in img else img.get("file_name", "")
            local_img = coco_root / coco_split / file_name
            if not local_img.exists():
                continue
            img_anns = []
            for a in anns_by_image.get(img["id"], []):
                cat_name = cats.get(a["category_id"], "object")
                img_anns.append({
                    "category_id": a["category_id"],
                    "category_name": cat_name,
                    "segmentation": a["segmentation"],
                })
            if not img_anns:
                continue
            yield {
                "image": str(local_img),
                "annotations": img_anns,
                "image_h": img["height"],
                "image_w": img["width"],
                "dataset": "lvis",
                "split": out_split,
            }


def build_stage2_manifest(coco_root: str | None):
    out = DATASETS_DIR / "manifest_stage2.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with out.open("w") as f:
        for entry in iter_ade20k():
            f.write(json.dumps(entry) + "\n")
            n_total += 1
        if coco_root:
            for entry in iter_lvis(Path(coco_root)):
                f.write(json.dumps(entry) + "\n")
                n_total += 1
        else:
            print("[note] --coco-root not given; LVIS rows skipped. ADE20K-only stage 2.")
    print(f"Stage-2 manifest: {n_total} entries -> {out}")


# -------------------- check --------------------

def check_manifests():
    for stage in (1, 2):
        m = DATASETS_DIR / f"manifest_stage{stage}.jsonl"
        if not m.exists():
            print(f"[stage {stage}] not built")
            continue
        n_ok = 0
        n_bad = 0
        with m.open() as f:
            for line in f:
                e = json.loads(line)
                if Path(e["image"]).exists() and (
                    "label_png" not in e or Path(e["label_png"]).exists()
                ):
                    n_ok += 1
                else:
                    n_bad += 1
        print(f"[stage {stage}] {n_ok} ok, {n_bad} missing -> {m}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--coco-root", default=None)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.check:
        check_manifests()
        return
    if args.stage in ("1", "all"):
        build_stage1_manifest()
    if args.stage in ("2", "all"):
        build_stage2_manifest(args.coco_root)


if __name__ == "__main__":
    main()
