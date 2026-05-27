"""Dataset downloaders for PhraseCut, ADE20K, and LVIS.

Usage:
    python -m textsam.data.download --dataset all
    python -m textsam.data.download --dataset phrasecut --limit 100  # smoke-test
    python -m textsam.data.download --dataset ade20k
    python -m textsam.data.download --dataset lvis --coco-root /path/to/coco

PhraseCut images come from Visual Genome — we download only the images
referenced by PhraseCut annotations to keep the footprint small.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


DATASETS_DIR = Path("datasets")
CHECKPOINTS_DIR = Path("checkpoints")


def _download(url: str, dest: Path, desc: str | None = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists ({dest.stat().st_size/1e6:.1f} MB)")
        return
    desc = desc or dest.name
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def _unzip(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in tqdm(z.namelist(), desc=f"unzip {zip_path.name}"):
            z.extract(name, dest)


# -------------------- SAM checkpoint --------------------

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

def download_sam_checkpoint():
    dest = CHECKPOINTS_DIR / "sam_vit_b_01ec64.pth"
    _download(SAM_URL, dest, desc="SAM ViT-B")


# -------------------- PhraseCut --------------------

# PhraseCut annotations live on Google Drive (see upstream download_dataset.py).
PHRASECUT_GDRIVE_IDS = {
    "refer_train.json": "1qx-0q6r9r0YUGpoyT0B8HJKmUFWQDSu7",
    "refer_val.json":   "1UyojArOFPlsSeNbA9fHWjCjOOU-OCohG",
    "refer_test.json":  "1jrzXm1gcq6f5hNDeamZd0UmyyHUv61IZ",
}
# VG images by image_id: https://cs.stanford.edu/people/rak248/VG_100K_2/{id}.jpg
VG_PRIMARY = "https://cs.stanford.edu/people/rak248/VG_100K/{}.jpg"
VG_SECONDARY = "https://cs.stanford.edu/people/rak248/VG_100K_2/{}.jpg"


def _image_ids_from_phrasecut(splits: Iterable[str]) -> set[int]:
    ids: set[int] = set()
    for sp in splits:
        path = DATASETS_DIR / "phrasecut" / f"refer_{sp}.json"
        for r in json.loads(path.read_text()):
            ids.add(int(r["image_id"]))
    return ids


def download_phrasecut(limit: int | None = None, start_from: int = 0):
    import gdown  # local import: only needed when actually downloading PhraseCut

    root = DATASETS_DIR / "phrasecut"
    root.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]

    for sp in splits:
        dest = root / f"refer_{sp}.json"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {dest} already exists ({dest.stat().st_size/1e6:.1f} MB)")
            continue
        gdown.download(id=PHRASECUT_GDRIVE_IDS[dest.name], output=str(dest), quiet=False)

    img_dir = root / "images"
    img_dir.mkdir(exist_ok=True)
    image_ids = sorted(_image_ids_from_phrasecut(splits))
    if start_from:
        image_ids = image_ids[start_from:]
    if limit is not None:
        image_ids = image_ids[:limit]

    todo = [i for i in image_ids if not (img_dir / f"{i}.jpg").exists()]
    print(f"PhraseCut: {len(image_ids)} unique images (after offset {start_from}), {len(todo)} to fetch")

    def _fetch_one(img_id: int) -> bool:
        dest = img_dir / f"{img_id}.jpg"
        for url_fmt in (VG_PRIMARY, VG_SECONDARY):
            try:
                with urllib.request.urlopen(url_fmt.format(img_id), timeout=30) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 16)
                return True
            except Exception:
                dest.unlink(missing_ok=True)
                continue
        return False

    failed = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(_fetch_one, i) for i in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="VG images", unit="img"):
            if not fut.result():
                failed += 1
    if failed:
        print(f"  warning: {failed} images failed to download (both VG mirrors 404'd)")


# -------------------- ADE20K --------------------

ADE_ZIP_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

def download_ade20k():
    root = DATASETS_DIR / "ade20k"
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ADEChallengeData2016.zip"
    _download(ADE_ZIP_URL, zip_path, desc="ADE20K")
    if not (root / "ADEChallengeData2016").exists():
        _unzip(zip_path, root)
    # Write class-name file (objectInfo150.txt is included in the zip).
    info = root / "ADEChallengeData2016" / "objectInfo150.txt"
    names_out = root / "class_names.txt"
    if info.exists() and not names_out.exists():
        lines = info.read_text().splitlines()[1:]  # header
        names = [line.split("\t")[-1].split(",")[0].strip() for line in lines if line.strip()]
        names_out.write_text("\n".join(names))
    print(f"ADE20K ready at {root}")


# -------------------- LVIS --------------------

LVIS_TRAIN = "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
LVIS_VAL   = "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"


def download_lvis(coco_root: str | None = None):
    root = DATASETS_DIR / "lvis"
    root.mkdir(parents=True, exist_ok=True)
    for url in (LVIS_TRAIN, LVIS_VAL):
        zip_path = root / Path(url).name
        _download(url, zip_path, desc=zip_path.name)
        if not (root / zip_path.stem).exists():
            _unzip(zip_path, root)

    # Symlink COCO images under datasets/lvis/images if a coco_root is given.
    if coco_root:
        coco_root_p = Path(coco_root)
        target = root / "images"
        if target.exists() and target.is_symlink():
            target.unlink()
        if not target.exists():
            target.symlink_to(coco_root_p.resolve(), target_is_directory=True)
        print(f"LVIS images -> {target} -> {coco_root_p}")
    else:
        print(
            "LVIS annotations downloaded. Provide --coco-root <path-to-coco-2017-root>\n"
            "containing train2017/ and val2017/ to wire image paths in prepare.py."
        )


# -------------------- entry point --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["all", "sam", "phrasecut", "ade20k", "lvis"], required=True)
    p.add_argument("--limit", type=int, default=None, help="cap VG image downloads (for smoke test)")
    p.add_argument("--start-from", type=int, default=0, help="skip the first N image IDs (sorted) — for resuming")
    p.add_argument("--coco-root", default=None, help="path to your local COCO 2017 root (with train2017/ and val2017/)")
    args = p.parse_args()

    if args.dataset in ("all", "sam"):
        download_sam_checkpoint()
    if args.dataset in ("all", "phrasecut"):
        download_phrasecut(limit=args.limit, start_from=args.start_from)
    if args.dataset in ("all", "ade20k"):
        download_ade20k()
    if args.dataset in ("all", "lvis"):
        download_lvis(coco_root=args.coco_root)


if __name__ == "__main__":
    main()
