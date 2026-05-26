# Inference

After training, run TextSAM with a single image and a single word/phrase to
get a binary mask of the named object.

## Command line

```
python -m textsam.inference.predict \
    --image  examples/dog.jpg \
    --word   "dog" \
    --checkpoint checkpoints/stage2/best.pt \
    --model-config configs/model.yaml \
    --out    out/dog_mask.png
```

Outputs:

- `out/dog_mask.png` — uint8 binary mask at the original image resolution.
- `out/dog_mask_viz.png` — RGB overlay (image + translucent red mask + contour).

The convenience wrapper does the same:

```
bash scripts/infer.sh examples/dog.jpg "dog"
bash scripts/infer.sh examples/street.jpg "stop sign"
bash scripts/infer.sh examples/kitchen.jpg "the red mug on the table"
```

Both **single nouns** ("dog", "cat", "stop sign") and **short referring
phrases** ("the red mug on the table") work — Stage 1 was trained on phrases
from PhraseCut, Stage 2 on bare class names from ADE20K and LVIS.

## Python API

```python
from PIL import Image
import yaml
from pathlib import Path

from textsam.models import TextSAM
from textsam.inference import predict_mask, overlay_mask
from textsam.utils.ckpt import load_checkpoint

cfg = yaml.safe_load(Path("configs/model.yaml").read_text())
model = TextSAM.from_config(cfg).to("cuda")
load_checkpoint("checkpoints/stage2/best.pt", model, strict=False)

image = Image.open("examples/dog.jpg")
mask = predict_mask(model, image, "dog", device="cuda")     # (H, W) uint8 0/255
viz  = overlay_mask(image, mask // 255)                     # PIL.Image
viz.save("out/dog_viz.png")
```

`predict_mask(..., return_logits=True)` returns float probabilities in
`[0, 1]` instead of a thresholded mask — useful for downstream tasks that
want to set their own threshold or combine masks.

## Multi-object inference

To segment several named objects in one image, call `predict_mask` once per
word. Each call shares the (cached) image encoding cost — for batched
efficiency, use the underlying `TextSAM.forward_multi_query` directly:

```python
images = ...                                # (1, 3, 512, 512), SAM-preprocessed
texts_per_image = [["dog", "person", "bicycle", "sky"]]
masks, iou = model.forward_multi_query(images, texts_per_image)
# masks: (1, 4, 1, 512, 512); iou: (1, 4)
```

## Tips

- For best quality, run at 1024² (Stage 1 resolution). The model is robust to
  inference at 512² (Stage 2 resolution) too, but boundaries are sharper at
  1024².
- The CLI handles arbitrary input image sizes via `SAMPreprocess` — images are
  resized so the longer side becomes the model's `image_size`, then padded to
  square, then masks are cropped back to the original aspect ratio before
  resampling to original resolution.
- If you have multiple GPUs you can wrap the model in `torch.nn.DataParallel`
  for a batched-inference server; the single-prompt forward pass is
  thread-safe.

## References

See `docs/citations.bib`.
