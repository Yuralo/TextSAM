"""Merge ADE20K and LVIS for stage-2 training.

Both datasets implement the same `return_multi=True` interface:
    {"image": (3,H,W), "masks": (K,H,W), "texts": [K], "dataset": str}

We wrap them in a ConcatDataset and optionally build a WeightedRandomSampler
that rebalances LVIS samples (heavy long-tail) against ADE20K.
"""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler


class MergedSegDataset(ConcatDataset):
    """Just ConcatDataset with a `.dataset_of(idx)` helper."""

    def dataset_of(self, idx: int) -> str:
        # ConcatDataset's cumulative_sizes lets us locate which sub-dataset.
        for i, end in enumerate(self.cumulative_sizes):
            if idx < end:
                return self.datasets[i].__class__.__name__
        return "unknown"


def build_stage2_sampler(
    datasets: List,
    weights: List[float] | None = None,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Per-sub-dataset uniform-then-merge sampler.

    By default each sub-dataset contributes the same number of samples per
    epoch regardless of size — so LVIS's ~100K images don't drown out
    ADE20K's ~20K.
    """
    if weights is None:
        weights = [1.0] * len(datasets)
    sizes = [len(d) for d in datasets]
    total = sum(sizes)
    per_sample_weights = []
    for w, n in zip(weights, sizes):
        # weight per sample so each sub-dataset's total expected count = w/sum(w) * total
        per_sample_weights.extend([w / n] * n)
    per_sample_weights_t = torch.tensor(per_sample_weights, dtype=torch.double)
    if num_samples is None:
        num_samples = total
    return WeightedRandomSampler(per_sample_weights_t, num_samples=num_samples, replacement=True)
