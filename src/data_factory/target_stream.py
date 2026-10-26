"""Read an explicitly ordered, preprocessed target-window archive; never fit a split."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TargetWindows(Dataset):
    """Archive row order is the stream order, not an inferred acquisition chronology."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"x", "y"}:
                raise ValueError("target archive requires exactly x and evaluator-only y")
            x, y = archive["x"], archive["y"]
        if x.dtype != np.float32 or x.ndim != 3 or min(x.shape) == 0:
            raise ValueError("target x must be nonempty float32 [N,L,C]; no casting or reshaping")
        if not np.isfinite(x).all():
            raise ValueError("target x contains non-finite observations")
        if y.dtype != np.int64 or y.shape != (len(x),):
            raise ValueError("target y must be int64 [N] in the frozen source label ontology")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # Traceability uses archive rows; only x is forwarded by B01/B02.
        return {"x": self.x[index], "y": self.y[index], "sample_id": index}


def build_target_stream(path: str | Path, *, batch_size: int) -> DataLoader:
    """Consume every prepared row once; preprocessing and population selection are upstream."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("target batch_size must be a positive integer")
    return DataLoader(TargetWindows(path), batch_size=batch_size, shuffle=False,
                      drop_last=False, num_workers=0)
