"""Equal-source/group/window sampling and source-only scalar RMS for TII."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.identifiers import validate_identifiers


def grouped_indices(groups: list) -> dict:
    values = validate_identifiers(groups, 'group')
    return {g: [i for i, v in enumerate(values) if v == g] for g in sorted(set(values))}


class SourceRounds(Dataset):
    """One item is one joint update: B windows from each source.

    Per-round PCG64 streams make the schedule reproducible after checkpoint
    restoration without depending on worker scheduling. Inputs are source-train
    window inventories, with one group, record and file identity per window.
    """
    def __init__(self, sources: dict, *, rounds: int, batch_size: int, seed: int):
        if not sources or rounds < 1 or batch_size < 1 or seed < 0:
            raise ValueError('nonempty sources and positive rounds/batch size required')
        validate_identifiers(sources, 'dataset')
        self.sources = sources
        self.rounds, self.batch_size, self.seed = rounds, batch_size, seed
        self.groups = {}
        for source, windows in sources.items():
            groups = grouped_indices(windows['group'])
            n = len(windows['group'])
            for field in ('recording_id', 'file_id'):
                validate_identifiers(windows[field], field)
            for field in ('x', 'incremental', 'availability', 'y', 'recording_id', 'file_id', 'role'):
                if len(windows[field]) != n:
                    raise ValueError(f'{source}: {field} does not match window inventory')
            if any(role != 'source_train' for role in windows['role']):
                raise ValueError('sampler may only access source_train windows')
            self.groups[source] = groups

    def __len__(self) -> int:
        return self.rounds

    def __getitem__(self, round_index: int) -> dict:
        if round_index < 0 or round_index >= self.rounds:
            raise IndexError(round_index)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, round_index])))
        result = {}
        for source in sorted(self.sources):
            windows, groups = self.sources[source], self.groups[source]
            names = list(groups)
            indices = [groups[names[int(rng.integers(len(names)))]] for _ in range(self.batch_size)]
            selected = [rows[int(rng.integers(len(rows)))] for rows in indices]
            result[source] = {
                key: value[selected] if torch.is_tensor(value) else [value[i] for i in selected]
                for key, value in windows.items()
            }
            result[source]['window_index'] = selected
        return result


def fit_source_rms(sources: dict) -> float:
    """Equal dataset/group/window mean square, without fitted centering.

    This intentionally reuses the training-inventory validation of SourceRounds;
    support, query or source-validation rows cannot enter this estimator.
    """
    inventory = SourceRounds(sources, rounds=1, batch_size=1, seed=0)
    source_means = []
    for source, windows in sources.items():
        c, p = windows['x'].double(), windows['incremental'].double()
        a = windows['availability']
        if c.ndim != 3 or c.shape != p.shape or a.shape != (len(c),):
            raise ValueError('RMS expects matching [windows,K,P] coordinates and availability')
        if not ((a == 0) | (a == 1)).all():
            raise ValueError('RMS availability must be binary')
        union = c + torch.where(a[:, None, None].bool(), p, torch.zeros_like(p))
        if not torch.isfinite(union).all():
            raise ValueError('source-train admitted union must be finite')
        energy = union.square().mean(dim=(1, 2))
        source_means.append(torch.stack([energy[idx].mean() for idx in inventory.groups[source].values()]).mean())
    rms = torch.stack(source_means).mean().sqrt().item()
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError('source RMS is zero or nonfinite')
    return rms
