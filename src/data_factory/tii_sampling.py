"""Equal-source/group/window sampling and source-only scalar RMS for TII."""
from __future__ import annotations

import numpy as np
import torch
from numbers import Integral
from torch.utils.data import Dataset
from src.utils.identifiers import validate_identifiers


def grouped_indices(groups: list) -> dict:
    values = validate_identifiers(groups, 'group')
    return {g: [i for i, v in enumerate(values) if v == g] for g in sorted(set(values))}


def assign_physical_groups(groups, allocations, *, seed):
    """Assign whole physical groups using explicit group counts and PCG64.

    ``allocations`` is an ordered sequence of ``(role, group_count)`` pairs.
    Sorting unique identifiers before permutation makes the assignment invariant
    to record order. No label, signal, digest, or implicit rounding participates.
    The result has one role per input record, in its original order.
    """
    values = validate_identifiers(groups, 'physical_group_id')
    unique = sorted(set(values))
    entries = list(allocations)
    if not unique or not entries or isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('nonempty physical groups, allocations and a nonnegative integer seed required')
    if any(not isinstance(role, str) or not role.strip() or isinstance(count, bool)
           or not isinstance(count, Integral) or count < 1 for role, count in entries):
        raise ValueError('each role requires a nonempty name and a positive integer group count')
    names = [role for role, _ in entries]
    if len(set(names)) != len(names) or sum(count for _, count in entries) != len(unique):
        raise ValueError('unique roles and exact coverage of physical groups are required')
    permutation = np.random.Generator(np.random.PCG64(int(seed))).permutation(len(unique))
    assignment = {}
    start = 0
    for role, count in entries:
        for index in permutation[start:start + count]:
            assignment[unique[int(index)]] = role
        start += count
    roles = [assignment[group] for group in values]
    sets = [{group for group, value in zip(values, roles) if value == role} for role in names]
    if any(left & right for i, left in enumerate(sets) for right in sets[i + 1:]):
        raise RuntimeError('physical group overlap across assigned roles')
    return roles


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


EPISODE_SEED = 1729


class FixedEpisodeError(ValueError):
    """Keep the one failed draw available for per-class coverage reporting."""
    def __init__(self, message: str, episode: list[dict]):
        super().__init__(message)
        self.episode = episode



def make_episode(records, shots=5, seed=EPISODE_SEED):
    """Fixed metadata-only stratified episode; never redraw a bad split.

Input is one record ID, physical group and local label per original record.
Unselected records in a support group are excluded, not moved into query.
This function is used by the split custodian before model evaluation.
"""
    records = list(records)
    validate_identifiers((r['recording_id'] for r in records), 'recording_id')
    validate_identifiers((r['group'] for r in records), 'group')
    if any(not isinstance(r['label'], Integral) or isinstance(r['label'], (bool, np.bool_))
           or r['label'] < 0 for r in records):
        raise ValueError('contiguous integer local labels required')
    ordered = sorted(records, key=lambda r: r['recording_id'])
    ids = [r['recording_id'] for r in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError('unique original record IDs and nonempty physical groups required')
    labels = sorted({r['label'] for r in ordered})
    if (len(labels) < 2 or labels != list(range(len(labels)))
            or not isinstance(shots, Integral) or isinstance(shots, (bool, np.bool_)) or shots < 1):
        raise ValueError('contiguous local labels and positive shots required')
    rng = np.random.Generator(np.random.PCG64(seed))
    support_ids = set()
    for label in labels:
        candidates = [r['recording_id'] for r in ordered if r['label'] == label]
        if len(candidates) < shots:
            raise ValueError('insufficient original records for the fixed support budget')
        support_ids.update(rng.permutation(candidates)[:shots])
    support_groups = {r['group'] for r in ordered if r['recording_id'] in support_ids}
    result = [dict(r, role=('support' if r['recording_id'] in support_ids else
                 'excluded_same_group' if r['group'] in support_groups else 'query')) for r in ordered]
    query = [r for r in result if r['role'] == 'query']
    if any(len({r['group'] for r in query if r['label'] == c}) < 2 for c in labels):
        raise FixedEpisodeError('fixed draw leaves fewer than two query groups per class; do not redraw', result)
    return result
