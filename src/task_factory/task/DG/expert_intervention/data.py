"""Explicit feature-pack axes, physical-unit splits, and source-only normalization."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .audit import validate_groups


def load_pack(path: Path, *, normalization: dict[str, np.ndarray] | None = None,
              partitions: tuple[str, ...] = ('train', 'val', 'match', 'test')) -> dict[str, np.ndarray]:
    """Load the declared population, fitting scaling only when none is supplied.

    ``group`` is the domain-specific aggregation unit. Optional ``specimen`` is
    the physical identity spanning domains; it must never cross partitions.
    Their presence does not establish independent sampling.
    """
    with np.load(path, allow_pickle=False) as f:
        declared = set(f['split'].astype(str))
        if declared != set(partitions):
            raise ValueError(f'Expected only {partitions} partitions, found {sorted(declared)}; separate source and audit packs')
        data = {k: f[k] for k in f.files}
    for key in ('views', 'raw', 'compatibility', 'labels', 'group', 'split', 'domain',
                'probe_views', 'control_views', 'probe_raw', 'control_raw'):
        if key not in data:
            raise ValueError(f'Missing feature-pack field: {key}')
    x = data['views']
    if x.ndim != 3:
        raise ValueError('views must have shape [N,K,D]')
    n, k, d = x.shape
    if n < 1 or d < 1 or k < 2 or data['raw'].shape != (n, d) or data['compatibility'].shape != (n, k):
        raise ValueError('Require K>=2, raw [N,D], compatibility [N,K]')
    for key in ('probe_views', 'control_views'):
        if data[key].shape != (n, k, k, d):
            raise ValueError(f'{key} must have shape [N,R=K,K,D]')
    for key in ('probe_raw', 'control_raw'):
        if data[key].shape != (n, k, d):
            raise ValueError(f'{key} must have shape [N,R=K,D]')
    for key in ('views', 'raw', 'compatibility', 'probe_views', 'control_views', 'probe_raw', 'control_raw'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'{key}: nonfinite input')
    if np.abs(data['compatibility']).max() > 1:
        raise ValueError('compatibility must be a declared score in [-1,1], not labels')
    for key in ('labels', 'group', 'split', 'domain'):
        if data[key].shape != (n,):
            raise ValueError(f'{key} must have shape [N]')
    for key in ('group', 'split', 'domain'):
        data[key] = data[key].astype(str)
    y = data['labels']
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError('labels must be contiguous nonnegative integer classes')
    if 'class_names' in data:
        names = data['class_names'].astype(str)
        if names.ndim != 1 or len(names) < 2 or len(set(names)) != len(names) or np.any(names == '') or y.max() >= len(names):
            raise ValueError('class_names must define distinct nonempty names for every label index')
        data['class_names'] = names
    elif not np.array_equal(np.unique(y), np.arange(y.max()+1)):
        raise ValueError('Without class_names, labels must be contiguous nonnegative integer classes')
    if 'role_names' in data:
        names = data['role_names'].astype(str)
        if names.shape != (k,) or len(set(names)) != k or np.any(names == ''):
            raise ValueError('role_names must define distinct nonempty names for each declared role/view')
        data['role_names'] = names
    validate_groups(data['group'], data['split'], data['domain'], y)
    if 'specimen' in data:
        specimens = data['specimen'].astype(str)
        if specimens.shape != (n,) or np.any(specimens == ''):
            raise ValueError('specimen must contain a nonempty physical identity for each sample')
        for specimen in np.unique(specimens):
            ix = specimens == specimen
            if len(np.unique(data['split'][ix])) != 1 or len(np.unique(y[ix])) != 1:
                raise ValueError(f'specimen {specimen}: physical identity crosses partitions or labels')
        for group in np.unique(data['group']):
            if len(np.unique(specimens[data['group'] == group])) != 1:
                raise ValueError(f'group {group}: aggregation cannot mix physical specimens')
        for specimen in np.unique(specimens):
            for domain in np.unique(data['domain'][specimens == specimen]):
                ix = (specimens == specimen) & (data['domain'] == domain)
                if len(np.unique(data['group'][ix])) != 1:
                    raise ValueError('A specimen/domain cell must have exactly one aggregation group')
        data['specimen'] = specimens
    train = data['split'] == 'train'
    classes = np.arange(len(data['class_names'])) if 'class_names' in data else np.unique(y)
    if train.any() and not np.array_equal(np.unique(y[train]), classes):
        raise ValueError('Closed-set protocol requires all classes in training')
    if normalization is None and not train.any():
        raise ValueError('A pack without training samples requires saved checkpoint normalization')
    # Fit normalization on training data only, then apply the SAME transform to probes.
    for clean, probe_keys in (('raw', ('probe_raw', 'control_raw')),
                              ('views', ('probe_views', 'control_views'))):
        if normalization is None:
            mean = data[clean][train].mean(0)
            sd = data[clean][train].std(0)
            sd = np.where(sd > 0, sd, 1.0)
        else:
            mean = np.asarray(normalization[f'{clean}_mean'])
            sd = np.asarray(normalization[f'{clean}_scale'])
            if mean.shape != data[clean].shape[1:] or sd.shape != mean.shape:
                raise ValueError(f'{clean}: checkpoint normalization shape does not match the pack')
            if not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
                raise ValueError(f'{clean}: checkpoint normalization must be finite with positive scale')
        data[clean] = ((data[clean]-mean)/sd).astype('float32')
        for key in probe_keys:
            data[key] = ((data[key]-mean)/sd).astype('float32')
        data[f'{clean}_mean'], data[f'{clean}_scale'] = mean, sd
    return data


def validate_dg(data: dict[str, np.ndarray], *, source_domains: set[str] | None = None) -> None:
    """Only source domains may enter training, selection, or role fitting."""
    domains = {part: set(data['domain'][data['split'] == part])
               for part in ('train', 'val', 'match', 'test')}
    sources = domains['train'] if source_domains is None else source_domains
    if len(sources) < 2:
        raise ValueError('Multi-source DG requires at least two training domains')
    for part in ('val', 'match'):
        if not domains[part] <= sources:
            raise ValueError(f'{part} domains must belong to the training source domains')
    if domains['test'] & set.union(sources, domains['val'], domains['match']):
        raise ValueError('Target domains cannot enter training, validation, or role matching')
