"""Projected physical-band windows for the DG/tii_joint task on the generic path.

This module owns the registered dataset adapter for projected windows: whole
records are read through the maintained H5 reader, projected onto orthogonal
common/incremental physical bands, and sliced into deterministic
whole-duration windows. It is the single home of that projection-windowing
core; task-side entrypoints compose this class instead of carrying private
windowing loops.

The classmethod :meth:`TII_projected_dataset.from_inventory` preserves the
historical reading order exactly (records grouped by file name, then table
order, then left-to-right whole-record windows), which the archived study
configurations rely on for bitwise reproduction.
"""
from __future__ import annotations

from numbers import Real
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from ...H5DataDict import H5DataDict
from ...tii_physical import project_window


class TII_projected_dataset(Dataset):
    """Stacked projected windows for one role, indexed by source dataset.

    Two constructions exist. ``from_inventory`` consumes explicit record
    inventories (role-declared groups, per-record physical specifications)
    through the maintained H5 reader. The standard factory signature consumes
    factory-read raw records and the same inventory metadata, for callers that
    route through the generic dataset dispatch. Both share one
    projection-windowing implementation.
    """

    def __init__(self):
        self.collections: dict = {}
        self.window_inventory: list[dict] = []

    def __len__(self) -> int:
        return sum(len(windows['y']) for windows in self.collections.values())

    @classmethod
    def _project_records(cls, selected, records, args_data, scratch, natural,
                         roles, expected_datasets, reader_backend):
        """Shared H5, physical projection, and identity-preserving window path.

        ``reader_backend(name)`` yields a context manager giving one record
        reader keyed by file id; the historical order of iteration is part of
        the reproduction contract and must not change.
        """
        instance = cls()
        by_id = records.set_index('Id')
        collections = {role: {} for role in roles}
        window_inventory = []
        for name, rows in selected.groupby('Name'):
            with reader_backend(name) as reader:
                for row in rows.to_dict('records'):
                    fid = row['Id']
                    spec = by_id.loc[fid].to_dict()
                    source = row['Dataset_id']
                    if spec['dataset_id'] != source:
                        raise ValueError('inventory dataset_id disagrees with metadata')
                    if spec['effective_rate_hz'] != row['Sample_rate']:
                        raise ValueError('effective sampling rate disagrees with metadata')
                    required_basis = 'documented_response' if natural else 'tensor_fixture'
                    if spec['support_basis'] != required_basis:
                        raise ValueError('support basis does not match declared evidence kind')
                    signal = torch.from_numpy(reader[fid])
                    if signal.ndim == 3 and signal.shape[-1] == 1:
                        signal = signal[..., 0]
                    if signal.ndim != 2:
                        raise ValueError('H5 must represent one original [samples,channels] record')
                    n_float = spec['effective_rate_hz'] * args_data.duration_s
                    if abs(n_float - round(n_float)) > 1e-8:
                        raise ValueError('physical duration does not give an integer raw window length')
                    n = int(round(n_float))
                    if n < 1 or len(signal) < n:
                        raise ValueError('record is too short for the declared physical interval')
                    collection = collections[spec['role']].setdefault(source, {
                        k: [] for k in ('x', 'incremental', 'availability', 'y', 'file_id',
                                        'recording_id', 'group', 'role')})
                    for start in range(0, len(signal) - n + 1, n):
                        projected = project_window(signal[start:start + n],
                            channel=int(spec['channel']), native_rate_hz=spec['native_rate_hz'],
                            effective_rate_hz=spec['effective_rate_hz'], grid_rate_hz=args_data.grid_rate_hz,
                            duration_s=args_data.duration_s, num_patches=args_data.num_patches,
                            patch_size=args_data.patch_size, common_bands_hz=args_data.common_bands_hz,
                            increment_bands_hz=args_data.increment_bands_hz,
                            common_support=spec['common_support'], increment_support=spec['increment_support'],
                            input_unit=spec['input_unit'], output_unit=args_data.output_unit,
                            unit_scale=spec['unit_scale'], support_basis=spec['support_basis'],
                            support_evidence=spec['support_evidence'])
                        collection['x'].append(projected['common'].float())
                        collection['incremental'].append(projected['incremental'].float())
                        collection['availability'].append(projected['increment_available'])
                        for key, value in (('y', int(row['Label'])), ('file_id', fid),
                                           ('recording_id', spec['recording_id']), ('group', spec['group']),
                                           ('role', spec['role'])):
                            collection[key].append(value)
                        window_inventory.append(dict(dataset=source, recording_id=spec['recording_id'],
                            group=spec['group'], role=spec['role'], file_id=fid, channel=int(spec['channel']),
                            window_start=start, window_end=start + n))
        for collection in collections.values():
            if set(collection) != set(expected_datasets):
                raise ValueError('every declared dataset needs windows in every requested role')
            for windows in collection.values():
                for field in ('x', 'incremental', 'availability'):
                    windows[field] = torch.stack(windows[field])
                windows['y'] = torch.tensor(windows['y'], dtype=torch.long)
        instance.collections = collections
        instance.window_inventory = window_inventory
        return instance

    @classmethod
    def from_inventory(cls, selected, records, args_data, scratch, natural, roles, expected_datasets):
        """Read whole records through the maintained H5 reader from a scratch link farm."""

        def backend(name):
            return H5DataDict(str(Path(scratch) / f'{name}.h5'))

        return cls._project_records(selected, records, args_data, scratch, natural,
                                    roles, expected_datasets, backend)

    @classmethod
    def from_root(cls, selected, records, args_data, root, natural, roles, expected_datasets):
        """Read whole records directly from the immutable source root."""

        def backend(name):
            return h5py.File(Path(root) / f'{name}.h5', 'r')

        return cls._project_records(selected, records, args_data, root, natural,
                                    roles, expected_datasets, backend)

    def validation_batches(self, batch_size, role='source_val'):
        """Complete predeclared window inventory once, homogeneous per-source batches."""
        batches = []
        for source, windows in self.collections[role].items():
            for start in range(0, len(windows['y']), batch_size):
                end = start + batch_size
                batch = {key: value[start:end] for key, value in windows.items()}
                batch['source'] = source
                batches.append(batch)
        return batches


# Generic dataset dispatch hook (dataset_task.{type}.{name}_dataset convention).
set_dataset = TII_projected_dataset
