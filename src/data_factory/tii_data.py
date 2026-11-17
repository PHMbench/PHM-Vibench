"""TII source inventories through native metadata/H5 and physical projection."""
from __future__ import annotations

from pathlib import Path
from numbers import Real
import tempfile

import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .H5DataDict import H5DataDict
from .data_utils import MetadataAccessor, read_metadata_table
from .tii_sampling import SourceRounds, fit_source_rms, make_episode
from .tii_physical import project_window
from src.utils.identifiers import validate_identifiers
from src.utils.label_ontology import validate_metadata_label_ontology


def _read_projected_windows(selected, records, args_data, scratch, natural, roles, expected_datasets):
    """Shared H5, physical projection, and identity-preserving window path."""
    by_id = records.set_index('Id')
    collections = {role: {} for role in roles}
    window_inventory = []
    for name, rows in selected.groupby('Name'):
        reader = H5DataDict(str(scratch / f'{name}.h5'))
        try:
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
                if abs(n_float-round(n_float)) > 1e-8:
                    raise ValueError('physical duration does not give an integer raw window length')
                n = int(round(n_float))
                if n < 1 or len(signal) < n:
                    raise ValueError('record is too short for the declared physical interval')
                collection = collections[spec['role']].setdefault(source, {
                    k: [] for k in ('x', 'incremental', 'availability', 'y', 'file_id',
                                    'recording_id', 'group', 'role')})
                for start in range(0, len(signal)-n+1, n):
                    projected = project_window(signal[start:start+n],
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
                        window_start=start, window_end=start+n))
        finally:
            reader.close()
    for collection in collections.values():
        if set(collection) != set(expected_datasets):
            raise ValueError('every declared dataset needs windows in every requested role')
        for windows in collection.values():
            for field in ('x', 'incremental', 'availability'):
                windows[field] = torch.stack(windows[field])
            windows['y'] = torch.tensor(windows['y'], dtype=torch.long)
    return collections, window_inventory


def initialize_tii_data(factory, args_data, args_task):
    """Build one source-training protocol; never invoke raw-reader cache repair."""
    if args_data.use_cache is not True or args_data.normalization != 'source_rms':
        raise ValueError('tii_joint requires explicit verified H5 reuse and source_rms normalization')
    source_ids = validate_identifiers(args_task.source_system_ids, 'dataset')
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(args_task.target_system_id):
        raise ValueError('source_system_ids and native selected system IDs must match exactly')
    if args_data.seed != args_task.sampling_seed:
        raise ValueError('sampling seed must agree between Data and Task')
    root = Path(args_data.data_dir)
    factory.metadata = factory._init_metadata(args_data)
    selected = factory.metadata.df[factory.metadata.df.Dataset_id.isin(source_ids)].copy()
    if set(selected.Dataset_id) != set(source_ids):
        raise ValueError('metadata does not contain every declared source')
    validate_metadata_label_ontology(selected, group_field='Dataset_id', require_labels=True)
    records = read_metadata_table(Path(args_data.record_inventory))
    for field in ('Id', 'recording_id', 'group'):
        validate_identifiers(records[field], field)
    channels = validate_identifiers(records.channel, 'channel')
    if any(not isinstance(channel, Real) or channel < 0 or
           channel != int(channel) for channel in channels):
        raise ValueError('channel must be a nonnegative integer without truncation')
    if records.Id.duplicated().any() or set(records.Id) != set(selected.Id):
        raise ValueError('record inventory must contain every selected metadata Id exactly once')
    if not records.role.isin(['source_train', 'source_val']).all():
        raise ValueError('source inventory contains a target/support/query role')
    if records[['dataset_id', 'recording_id']].duplicated().any():
        raise ValueError('(dataset_id, recording_id) must identify one original record')
    natural = args_data.evidence_kind == 'natural_acquisition'
    if natural:
        qualification = pd.read_csv(args_data.qualification_file)
        for source in source_ids:
            rows = qualification[qualification.dataset_id == source]
            if len(rows) != 1 or rows.eligible.iloc[0] != True:
                raise ValueError(f'source {source} is not qualified; no real micro-run')
        if len(source_ids) < 2:
            raise ValueError('J1 micro-run requires at least two qualified sources')
    elif args_data.evidence_kind != 'tensor_fixture':
        raise ValueError('evidence_kind must explicitly distinguish natural acquisition and tensor fixture')
    for source in source_ids:
        source_records = records[records.dataset_id == source]
        train = set(source_records.loc[source_records.role == 'source_train', 'group'])
        val = set(source_records.loc[source_records.role == 'source_val', 'group'])
        if not train or not val or train & val:
            raise ValueError('source train/validation require nonempty disjoint physical groups')
    train = set(records.loc[records.role == 'source_train', 'group'])
    val = set(records.loc[records.role == 'source_val', 'group'])
    if train & val:
        raise ValueError('source train/validation physical groups overlap across datasets')
    # Complete cache-key checks precede read-only links. A fresh per-invocation
    # directory avoids an old aggregate cache and never calls cache rebuild code.
    cache_root = Path(args_data.cache_dir)
    if cache_root.resolve() == root.resolve():
        raise ValueError('cache_dir must be separate from immutable source data')
    cache_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='tii-', dir=cache_root))
    factory.cache_path = str(scratch)
    for name, rows in selected.groupby('Name'):
        path = root / f'{name}.h5'
        with h5py.File(path, 'r') as h5:
            missing = [fid for fid in rows.Id if str(fid) not in h5]
            if missing:
                raise ValueError(f'{name}: selected IDs absent from H5: {missing}')
        (scratch / path.name).symlink_to(path.resolve())
    factory.target_metadata = MetadataAccessor(selected, key_column='Id')
    collections, window_inventory = _read_projected_windows(
        selected, records, args_data, scratch, natural,
        ('source_train', 'source_val'), source_ids)
    factory.source_rms = fit_source_rms(collections['source_train'])
    factory.window_inventory = window_inventory
    factory.train_dataset = SourceRounds(collections['source_train'], rounds=args_data.rounds,
        batch_size=args_data.source_batch_size, seed=args_data.seed)
    # Validation visits the complete predeclared window inventory once, with
    # homogeneous local-label batches; Task aggregates group means after all batches.
    validation = []
    for source, windows in collections['source_val'].items():
        for start in range(0, len(windows['y']), args_data.source_batch_size):
            end = start + args_data.source_batch_size
            batch = {key: value[start:end] for key, value in windows.items()}
            batch['source'] = source
            validation.append(batch)
    factory.val_dataset = validation
    factory.test_dataset = None
    factory.train_loader = DataLoader(factory.train_dataset, batch_size=None, num_workers=0)
    factory.val_loader = DataLoader(validation, batch_size=None, num_workers=0)
    factory.test_loader = None
    factory.data = None


def load_tii_target_data(args_data, metadata_file, record_inventory, *, target_dataset_id,
                         source_dataset_ids, source_group_ids):
    """Load one fixed target episode using the same projection as source data.

    Call only after encoder, source RMS, checkpoint and source-only readout
    selection have been fixed. Target IDs, labels and groups are used solely to
    construct the fixed support/query episode; none enters the source factory.
    An excluded record sharing a support group is never projected or evaluated.
    """
    target_dataset_id = validate_identifiers([target_dataset_id], 'target dataset')[0]
    sources = validate_identifiers(source_dataset_ids, 'source dataset')
    if not sources or target_dataset_id in sources:
        raise ValueError('target dataset must be excluded from nonempty source datasets')
    if args_data.use_cache is not True or args_data.normalization != 'source_rms':
        raise ValueError('target loading requires the same verified H5 and source_rms protocol')
    natural = args_data.evidence_kind == 'natural_acquisition'
    if natural:
        qualification = pd.read_csv(args_data.qualification_file)
        matches = qualification[qualification.dataset_id == target_dataset_id]
        if len(matches) != 1 or matches.eligible.iloc[0] != True:
            raise ValueError('target is not qualified for the declared physical support')
    elif args_data.evidence_kind != 'tensor_fixture':
        raise ValueError('evidence kind must explicitly identify the data population')
    selected = read_metadata_table(Path(metadata_file))
    records = read_metadata_table(Path(record_inventory))
    if selected.empty or set(selected.Dataset_id) != {target_dataset_id}:
        raise ValueError('target metadata must contain exactly the declared target dataset')
    validate_metadata_label_ontology(selected, group_field='Dataset_id', require_labels=True)
    for field in ('Id', 'recording_id', 'group'):
        validate_identifiers(records[field], field)
    channels = validate_identifiers(records.channel, 'channel')
    if any(not isinstance(channel, Real) or channel < 0 or channel != int(channel) for channel in channels):
        raise ValueError('channel must be a nonnegative integer without truncation')
    if (records.Id.duplicated().any() or selected.Id.duplicated().any()
            or set(records.Id) != set(selected.Id) or set(records.dataset_id) != {target_dataset_id}
            or records.recording_id.duplicated().any() or not records.role.eq('target').all()):
        raise ValueError('target inventory must identify each target record exactly once with role target')
    source_groups = validate_identifiers(source_group_ids, 'source physical group')
    validate_identifiers(list(source_groups) + records.group.tolist(), 'physical group')
    if not source_groups or set(source_groups) & set(records.group):
        raise ValueError('source and target physical groups must be nonempty and disjoint')
    labels = selected.set_index('Id').Label
    episode = make_episode([dict(recording_id=row.recording_id, group=row.group,
                                 label=int(labels.loc[row.Id])) for row in records.itertuples()],
                           shots=5, seed=1729)
    role_by_record = {row['recording_id']: row['role'] for row in episode}
    records = records.copy()
    records['role'] = records.recording_id.map(role_by_record)
    records = records[records.role.isin(['support', 'query'])].copy()
    selected = selected[selected.Id.isin(records.Id)].copy()
    root = Path(args_data.data_dir)
    for name, rows in selected.groupby('Name'):
        with h5py.File(root / f'{name}.h5', 'r') as h5:
            missing = [fid for fid in rows.Id if str(fid) not in h5]
            if missing:
                raise ValueError(f'{name}: selected target IDs absent from H5: {missing}')
    collections, inventory = _read_projected_windows(
        selected, records, args_data, root, natural, ('support', 'query'), [target_dataset_id])
    return collections, inventory, episode
