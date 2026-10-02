"""P08 source-only data: native readers, explicit record roles, no target fitting."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from .data_factory import register_data_factory
from .explicit_data_factory import ExplicitDataFactory
from .data_utils import MetadataAccessor, read_metadata_table
from .H5DataDict import H5DataDict


def _specs(fields):
    values = [dict(v) if isinstance(v, dict) else vars(v).copy() for v in fields]
    names = [s['name'] for s in values]
    if not values or len(set(names)) != len(names):
        raise ValueError('Declare distinct physical condition fields')
    forbidden = {'id', 'file', 'label', 'labelname', 'dataset_id', 'domain_id', 'group', 'role', 'sample_rate', 'record_id', 'name', 'channel'}
    for s in values:
        if s['name'].lower() in forbidden or s['kind'] not in {'continuous', 'categorical'}:
            raise ValueError('Condition field is an identity/label/rate or has an unsupported kind')
        if any(not isinstance(s.get(k), str) or not s[k].strip() for k in ('unit', 'meaning', 'availability')):
            raise ValueError('Every physical field requires units, meaning and deployment availability')
    return values


def fit_conditions(source_train, fields):
    if source_train.empty or not source_train.role.eq('source_train').all():
        raise ValueError('Fit physical conditions on source_train records only')
    fitted = []
    for spec in _specs(fields):
        s = deepcopy(spec)
        values = source_train[s['name']]
        flags = source_train.get(s['name']+'__defaulted', pd.Series(False, index=source_train.index))
        if not flags.isin([True, False]).all():
            raise ValueError('Default indicators must be explicit booleans without missing values')
        column = values[~flags.astype(bool)].dropna()
        if column.empty:
            raise ValueError(f'No observed source-training values for {s["name"]}')
        if s['kind'] == 'continuous':
            v = pd.to_numeric(column, errors='raise').to_numpy(float)
            if not np.isfinite(v).all(): raise ValueError('Nonfinite physical measurements')
            q25, q75 = np.quantile(v, [.25, .75]); scale = q75-q25
            if scale <= 0: raise ValueError(f'Zero source IQR: {s["name"]}; specify a defensible field or calibration')
            s.update(median=float(np.median(v)), iqr=float(scale), lower=float(v.min()), upper=float(v.max()))
        else:
            s['vocabulary'] = sorted(set(str(v) for v in column))
        fitted.append(s)
    return fitted


def encode_conditions(frame, fitted):
    encoded = []
    for row in frame.to_dict('records'):
        parts = []
        for s in fitted:
            value = row[s['name']]
            defaulted = row.get(s['name']+'__defaulted', False)
            if defaulted not in (True, False, 0, 1): raise ValueError('Default indicator must be boolean')
            observed = not pd.isna(value) and not defaulted
            if s['kind'] == 'continuous':
                v = float(value) if observed else s['median']
                if not np.isfinite(v): raise ValueError('Nonfinite physical measurement')
                parts.extend([(v-s['median'])/s['iqr'], float(observed), float(defaulted),
                              float(observed and not s['lower'] <= v <= s['upper'])])
            else:
                word = str(value) if observed else None
                vector = [float(observed and word == item) for item in s['vocabulary']]
                vector += [float(not observed or word not in s['vocabulary']), float(observed), float(defaulted)]
                parts.extend(vector)
        encoded.append(parts)
    return torch.tensor(encoded, dtype=torch.float32)


def read_inventory(path, fields):
    # Preserve identities/category spellings (including leading zeros) in every
    # supported native table format; do not implement another CSV/Excel reader.
    types = {key: 'string' for key in ('Name','File','Group','record_id')}
    types.update({s['name']: 'string' for s in _specs(fields) if s['kind']=='categorical'})
    return read_metadata_table(path, dtype=types)


def validate_inventory(frame, args, *, evaluation=False, source_groups=()):
    required = {'Id','Name','File','Dataset_id','Label','LabelName','Group','record_id','role','Channel','Sample_rate'}
    required.update(s['name'] for s in _specs(args.condition_fields))
    if frame.empty or not required <= set(frame):
        raise ValueError(f'Nonempty native record inventory requires {sorted(required)}')
    identifiers = ['Id','Name','File','Dataset_id','LabelName','Group','record_id','role']
    if frame[identifiers].isna().any().any() or frame.Id.duplicated().any() or frame.record_id.duplicated().any():
        raise ValueError('Missing identity or duplicate original record/Id in inventory')
    frame = frame.copy()
    frame['Group'] = frame.Group.astype(str)
    frame['record_id'] = frame.record_id.astype(str)
    names = list(args.label_names)
    if len(names) < 2 or len(set(names)) != len(names): raise ValueError('Bind a common physical label ontology')
    for row in frame.to_dict('records'):
        y, channel, fs = row['Label'], row['Channel'], row['Sample_rate']
        if isinstance(y, bool) or not float(y).is_integer() or not 0 <= y < len(names) or row['LabelName'] != names[int(y)]:
            raise ValueError('Label integer/physical meaning mismatch; never re-encode automatically')
        if not float(channel).is_integer() or channel < 0 or not np.isfinite(fs) or fs <= 0:
            raise ValueError('Require an explicit channel and positive sampling rate')
    sources, target = set(args.source_system_ids), args.target_system_id
    if len(sources) < 2 or target in sources: raise ValueError('Require multiple sources and one unseen target')
    if evaluation:
        if set(frame.Dataset_id) != {target} or not frame.role.eq('target_test').all():
            raise ValueError('Evaluation inventory must contain only the declared target_test system')
        if set(frame.Group) & set(source_groups): raise ValueError('Target physical unit appears in source calibration/training')
    else:
        if set(frame.Dataset_id) != sources or not set(frame.role) <= {'source_train','source_val'}:
            raise ValueError('Source inventory contains target or unknown roles/systems')
        train = frame[frame.role == 'source_train']; val = frame[frame.role == 'source_val']
        if set(train.Group) & set(val.Group): raise ValueError('Physical-unit leakage across source training/validation')
        if set(train.Dataset_id) != sources or set(val.Dataset_id) != sources:
            raise ValueError('Each source must have nonempty train and validation physical units')
        if set(train.Label) != set(range(len(names))): raise ValueError('Source training does not cover the declared label space')
    return frame.copy()


def read_records(frame, args):
    """Delegate raw/H5 reading to the maintained Data Factory, without cache repair."""
    reader = object.__new__(ExplicitDataFactory)
    signals = []
    for row in frame.to_dict('records'):
        if args.storage == 'raw':
            _, array, _ = reader._read_single_data(row['Id'], row, args)
            # _validate_reader_output appends a singleton to native 2-D signals.
            if array.ndim == 3 and array.shape[-1] == 1: array = array[..., 0]
        elif args.storage == 'h5':
            h5 = H5DataDict(str(Path(args.data_dir) / f'{row["Name"]}.h5'))
            try: array = h5[row['Id']]
            finally: h5.close()
            if args.h5_layout == 'sample_channel_singleton' and array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            elif args.h5_layout != 'sample_channel':
                raise ValueError('H5 layout must explicitly identify an original sample/channel record')
        else: raise ValueError('storage must be raw or h5; no backend fallback')
        if array.ndim == 1: array = array[:, None]
        channel = int(row['Channel'])
        if array.ndim != 2 or channel >= array.shape[1] or len(array) < args.window_size or not np.isfinite(array).all():
            raise ValueError(f'Invalid original record shape/length/channel: {row["Id"]}')
        signals.append(torch.as_tensor(np.array(array[:, channel:channel+1], dtype=np.float32, copy=True)))
    return signals


class P08Windows(Dataset):
    def __init__(self, frame, signals, conditions, args):
        self.records = frame.reset_index(drop=True); self.signals = signals; self.conditions = conditions
        self.normalization = args.normalization
        if self.normalization not in {'none','per_window'}: raise ValueError('P08 normalization must be none or per_window')
        for key in ('window_size','stride'):
            value = getattr(args, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1: raise ValueError(f'Invalid {key}')
        self.windows = [(i, start, start+args.window_size) for i, x in enumerate(signals)
                        for start in range(0, len(x)-args.window_size+1, args.stride)]
        counts = Counter(i for i, _, _ in self.windows)
        self.expected = {str(row.record_id): counts[i] for i, row in self.records.iterrows()}
        if not all(self.expected.values()): raise ValueError('A selected record has no windows')

    def __len__(self): return len(self.windows)

    def __getitem__(self, index):
        i, start, end = self.windows[index]; row = self.records.iloc[i]
        x = self.signals[i][start:end].clone()
        if self.normalization == 'per_window':
            scale = x.std(unbiased=False)
            if scale <= 0: raise ValueError('Cannot normalize a constant window')
            x = (x-x.mean())/scale
        return {'x': x, 'condition': self.conditions[i], 'fs': float(row.Sample_rate),
                'y': int(row.Label), 'record_id': str(row.record_id), 'unit': str(row.Group),
                'system': str(row.Dataset_id), 'window_start': start}

    def sampling_weights(self):
        # Uniform system -> unit -> original record -> window, as in the manuscript.
        frame = self.records
        units = frame.groupby('Dataset_id').Group.nunique().to_dict()
        records = Counter(zip(frame.Dataset_id, frame.Group))
        return torch.tensor([1/(units[row.Dataset_id]*records[row.Dataset_id, row.Group]*self.expected[str(row.record_id)])
            for i, _, _ in self.windows for row in [frame.iloc[i]]], dtype=torch.double)


@register_data_factory('p08')
class P08DataFactory(ExplicitDataFactory):
    def __init__(self, args_data, args_task):
        if (args_task.type, args_task.name) != ('DG','p08'):
            raise ValueError('P08DataFactory requires task DG/p08')
        self.args_data, self.args_task = args_data, args_task
        if args_data.num_workers != 0 or args_data.evidence_kind not in {'software_fixture','natural_acquisition'}:
            raise ValueError('P08 requires num_workers=0 and explicit evidence_kind')
        frame = read_inventory(Path(args_data.data_dir)/args_data.metadata_file, args_data.condition_fields)
        frame = validate_inventory(frame, args_data)
        schema = fit_conditions(frame[frame.role == 'source_train'], args_data.condition_fields)
        conditions = encode_conditions(frame, schema)
        signals = read_records(frame, args_data)
        self.metadata = MetadataAccessor(frame, key_column='Id')
        self.metadata.p08_contract = {'condition_schema': schema, 'label_names': list(args_data.label_names),
            'source_groups': sorted(str(x) for x in set(frame.Group)),
            'source_records': sorted(str(x) for x in set(frame.record_id)),
            'source_system_ids': list(args_data.source_system_ids), 'target_system_id': args_data.target_system_id,
            'evidence_kind': args_data.evidence_kind, 'window_size': args_data.window_size, 'stride': args_data.stride, 'normalization': args_data.normalization}
        datasets = {}
        for role in ('source_train', 'source_val'):
            indices = list(np.flatnonzero(frame.role.to_numpy() == role))
            datasets[role] = P08Windows(frame.iloc[indices], [signals[i] for i in indices], conditions[indices], args_data)
        self.train_dataset, self.val_dataset = datasets['source_train'], datasets['source_val']
        for key in ('batch_size','train_batches_per_epoch'):
            v = getattr(args_data, key)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1: raise ValueError(f'Invalid {key}')
        generator = torch.Generator().manual_seed(args_data.sampling_seed)
        sampler = WeightedRandomSampler(self.train_dataset.sampling_weights(),
            args_data.batch_size*args_data.train_batches_per_epoch, replacement=True, generator=generator)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args_data.batch_size, sampler=sampler, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args_data.batch_size, shuffle=False, num_workers=0)
        self.test_loader = self.test_dataset = self.data = None

    def get_dataloader(self, mode='train'):
        if mode not in {'train','val'}: raise ValueError('P08 training cannot open a target loader; use frozen evaluation')
        return self.train_loader if mode == 'train' else self.val_loader
