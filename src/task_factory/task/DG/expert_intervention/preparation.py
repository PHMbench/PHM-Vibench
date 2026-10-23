"""Build explicit order-band views and matched spectral edits from a MAT-file manifest.

This is a signal-intervention builder, not validation that a probe preserves a
fault label or corresponds to a physical repair. No metadata are inferred.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .audit import validate_groups


def read_signal(row: dict[str, str], raw_root: Path) -> np.ndarray:
    """Read an explicit MAT variable or the maintained PU reader/channel contract."""
    path = raw_root/row['file']
    reader = row.get('reader', 'mat') or 'mat'
    if reader == 'mat':
        return np.asarray(loadmat(path)[row['signal_key']]).squeeze()
    if reader == 'h5':
        import h5py

        with h5py.File(path, 'r') as stream:
            values = np.asarray(stream[row['signal_key']])
        channel = int(row['channel'])
        if values.ndim != 3 or values.shape[-1] != 1 or not 0 <= channel < values.shape[1]:
            raise ValueError('Declared h5 input must have [samples, channels, 1] axes and a valid channel')
        return values[:, channel, 0]
    if reader != 'RM_027_PU':
        raise ValueError(f'Unsupported declared reader: {reader}')
    # Reuse the existing three-channel PU reader; never duplicate nested MAT parsing.
    from src.data_factory.reader.RM_027_PU import read
    channel = int(row['channel'])
    values = np.asarray(read(str(path)))
    if values.ndim != 2 or not 0 <= channel < values.shape[1]:
        raise ValueError('PU channel must index the explicit [samples, channels] reader output')
    return values[:, channel]


def components(x: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(x)[None, :] * masks, n=len(x), axis=-1)


def paired_edits(x: np.ndarray, target: np.ndarray, control: np.ndarray,
                 strength: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < strength < 1:
        raise ValueError('strength must be in (0,1)')
    parts=components(x,np.stack([target,control]))
    norms=np.linalg.norm(parts,axis=1)
    if (norms <= 0).any():
        raise ValueError('A requested probe/control band has zero energy')
    # Same perturbation L2 norm; neither component is attenuated by more than strength.
    magnitude=strength*norms.min()
    return x-magnitude*parts[0]/norms[0], x-magnitude*parts[1]/norms[1]


def prepare(manifest: Path, raw_root: Path, roles_path: Path, output: Path,
            window: int, strength: float, *, partitions: tuple[str, ...] | None = None,
            class_names: list[str] | None = None) -> None:
    if window < 8 or not 0 < strength < 1:
        raise ValueError('Require window >= 8 and 0 < strength < 1')
    for destination in (output, output.with_suffix('.json')):
        if destination.exists():
            raise FileExistsError(destination)
    roles=json.loads(roles_path.read_text())
    if not isinstance(roles,list) or len(roles)<2:
        raise ValueError('roles JSON must be a list with at least two named roles')
    for role in roles:
        if not role.get('name'):
            raise ValueError('Each role needs a name')
        for key in ('target_orders','control_orders'):
            low,high=role[key]
            if not np.isfinite([low,high]).all() or not 0 <= low < high:
                raise ValueError(f'{role["name"]}: invalid {key}')
    if len({role['name'] for role in roles}) != len(roles):
        raise ValueError('Role names must be distinct')
    values={k:[] for k in ('views','raw','compatibility','labels','group','split','domain',
                            'probe_views','control_views','probe_raw','control_raw')}
    counts=[]
    with manifest.open() as f:
        records=list(csv.DictReader(f))
    if not records:
        raise ValueError('Manifest is empty')
    has_specimen = ['specimen' in row and bool(row['specimen']) for row in records]
    if any(has_specimen) and not all(has_specimen):
        raise ValueError('specimen must be supplied for every manifest record or omitted throughout')
    if all(has_specimen):
        values['specimen'] = []
    group, split, domain = (np.asarray([row[key] for row in records]) for key in ('group','split','domain'))
    labels = np.asarray([int(row['label']) for row in records])
    validate_groups(group, split, domain, labels)
    if all(has_specimen):
        specimens = np.asarray([row['specimen'] for row in records])
        for specimen in np.unique(specimens):
            ix = specimens == specimen
            if len(np.unique(split[ix])) != 1 or len(np.unique(labels[ix])) != 1:
                raise ValueError(f'specimen {specimen}: physical identity crosses partitions or labels')
            for dom in np.unique(domain[ix]):
                if len(np.unique(group[ix & (domain == dom)])) != 1:
                    raise ValueError('A specimen/domain cell must have exactly one aggregation group')
        for aggregation in np.unique(group):
            if len(np.unique(specimens[group == aggregation])) != 1:
                raise ValueError('An aggregation group cannot mix specimens')
    acquisitions = set()
    for row in records:
        identity = ((raw_root/row['file']).resolve(), row.get('reader') or 'mat', row.get('signal_key', ''))
        if identity in acquisitions:
            raise ValueError(f'Duplicate declared acquisition: {identity}; do not alias one recording across units or channels')
        acquisitions.add(identity)
    if class_names is not None:
        if len(class_names) < 2 or len(set(class_names)) != len(class_names) or not all(class_names):
            raise ValueError('class_names must provide distinct names in label-index order')
        if labels.min() < 0 or labels.max() >= len(class_names):
            raise ValueError('Manifest labels exceed the declared class_names')
    if partitions is not None:
        if len(set(partitions)) != len(partitions) or not set(partitions) <= {'train','val','match','test'}:
            raise ValueError('partitions must name distinct train/val/match/test partitions')
        records = [row for row in records if row['split'] in partitions]
        if set(row['split'] for row in records) != set(partitions):
            raise ValueError('The manifest must contain all requested partitions')
    for row in records:
        fs,rpm=float(row['fs_hz']),float(row['rpm'])
        if not np.isfinite([fs,rpm]).all() or fs<=0 or rpm<=0:
            raise ValueError(f'{row["file"]}: positive measured fs_hz and rpm required')
        signal=read_signal(row, raw_root)
        if signal.ndim!=1 or not np.isfinite(signal).all() or len(signal)<window:
            raise ValueError(f'{row["file"]}: require a finite 1D signal of at least one window')
        orders=np.fft.rfftfreq(window,d=1/fs)/(rpm/60)
        masks=[]; controls=[]
        for role in roles:
            low,high=role['target_orders']; clow,chigh=role['control_orders']
            if max(high,chigh)>orders[-1]:
                raise ValueError(f'{role["name"]}: order band exceeds Nyquist')
            target=(orders>=low)&(orders<high); control=(orders>=clow)&(orders<chigh)
            if not target.any() or not control.any() or np.any(target&control):
                raise ValueError(f'{role["name"]}: bands need disjoint nonempty FFT bins')
            masks.append(target); controls.append(control)
        masks=np.stack(masks)
        for start in range(0,len(signal)-window+1,window):
            x=signal[start:start+window].astype(float)
            x=x-x.mean()
            energy=float(x@x)
            if energy<=0:
                raise ValueError(f'{row["file"]}, window {start}: zero signal energy')
            views=components(x,masks)
            pairs=[paired_edits(x,t,c,strength) for t,c in zip(masks,controls)]
            probe,control=np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs])
            values['raw'].append(x); values['views'].append(views)
            values['compatibility'].append(2*np.square(views).sum(1)/energy-1)
            values['probe_raw'].append(probe); values['control_raw'].append(control)
            values['probe_views'].append(np.stack([components(p,masks) for p in probe]))
            values['control_views'].append(np.stack([components(c,masks) for c in control]))
            values['labels'].append(int(row['label']))
            for key in ('group','split','domain'):
                values[key].append(row[key])
            if all(has_specimen):
                values['specimen'].append(row['specimen'])
        counts.append(dict(row,windows=len(signal)//window,
                           discarded_tail_samples=len(signal)%window))
    arrays={k:np.asarray(v,dtype='float32') if k not in ('labels','group','split','domain','specimen')
            else np.asarray(v) for k,v in values.items()}
    arrays['role_names'] = np.asarray([role['name'] for role in roles])
    if class_names is not None:
        arrays['class_names'] = np.asarray(class_names)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f,**arrays)
    output.with_suffix('.json').write_text(json.dumps(dict(roles=roles,window=window,
        strength=strength,records=counts,independence='requires acquisition-design justification',
        probe_validity='matched signal edits; physical/label validity not certified'),indent=2))
