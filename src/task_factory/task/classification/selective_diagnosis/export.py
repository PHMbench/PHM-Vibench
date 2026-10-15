"""Package existing unstandardized features and explicit metadata for P4.

This command neither extracts features nor infers specimen identities. Producer
metadata is a declaration; it cannot establish physical independence or prove
which observations fitted an upstream feature extractor. Standardization belongs
to the P4 runner and is fitted there on training units only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from . import run


def _integer(value: str, field: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r'[+-]?[0-9]+', value) is None:
        raise ValueError(f'{field} must be an integer, got {value!r}')
    return int(value)


def export_archive(
    features: Path,
    manifest: Path,
    feature_names: Path,
    producer: Path,
    output: Path,
    kind: str = 'real',
) -> Path:
    """Export without changing source values; reject ambiguous row alignment."""
    if kind not in {'real', 'synthetic'}:
        raise ValueError('kind must be real or synthetic')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'output is not empty: {output}; use a new run directory')
    if features.suffix.lower() != '.npy':
        raise ValueError('--features must be an existing .npy matrix')
    x = np.load(features, allow_pickle=False)
    if (x.ndim != 2 or min(x.shape) < 1 or x.dtype.kind not in {'i', 'u', 'f'}
            or not np.isfinite(x).all()):
        raise ValueError('features must be a finite real N x D matrix')
    names = json.loads(feature_names.read_text(encoding='utf-8'))
    if (not isinstance(names, list) or len(names) != x.shape[1]
            or any(not isinstance(v, str) or not v.strip() for v in names)
            or len(set(names)) != len(names)):
        raise ValueError('feature names must be a JSON array of D unique nonempty strings')
    with manifest.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        required = {'feature_row', 'y', 'unit', 'split', 'domain'}
        if (reader.fieldnames is None or set(reader.fieldnames) != required
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError('manifest requires unique columns: feature_row,y,unit,split,domain')
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError('manifest row has fields beyond the declared columns')
    if len(rows) != len(x):
        raise ValueError('manifest must contain one row per feature row')
    indices = [_integer(row['feature_row'], 'feature_row') for row in rows]
    if len(set(indices)) != len(x) or set(indices) != set(range(len(x))):
        raise ValueError('feature_row must cover 0..N-1 exactly once')
    rows = [row for _, row in sorted(zip(indices, rows), key=lambda item: item[0])]
    y = np.array([_integer(row['y'], 'y') for row in rows], dtype=np.int64)
    metadata: dict[str, np.ndarray] = {}
    for field in ('unit', 'split', 'domain'):
        values = [row[field] for row in rows]
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f'{field} identifiers must be nonempty strings')
        metadata[field] = np.array(values, dtype=str)

    declaration = json.loads(producer.read_text(encoding='utf-8'))
    if not isinstance(declaration, dict):
        raise ValueError('producer must be a JSON object')
    for field in ('feature_definition', 'producer_version'):
        if not isinstance(declaration.get(field), str) or not declaration[field].strip():
            raise ValueError(f'producer {field} must be a nonempty string')
    if declaration.get('standardized') is not False:
        raise ValueError('producer must declare standardized=false; export unstandardized features')
    feature_kind = declaration.get('feature_kind')
    if feature_kind == 'fixed':
        if declaration.get('fit_role') != 'none':
            raise ValueError('fixed features require fit_role=none')
    elif feature_kind == 'learned':
        if declaration.get('fit_role') != 'train':
            raise ValueError('learned features require fit_role=train')
        declared_units = declaration.get('train_unit_ids')
        actual_units = set(metadata['unit'][metadata['split'] == 'train'])
        if (not isinstance(declared_units, list)
                or any(not isinstance(v, str) or not v.strip() for v in declared_units)
                or len(set(declared_units)) != len(declared_units)
                or set(declared_units) != actual_units):
            raise ValueError('producer train_unit_ids must equal the manifest training-unit set')
        if not isinstance(declaration.get('checkpoint'), str) or not declaration['checkpoint'].strip():
            raise ValueError('learned features require a nonempty checkpoint identifier')
    else:
        raise ValueError('producer feature_kind must be fixed or learned')

    # Reuse the runner's create-only output, code provenance and terminal state.
    # The final archive goes through exactly the same contract as training.
    try:
        run.reserve_output(output)
        sources = {name: str(path.resolve()) for name, path in {
            'features': features, 'manifest': manifest,
            'feature_names': feature_names, 'producer': producer,
        }.items()}
        config = dict(sources=sources, producer_declaration=declaration, kind=kind,
                      rows=len(x), columns=x.shape[1], feature_names=names,
                      row_alignment='manifest feature_row indexes unchanged input matrix',
                      declaration_boundary='Producer fit scope and physical-unit independence '
                      'are declared, not established by this export.')
        (output / 'export_config.json').write_text(
            json.dumps(config, indent=2, allow_nan=False), encoding='utf-8')
        archive = output / 'features.npz'
        np.savez_compressed(archive, x=x, y=y, **metadata,
                            feature_names=np.array(names), kind=np.array(kind))
        run.load_data(archive)
        run.write_state('completed', archive=str(archive.resolve()),
                        training_executed=False, data_kind=kind)
        return archive
    except Exception as exc:
        run.write_state('failed', error_type=type(exc).__name__, error=str(exc),
                        training_executed=False)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True, help='Unstandardized N x D .npy')
    parser.add_argument('--manifest', type=Path, required=True,
                        help='CSV: feature_row,y,unit,split,domain; genuine specimen IDs')
    parser.add_argument('--feature-names', type=Path, required=True,
                        help='JSON file containing D unique feature-name strings')
    parser.add_argument('--producer', type=Path, required=True,
                        help='JSON file declaring feature definition, fit scope and version')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--kind', choices=('real', 'synthetic'), default='real')
    args = parser.parse_args(argv)
    print(export_archive(**vars(args)))


if __name__ == '__main__':
    main()
