"""Synthetic checks for feature alignment and producer/split boundaries."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.task_factory.task.classification.selective_diagnosis.export import export_archive
from src.task_factory.task.classification.selective_diagnosis.run import load_data


FIELDS = ('feature_row', 'y', 'unit', 'split', 'domain')


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    x = np.arange(16, dtype=np.float32).reshape(8, 2) / 3
    rows = [dict(feature_row=i, y=i % 2, unit='NA' if i == 0 else f'unit_{i}',
                 split=role, domain='NA' if i < 2 else 'condition_b')
            for i, role in enumerate(np.repeat(['train', 'tune', 'cal', 'test'], 2))]
    producer = dict(feature_kind='fixed', fit_role='none',
                    feature_definition='Two predetermined synthetic coordinates',
                    producer_version='synthetic-test-v1', standardized=False)
    paths = {key: tmp_path / name for key, name in (
        ('features', 'features.npy'), ('manifest', 'manifest.csv'),
        ('feature_names', 'feature_names.json'), ('producer', 'producer.json'),
        ('output', 'archive'))}
    np.save(paths['features'], x)
    write_manifest(paths['manifest'], rows)
    paths['feature_names'].write_text(json.dumps(['coordinate_0', 'coordinate_1']))
    paths['producer'].write_text(json.dumps(producer))
    return paths, x, rows, producer


def test_permuted_manifest_aligns_labels_and_ids_without_reordering_features(inputs):
    paths, x, rows, _ = inputs
    write_manifest(paths['manifest'], [rows[i] for i in [7, 1, 4, 0, 6, 3, 5, 2]])
    output = export_archive(**paths, kind='synthetic')
    assert output == paths['output'] / 'features.npz'
    data = load_data(output)
    np.testing.assert_array_equal(data['x'], x)
    for key in ('y', 'unit', 'split', 'domain'):
        np.testing.assert_array_equal(data[key], [row[key] for row in rows])
    np.testing.assert_array_equal(data['feature_names'], ['coordinate_0', 'coordinate_1'])
    assert data['kind'] == 'synthetic'
    assert data['unit'][0] == data['domain'][0] == 'NA'
    np.testing.assert_array_equal(np.load(paths['features']), x)
    assert json.loads((paths['output'] / 'run_state.json').read_text())['status'] == 'completed'
    assert json.loads((paths['output'] / 'export_config.json').read_text())['producer_declaration']['standardized'] is False


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_nonfinite_features_are_rejected(inputs, bad_value):
    paths, x, _, _ = inputs
    x[2, 1] = bad_value
    np.save(paths['features'], x)
    with pytest.raises(ValueError):
        export_archive(**paths, kind='synthetic')


@pytest.mark.parametrize('issue', ['duplicate', 'missing', 'noninteger'])
def test_feature_rows_must_be_unique_complete_integers(inputs, issue):
    paths, _, rows, _ = inputs
    if issue == 'duplicate':
        rows[-1]['feature_row'] = 0
    elif issue == 'missing':
        rows.pop()
    else:
        rows[1]['feature_row'] = '1.5'
    write_manifest(paths['manifest'], rows)
    with pytest.raises(ValueError):
        export_archive(**paths, kind='synthetic')


def test_fractional_label_cannot_be_silently_truncated(inputs):
    paths, _, rows, _ = inputs
    rows[1]['y'] = '1.5'
    write_manifest(paths['manifest'], rows)
    with pytest.raises(ValueError):
        export_archive(**paths, kind='synthetic')


def test_unit_cannot_cross_train_and_calibration_roles(inputs):
    paths, _, rows, _ = inputs
    rows[4]['unit'] = rows[0]['unit']
    write_manifest(paths['manifest'], rows)
    with pytest.raises(ValueError):
        export_archive(**paths, kind='synthetic')
    state = json.loads((paths['output'] / 'run_state.json').read_text())
    assert state['status'] == 'failed'
    assert state['error_type'] == 'ValueError'


def test_learned_features_accept_exact_training_unit_set(inputs):
    paths, _, _, producer = inputs
    producer.update(feature_kind='learned', fit_role='train',
                    train_unit_ids=['unit_1', 'NA'], checkpoint='synthetic-checkpoint.pt')
    paths['producer'].write_text(json.dumps(producer))
    output = export_archive(**paths, kind='synthetic')
    assert set(load_data(output)['split']) == {'train', 'tune', 'cal', 'test'}


@pytest.mark.parametrize('fit_units', [['NA'], ['NA', 'unit_1', 'unit_4']])
def test_learned_producer_requires_exact_training_unit_set(inputs, fit_units):
    paths, _, _, producer = inputs
    producer.update(feature_kind='learned', fit_role='train',
                    train_unit_ids=fit_units, checkpoint='synthetic-checkpoint.pt')
    paths['producer'].write_text(json.dumps(producer))
    with pytest.raises(ValueError):
        export_archive(**paths, kind='synthetic')


def test_existing_output_is_not_overwritten(inputs):
    paths, _, _, _ = inputs
    paths['output'].write_bytes(b'existing experiment artifact')
    with pytest.raises(FileExistsError):
        export_archive(**paths, kind='synthetic')
    assert paths['output'].read_bytes() == b'existing experiment artifact'


def test_duplicate_feature_names_are_rejected(inputs):
    paths, _, _, _ = inputs
    paths['feature_names'].write_text(json.dumps(['coordinate', 'coordinate']))
    with pytest.raises(ValueError):
        export_archive(**paths, kind='synthetic')


def test_already_standardized_features_are_rejected(inputs):
    paths, _, _, producer = inputs
    producer['standardized'] = True
    paths['producer'].write_text(json.dumps(producer))
    with pytest.raises(ValueError):
        export_archive(**paths, kind='synthetic')
