"""A domain number must not pass as physical operating-condition evidence."""
from __future__ import annotations

import copy

import h5py
import numpy as np
import pandas as pd
import pytest

from experiments.p01.condition_contract import (
    ConditionContractError, PROTOCOL, is_condition_dg, validate_condition_task,
)
from experiments.p01.fusion_data import read_records


def task_contract():
    evidence = dict(source='test/test_p01_condition_contract.py', location='task_contract fixture',
                    statement='Explicitly constructed RPM settings and independent specimen identities; software evidence only.')
    return dict(protocol=PROTOCOL, model=dict(num_classes=2),
        specimen_basis=dict(unit='physical_specimen', definition='One independently constructed bearing', evidence=evidence),
        physical_conditions=dict(
            variables=dict(rpm=dict(unit='rpm', kind='setpoint', quantity='shaft rotational speed')),
            mapping=dict(kind='documented_metadata', source_column='Domain', evidence=evidence),
            conditions={cid:dict(values=dict(rpm=rpm), evidence=evidence)
                        for cid, rpm in [('c0', 1500), ('c1', 1200), ('c2', 900)]}),
        dataset=dict(source_domains=['c0', 'c1'], domain_sequence=['c2'], rotation_speed_kind='setpoint',
            columns=dict(id='Id', unit_id='Specimen', domain='Domain', split='Split', label='Label',
                         sample_rate_hz='Fs', rotation_speed_rpm='RPM')))


def assignment_records():
    rows = []
    for cid in ('c0', 'c1', 'c2'):
        for split in (('update', 'validation', 'test') if cid != 'c2' else ('test',)):
            for label in (0, 1):
                rows.append(dict(Id=str(len(rows)+1), Specimen=f'{split}-{label}', Domain=cid,
                                 Split=split, Label=str(label), Fs='12000',
                                 RPM=str({'c0':1500, 'c1':1200, 'c2':900}[cid])))
    return rows


def test_condition_table_counts_repeat_specimens_without_relabeling_them():
    rows = validate_condition_task(task_contract(), assignment_records())
    assert [r['role'] for r in rows] == ['source', 'source', 'target']
    assert [r['specimens'] for r in rows] == [6, 6, 2]
    assert rows[2]['class_support'] == 'test:0,1'
    assert rows[0]['rpm_kind'] == 'setpoint'


@pytest.mark.parametrize('defect', ['duplicate_tuple', 'unknown', 'identifier_variable',
                                   'undocumented_mapping', 'missing_evidence', 'unknown_key'])
def test_identifier_or_unsupported_physics_cannot_admit_a_task(defect):
    task = task_contract()
    physical = task['physical_conditions']
    if defect == 'duplicate_tuple':
        physical['conditions']['c2']['values'] = dict(rpm=1200)
    elif defect == 'unknown':
        physical['conditions']['c2']['values']['rpm'] = 'unknown'
    elif defect == 'identifier_variable':
        physical['variables']['domain_id'] = physical['variables'].pop('rpm')
    elif defect == 'undocumented_mapping':
        physical['mapping']['kind'] = 'filename_guess'
    elif defect == 'missing_evidence':
        physical['conditions']['c2']['evidence'] = dict(source='paper', location='', statement='unknown')
    else:
        physical['conditions']['c2']['nominal_rpm'] = 900
    with pytest.raises(ConditionContractError, match='CONDITION_UNVERIFIED'):
        validate_condition_task(task)


def test_authoritative_filename_mapping_is_explicitly_distinct_from_guessing():
    task = task_contract()
    task['physical_conditions']['mapping'].update(kind='documented_filename', source_column='File')
    task['physical_conditions']['mapping']['evidence'] = dict(
        source='fixture documentation', location='naming convention table',
        statement='The fixture table explicitly maps C0/C1/C2 filenames to 1500/1200/900 rpm.')
    assert len(validate_condition_task(task)) == 3


def test_independent_run_does_not_establish_unseen_specimen():
    task = task_contract()
    task['specimen_basis']['unit'] = 'independent_run'
    with pytest.raises(ConditionContractError, match='SPECIMEN_UNVERIFIED'):
        validate_condition_task(task)


def test_specimen_and_condition_aliases_must_agree():
    records = assignment_records()
    records[0]['condition_id'] = 'c2'
    with pytest.raises(ConditionContractError, match='aliases disagree'):
        validate_condition_task(task_contract(), records)
    records[0].pop('condition_id')
    records[0]['specimen_id'] = 'a-different-bearing'
    with pytest.raises(ConditionContractError, match='SPECIMEN_UNVERIFIED'):
        validate_condition_task(task_contract(), records)


def test_same_specimen_at_other_conditions_stays_in_one_partition():
    records = assignment_records()
    records[6]['Split'] = 'validation'
    with pytest.raises(ConditionContractError, match='SPLIT_INVALID.*crosses partitions'):
        validate_condition_task(task_contract(), records)


def test_per_condition_per_partition_class_support_is_required():
    records = assignment_records()
    records = [r for r in records if not (r['Domain'] == 'c1' and r['Split'] == 'validation' and r['Label'] == '1')]
    with pytest.raises(ConditionContractError, match='c1/validation lacks full class support'):
        validate_condition_task(task_contract(), records)


def test_target_condition_cannot_enter_development_even_on_training_specimens():
    records = assignment_records()
    records.append(dict(Id='99', Specimen='update-0', Domain='c2', Split='update', Label='0'))
    with pytest.raises(ConditionContractError, match='Unseen target condition cannot participate'):
        validate_condition_task(task_contract(), records)


def test_metadata_audit_binds_tuple_rpm_before_any_payload_can_be_read():
    records = assignment_records()
    records[0]['RPM'] = '900'
    with pytest.raises(ConditionContractError, match='record RPM.*differs from documented setpoint'):
        validate_condition_task(task_contract(), records)


def source_binding(tmp_path):
    task = task_contract()
    dataset = copy.deepcopy(task['dataset'])
    dataset.update(format='vibench_h5', metadata_file=str(tmp_path/'source.csv'),
                   h5_file=str(tmp_path/'source.h5'), access_scope='source', domain_sequence=[])
    for key in ('protocol', 'specimen_basis', 'physical_conditions'):
        dataset[key] = task[key]
    rows = [r for r in assignment_records() if r['Split'] in {'update', 'validation'}]
    pd.DataFrame(rows).to_csv(dataset['metadata_file'], index=False)
    with h5py.File(dataset['h5_file'], 'w') as h5:
        for row in rows:
            h5[row['Id']] = np.ones((8, 1), dtype=np.float32)
    return dataset, dict(model=task['model'], datasets=[dataset])


def test_source_reader_preserves_physical_identity_and_never_loads_payload(tmp_path, monkeypatch):
    dataset, config = source_binding(tmp_path)
    from src.data_factory.H5DataDict import H5DataDict
    def forbidden_payload(*args, **kwargs):
        raise AssertionError('A metadata contract must not inspect waveforms.')
    monkeypatch.setattr(H5DataDict, '__getitem__', forbidden_payload)
    records = read_records(dataset, config)
    assert len(records) == 8
    assert all(r['condition_id'] == r['domain'] and r['specimen_id'] == r['unit_id'] for r in records)
    assert all(r['rotation_speed_kind'] == 'setpoint' for r in records)


def test_stripped_source_contract_fails_before_any_file_is_opened(monkeypatch):
    dataset = dict(access_scope='source')
    assert is_condition_dg(dataset)
    def forbidden_read(*args, **kwargs):
        raise AssertionError('An unbound task must fail before metadata is read.')
    monkeypatch.setattr(pd, 'read_csv', forbidden_read)
    with pytest.raises(ConditionContractError, match='CONDITION_UNVERIFIED'):
        read_records(dataset, dict(model=dict(num_classes=2)))


def test_mapped_rpm_cannot_disagree_with_documented_same_kind(tmp_path):
    dataset, config = source_binding(tmp_path)
    frame = pd.read_csv(dataset['metadata_file'])
    frame.loc[0, 'RPM'] = 900
    frame.to_csv(dataset['metadata_file'], index=False)
    with pytest.raises(ConditionContractError, match='record RPM.*differs from documented setpoint'):
        read_records(dataset, config)


def test_measured_rpm_is_not_silently_declared_a_setpoint(tmp_path):
    dataset, config = source_binding(tmp_path)
    dataset['rotation_speed_kind'] = 'measured'
    frame = pd.read_csv(dataset['metadata_file'])
    frame.loc[0, 'RPM'] = 1497
    frame.to_csv(dataset['metadata_file'], index=False)
    records = read_records(dataset, config)
    assert records[0]['rotation_speed_rpm'] == 1497
    assert records[0]['rotation_speed_kind'] == 'measured'
    assert dataset['physical_conditions']['conditions']['c0']['values']['rpm'] == 1500


def test_legacy_reproduction_does_not_acquire_a_condition_dg_claim(tmp_path):
    dataset, config = source_binding(tmp_path)
    for key in ('protocol', 'specimen_basis', 'physical_conditions'):
        dataset.pop(key)
    dataset['access_scope'] = 'legacy'
    assert not is_condition_dg(dataset, config)
    # Legacy keeps its historical requirement for permanent tests in every domain;
    # it is not silently made a source-only formal run by removing the contract.
    with pytest.raises(ValueError, match='permanent test'):
        read_records(dataset, config)


def heldout_binding(tmp_path, rows):
    task = task_contract()
    task['dataset']['access_scope'] = 'test'
    dataset = copy.deepcopy(task['dataset'])
    dataset.update(format='vibench_h5', metadata_file=str(tmp_path/'heldout.csv'),
                   h5_file=str(tmp_path/'heldout.h5'))
    for key in ('protocol', 'specimen_basis', 'physical_conditions'):
        dataset[key] = task[key]
    pd.DataFrame(rows).to_csv(dataset['metadata_file'], index=False)
    with h5py.File(dataset['h5_file'], 'w') as h5:
        for row in rows:
            h5[row['Id']] = np.ones((8, 1), dtype=np.float32)
    return task, dataset, dict(model=task['model'], datasets=[dataset])


def test_missing_source_controls_do_not_block_valid_primary_target(tmp_path):
    rows = [r for r in assignment_records() if r['Domain'] == 'c2']
    task, dataset, config = heldout_binding(tmp_path, rows)
    audit = {r['condition_id']:r for r in validate_condition_task(task, rows)}
    assert audit['c0']['heldout_status'] == audit['c1']['heldout_status'] == 'ABSENT'
    assert audit['c2']['heldout_status'] == 'COMPLETE'
    assert {r['domain'] for r in read_records(dataset, config)} == {'c2'}


def test_partial_source_control_reports_actual_coverage_without_dropping_target(tmp_path):
    rows = [r for r in assignment_records() if r['Split'] == 'test'
            and (r['Domain'] == 'c2' or (r['Domain'] == 'c0' and r['Label'] == '0'))]
    task, dataset, config = heldout_binding(tmp_path, rows)
    audit = {r['condition_id']:r for r in validate_condition_task(task, rows)}
    assert audit['c0']['heldout_status'] == 'PARTIAL'
    assert audit['c0']['heldout_class_support'] == '0'
    assert audit['c0']['paired_target_specimens'] == 1
    assert audit['c1']['heldout_status'] == 'ABSENT'
    assert len(read_records(dataset, config)) == 3


@pytest.mark.parametrize('defect', ['missing_target', 'missing_target_class', 'undeclared_condition', 'development_row'])
def test_optional_source_controls_do_not_weaken_primary_test_requirements(tmp_path, defect):
    rows = [r for r in assignment_records() if r['Split'] == 'test']
    if defect == 'missing_target':
        rows = [r for r in rows if r['Domain'] != 'c2']
    elif defect == 'missing_target_class':
        rows = [r for r in rows if not (r['Domain'] == 'c2' and r['Label'] == '1')]
    elif defect == 'undeclared_condition':
        rows[0]['Domain'] = 'c9'
    else:
        rows[0]['Split'] = 'update'
    _, dataset, config = heldout_binding(tmp_path, rows)
    with pytest.raises(ConditionContractError, match='SPLIT_INVALID|CONDITION_UNVERIFIED'):
        read_records(dataset, config)


def test_metadata_only_control_inventory_does_not_claim_class_qualification():
    task = task_contract()
    rows = assignment_records()
    for record in rows:
        record.pop('Label')
    audit = validate_condition_task(task, rows)
    assert all(r['heldout_status'] == 'PRESENT_CLASS_SUPPORT_UNVERIFIED' for r in audit)
    assert all(r['heldout_class_support'] == '' for r in audit)


def test_source_only_support_omits_unobserved_test_partitions():
    task = task_contract()
    task['dataset'].update(access_scope='source', domain_sequence=[])
    records = [r for r in assignment_records() if r['Split'] in {'update', 'validation'}]
    audit = {row['condition_id']:row for row in validate_condition_task(task, records)}
    assert audit['c0']['class_support'] == 'update:0,1;validation:0,1'
    assert audit['c1']['class_support'] == 'update:0,1;validation:0,1'
    assert audit['c2']['class_support'] == ''
