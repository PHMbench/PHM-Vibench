"""Physical admission for the P01 specimen-disjoint condition study.

This validates a documented scientific declaration, not the truth of a citation.
The custodian must read the cited passages before supplying a bound task. Nothing
in this module opens an acquisition, estimates a condition, or fits a model.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


PROTOCOL = 'specimen_disjoint_condition_dg'
_UNBOUND = {'', 'unknown', 'unverified', 'unbound', 'todo', 'none', 'null', 'nan'}


class ConditionContractError(ValueError):
    """Invalid physics, identity, or split; never an instruction to repair data."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f'{code}: {message}')


def _fail(code: str, message: str) -> None:
    raise ConditionContractError(code, message)


def _text(value: Any, where: str, code: str = 'CONDITION_UNVERIFIED') -> str:
    if (not isinstance(value, str) or value.strip().lower() in _UNBOUND
            or 'replace' in value.lower()):
        _fail(code, f'{where} requires a documented non-placeholder value.')
    return value


def _fields(value: Any, fields: set[str], where: str,
            code: str = 'CONDITION_UNVERIFIED') -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(code, f'{where} requires exactly {sorted(fields)}; unknown or missing fields are invalid.')
    return value


def _evidence(value: Any, where: str, code: str = 'CONDITION_UNVERIFIED') -> str:
    value = _fields(value, {'source', 'location', 'statement'}, where, code)
    return ' | '.join(_text(value[key], f'{where}.{key}', code)
                      for key in ('source', 'location', 'statement'))


def is_condition_dg(dataset: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> bool:
    """An isolated DG input cannot become legacy by dropping just its contract."""
    config = config or {}
    return (dataset.get('access_scope') in {'source', 'test'}
            or any(key in owner for owner in (dataset, config)
                   for key in ('protocol', 'physical_conditions', 'specimen_basis')))


def bound_condition_task(dataset: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the same physical declaration for direct trainer invocations."""
    result: dict[str, Any] = {'dataset': dataset, 'model': config['model']}
    for key in ('protocol', 'physical_conditions', 'specimen_basis'):
        if key in dataset and key in config and dataset[key] != config[key]:
            _fail('CONDITION_UNVERIFIED', f'Conflicting {key} at dataset and configuration levels.')
        result[key] = dataset.get(key, config.get(key))
    return result


def _record_id(record: Mapping[str, Any], mapping: Mapping[str, str], canonical: str,
               legacy: str, code: str) -> str:
    values = [record[key] for key in {canonical, legacy, mapping.get(canonical), mapping.get(legacy)}
              if key is not None and key in record]
    if not values:
        _fail(code, f'Acquisition lacks {canonical}; an ID cannot be inferred from row order.')
    values = [_text(str(value), canonical, code) for value in values]
    if len(set(values)) != 1:
        _fail(code, f'{canonical}/{legacy} aliases disagree: {values}.')
    return values[0]


def _record_rpm(dataset: Mapping[str, Any], physical: Mapping[str, Any],
                record: Mapping[str, Any], cid: str) -> None:
    """Bind declared settings to administrative measurements before H5 access."""
    variable = physical['variables'].get('rpm')
    if variable is None or variable['kind'] == 'trajectory':
        return
    value = record.get('rotation_speed_rpm', record.get(dataset['columns'].get('rotation_speed_rpm')))
    if value is None:
        _fail('CONDITION_UNVERIFIED', f'{cid}: missing mapped RPM; declared physical tuples must bind to records.')
    pattern = dataset.get('rotation_speed_pattern')
    if pattern is not None:
        try:
            match = re.fullmatch(pattern, str(value))
        except re.error:
            match = None
        if match is None or 'rpm' not in match.groupdict():
            _fail('CONDITION_UNVERIFIED', f'{cid}: RPM metadata does not match the documented named-rpm pattern.')
        value = match.group('rpm')
    try:
        rpm = float(value)
    except (ValueError, TypeError, OverflowError):
        _fail('CONDITION_UNVERIFIED', f'{cid}: mapped RPM must be finite and positive.')
    if isinstance(value, bool) or not math.isfinite(rpm) or rpm <= 0:
        _fail('CONDITION_UNVERIFIED', f'{cid}: mapped RPM must be finite and positive.')
    if variable['kind'] == dataset['rotation_speed_kind']:
        declared = physical['conditions'][cid]['values']['rpm']
        if rpm != declared:
            _fail('CONDITION_UNVERIFIED',
                  f'{cid}: record RPM {rpm} differs from documented {variable["kind"]} {declared}.')


def validate_condition_task(task: Mapping[str, Any],
                            assignments: Sequence[Mapping[str, Any]] | None = None
                            ) -> list[dict[str, Any]]:
    """Validate a full custodian task or isolated source/test binding.

    Assignments can use canonical specimen_id/condition_id or the existing mapped
    unit_id/domain columns. Labels are optional for custodian structural records;
    Source records must cover every class in every source partition. Final target
    records must cover every class; source-condition held-out controls are optional
    and their missing class/specimen coverage is reported rather than fabricated.
    Returned rows are directly writable as a condition-audit CSV.
    """
    if task.get('protocol') != PROTOCOL:
        _fail('CONDITION_UNVERIFIED', f'Formal DG requires protocol={PROTOCOL!r}.')
    specimen = _fields(task.get('specimen_basis'), {'unit', 'definition', 'evidence'},
                       'specimen_basis', 'SPECIMEN_UNVERIFIED')
    if specimen['unit'] != 'physical_specimen':
        _fail('SPECIMEN_UNVERIFIED', 'Independent runs/files are not physical specimens.')
    _text(specimen['definition'], 'specimen_basis.definition', 'SPECIMEN_UNVERIFIED')
    specimen_evidence = _evidence(specimen['evidence'], 'specimen_basis.evidence', 'SPECIMEN_UNVERIFIED')
    physical = _fields(task.get('physical_conditions'), {'variables', 'mapping', 'conditions'},
                       'physical_conditions')
    variables = physical['variables']
    if not isinstance(variables, Mapping) or not variables:
        _fail('CONDITION_UNVERIFIED', 'Declare the physical variables defining the operating settings.')
    for name, spec in variables.items():
        _text(name, 'physical variable name')
        if name.lower() in {'domain', 'domain_id', 'condition_id', 'dataset', 'dataset_id',
                            'file', 'file_id', 'acquisition', 'acquisition_id', 'specimen', 'specimen_id', 'cluster'}:
            _fail('CONDITION_UNVERIFIED', f'{name} is an identifier, not a physical operating variable.')
        _fields(spec, {'unit', 'kind', 'quantity'}, f'variables.{name}')
        _text(spec['unit'], f'variables.{name}.unit')
        _text(spec['quantity'], f'variables.{name}.quantity')
        if name == 'rpm' and spec['unit'] != 'rpm':
            _fail('CONDITION_UNVERIFIED', 'The rpm field has unit rpm; use a separately documented quantity for another unit.')
        if spec['kind'] not in {'setpoint', 'measured', 'trajectory'}:
            _fail('CONDITION_UNVERIFIED', f'{name}: distinguish setpoint, measured, and trajectory.')
    mapping = _fields(physical['mapping'], {'kind', 'source_column', 'evidence'}, 'physical_conditions.mapping')
    if mapping['kind'] not in {'documented_metadata', 'documented_filename'}:
        _fail('CONDITION_UNVERIFIED', 'Condition mapping must be explicitly defined by authoritative documentation.')
    _text(mapping['source_column'], 'physical_conditions.mapping.source_column')
    mapping_evidence = _evidence(mapping['evidence'], 'physical_conditions.mapping.evidence')
    dataset = task['dataset']
    columns = dataset['columns']
    if dataset.get('rotation_speed_kind') not in {'setpoint', 'measured'}:
        _fail('CONDITION_UNVERIFIED', 'dataset.rotation_speed_kind must distinguish setpoint from measured RPM.')
    if mapping['source_column'] in {columns.get(k) for k in ('id', 'unit_id', 'specimen_id', 'label', 'split')}:
        _fail('CONDITION_UNVERIFIED', 'Acquisition/specimen/class/partition cannot define operating condition.')
    conditions = physical['conditions']
    if not isinstance(conditions, Mapping) or len(conditions) < 3:
        _fail('CONDITION_UNVERIFIED', 'At least three documented physical operating-condition tuples are required.')
    tuples: dict[tuple[Any, ...], str] = {}
    rows: dict[str, dict[str, Any]] = {}
    for cid, condition in conditions.items():
        _text(cid, 'condition_id')
        _fields(condition, {'values', 'evidence'}, f'conditions.{cid}')
        values = _fields(condition['values'], set(variables), f'conditions.{cid}.values')
        key: list[Any] = []
        row: dict[str, Any] = {'condition_id': cid}
        for name in sorted(variables):
            value = values[name]
            if variables[name]['kind'] == 'trajectory':
                profile = _fields(value, {'profile_id', 'description'}, f'{cid}.{name}')
                key.append(tuple(_text(profile[k], f'{cid}.{name}.{k}') for k in ('profile_id', 'description')))
                row[name] = f"{profile['profile_id']}: {profile['description']}"
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    _fail('CONDITION_UNVERIFIED', f'{cid}.{name} requires a finite physical value, not an unknown or ID.')
                if name == 'rpm' and value <= 0:
                    _fail('CONDITION_UNVERIFIED', f'{cid}.rpm must be positive for this rotating-machine task.')
                key.append(float(value))
                row[name] = value
            row[name + '_unit'] = variables[name]['unit']
            row[name + '_kind'] = variables[name]['kind']
        physical_tuple = tuple(key)
        if physical_tuple in tuples:
            _fail('CONDITION_UNVERIFIED', f'{cid} and {tuples[physical_tuple]} name the same physical tuple; IDs do not create a shift.')
        tuples[physical_tuple] = cid
        row.update(evidence=_evidence(condition['evidence'], f'conditions.{cid}.evidence'),
                   mapping_evidence=mapping_evidence, specimen_evidence=specimen_evidence,
                   specimens='', acquisitions='', class_support='',
                   heldout_specimens='', heldout_acquisitions='', heldout_class_support='',
                   heldout_status='NOT_INSPECTED', paired_target_specimens='')
        rows[cid] = row
    sources = list(map(str, dataset['source_domains']))
    targets = list(map(str, dataset['domain_sequence']))
    scope = dataset.get('access_scope', 'custodian')
    if scope not in {'custodian', 'legacy', 'source', 'test'}:
        _fail('SPLIT_INVALID', f'Unknown access_scope {scope!r}.')
    if len(set(sources + targets)) != len(sources + targets) or not set(sources + targets).issubset(conditions):
        _fail('SPLIT_INVALID', 'Source/target assignments must be distinct documented condition IDs.')
    if scope == 'source':
        if len(sources) < 2 or targets or len(set(conditions) - set(sources)) != 1:
            _fail('SPLIT_INVALID', 'Source binding requires >=2 sources, no target records, and one documented withheld condition.')
        targets = sorted(set(conditions) - set(sources))
    elif len(sources) < 2 or len(targets) != 1 or set(sources + targets) != set(conditions):
        _fail('SPLIT_INVALID', 'Declare >=2 source conditions and exactly one unseen target; every documented tuple must have a role.')
    for cid, row in rows.items():
        row['role'] = 'source' if cid in sources else 'target'
    if assignments is None:
        return list(rows.values())
    if not assignments:
        _fail('SPLIT_INVALID', 'No physical acquisition assignments were supplied.')
    partitions: dict[str, str] = {}
    units: dict[str, set[str]] = defaultdict(set)
    count: dict[str, int] = defaultdict(int)
    split_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_count: dict[tuple[str, str], int] = defaultdict(int)
    support: dict[tuple[str, str], set[int]] = defaultdict(set)
    acquisitions: set[str] = set()
    labels_present = []
    for record in assignments:
        cid = _record_id(record, columns, 'condition_id', 'domain', 'CONDITION_UNVERIFIED')
        sid = _record_id(record, columns, 'specimen_id', 'unit_id', 'SPECIMEN_UNVERIFIED')
        if cid not in conditions:
            _fail('CONDITION_UNVERIFIED', f'{cid} has no verified physical tuple.')
        split = record.get('split', record.get(columns.get('split')))
        if split not in {'update', 'validation', 'test', 'exclude'}:
            _fail('SPLIT_INVALID', f'{sid}: explicit update/validation/test/exclude partition required.')
        if sid in partitions and partitions[sid] != split:
            _fail('SPLIT_INVALID', f'Physical specimen {sid} crosses partitions, including repeated conditions.')
        partitions[sid] = split
        if scope == 'source' and (cid not in sources or split not in {'update', 'validation'}):
            _fail('SPLIT_INVALID', 'Source binding contains protected condition/specimen records.')
        if scope == 'test' and split != 'test':
            _fail('SPLIT_INVALID', 'Test binding may contain only held-out specimens.')
        if cid in targets and split in {'update', 'validation'}:
            _fail('SPLIT_INVALID', 'Unseen target condition cannot participate in development.')
        identifier = record.get('acquisition_id', record.get('id', record.get(columns.get('id'))))
        if identifier is not None:
            identifier = str(identifier)
            if identifier in acquisitions:
                _fail('SPLIT_INVALID', f'Duplicate acquisition {identifier}.')
            acquisitions.add(identifier)
        if split == 'exclude':
            continue
        _record_rpm(dataset, physical, record, cid)
        units[cid].add(sid)
        count[cid] += 1
        split_units[cid, split].add(sid)
        split_count[cid, split] += 1
        label = record.get('label', record.get(columns.get('label')))
        labels_present.append(label is not None)
        if label is not None:
            try:
                label = dataset.get('label_map', {}).get(str(label), label)
                numeric = float(label)
                if isinstance(label, bool) or not math.isfinite(numeric) or numeric != int(numeric):
                    raise ValueError('nonintegral class')
                label = int(numeric)
            except (TypeError, ValueError, OverflowError):
                _fail('SPLIT_INVALID', 'Assignments require the declared integral class mapping.')
            if not 0 <= label < int(task['model']['num_classes']):
                _fail('SPLIT_INVALID', f'Class {label} is outside the declared taxonomy.')
            support[cid, split].add(label)
    if any(labels_present) and not all(labels_present):
        _fail('SPLIT_INVALID', 'Partially populated labels cannot establish per-condition class support.')
    if scope in {'source', 'test'} and not all(labels_present):
        _fail('SPLIT_INVALID', 'Isolated source/test records require labels for their declared evaluation scope.')
    expected = set(range(int(task['model']['num_classes'])))
    if all(labels_present) and scope != 'test':
        for cid in sources:
            for split in ('update', 'validation'):
                if support[cid, split] != expected:
                    _fail('SPLIT_INVALID', f'{cid}/{split} lacks full class support: expected {sorted(expected)}, observed {sorted(support[cid, split])}.')
    if scope != 'source':
        for cid in targets:
            if not split_units[cid, 'test']:
                _fail('SPLIT_INVALID', f'{cid}: primary target requires held-out physical specimens.')
            if all(labels_present) and support[cid, 'test'] != expected:
                _fail('SPLIT_INVALID', f'{cid}/test lacks full target class support: expected {sorted(expected)}, observed {sorted(support[cid, "test"])}.')
    target_units = set().union(*(split_units[cid, 'test'] for cid in targets))
    for cid, row in rows.items():
        heldout = split_units[cid, 'test']
        labels = support[cid, 'test']
        if scope == 'source':
            availability = 'NOT_ACCESSED'
        elif not heldout:
            availability = 'ABSENT'
        elif not all(labels_present):
            availability = 'PRESENT_CLASS_SUPPORT_UNVERIFIED'
        elif labels == expected and target_units.issubset(heldout):
            availability = 'COMPLETE'
        else:
            availability = 'PARTIAL'
        row.update(specimens=len(units[cid]), acquisitions=count[cid],
                   class_support=';'.join(f'{split}:{",".join(map(str, sorted(labels)))}'
                                          for (condition, split), labels in sorted(support.items()) if condition == cid and labels),
                   heldout_specimens=len(heldout), heldout_acquisitions=split_count[cid, 'test'],
                   heldout_class_support=','.join(map(str, sorted(labels))), heldout_status=availability,
                   paired_target_specimens=len(heldout & target_units))
    return list(rows.values())
