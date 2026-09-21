"""Offline query validation and group NLL against independently saved inventories.

The split custodian supplies record, support, query and complete window CSVs;
none is inferred from predictions. ``expected_windows_file`` includes every
declared method/seed and its checkpoint, including runs with no predictions.
CSV identity fields are text (so leading zeroes survive); true_label is the
integer local class value, and class-map order is the logits column order.
This module does not train, select checkpoints, filter invalid rows or resample.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from src.utils.identifiers import validate_identifiers


RECORD_COLUMNS = ("dataset", "recording_id", "group", "true_label")
EPISODE_KEY = ("target", "fold", "episode", "dataset", "recording_id")
SPLIT_COLUMNS = EPISODE_KEY + (
    "group", "role", "true_label", "support_split_id", "query_split_id",
)
PREDICTION_KEY = (
    "target", "fold", "method", "seed", "episode", "recording_id", "channel",
    "window_start", "window_end",
)
WINDOW_COLUMNS = SPLIT_COLUMNS + (
    "method", "seed", "channel", "window_start", "window_end", "checkpoint",
)
_INTEGER_COLUMNS = {"seed", "episode", "true_label", "window_start", "window_end"}


def _read_inventory(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing inventory/split file: {path}")
    table = pd.read_csv(path, dtype=object)
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    for column in columns:
        if column in ("logits", "probabilities"):
            continue
        values = table[column].tolist()
        if column not in _INTEGER_COLUMNS:
            values = [None if isinstance(v, str) and v.strip().casefold() in
                      {"none", "null", "nan", "inf", "+inf", "-inf",
                       "infinity", "+infinity", "-infinity"} else v for v in values]
        validate_identifiers(values, f"{path.name}:{column}")
        if column in _INTEGER_COLUMNS:
            numeric = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
            if (not np.isfinite(numeric).all() or (numeric < 0).any()
                    or (numeric != np.floor(numeric)).any()
                    or (numeric >= 2**53).any()):
                raise ValueError(f"{path.name}:{column}: exact nonnegative integers required")
            table[column] = numeric.astype(np.int64)
    return table


def _unique(table: pd.DataFrame, key: Sequence[str], name: str) -> None:
    if table.duplicated(list(key)).any():
        raise ValueError(f"{name}: duplicate composite key {tuple(key)}")


def _keys(table: pd.DataFrame, columns: Sequence[str]) -> set[tuple]:
    return set(table.loc[:, list(columns)].itertuples(index=False, name=None))


def _match_records(table: pd.DataFrame, records: pd.DataFrame, name: str) -> None:
    actual = table.loc[:, list(RECORD_COLUMNS)].drop_duplicates()
    reference = records.loc[:, list(RECORD_COLUMNS)]
    if not _keys(actual, RECORD_COLUMNS) <= _keys(reference, RECORD_COLUMNS):
        raise ValueError(f"{name}: dataset/record/group/label provenance differs from records")


def _validate_query_rows(
    table: pd.DataFrame, query: pd.DataFrame, support: pd.DataFrame, name: str,
) -> None:
    if not table["role"].eq("query").all():
        raise ValueError(f"{name}: only predeclared query role is allowed")
    group_key = ("target", "fold", "episode", "dataset", "group")
    if _keys(table, group_key) & _keys(support, group_key):
        raise ValueError(f"{name}: support group contamination (including another recording)")
    if not _keys(table, SPLIT_COLUMNS) <= _keys(query, SPLIT_COLUMNS):
        raise ValueError(f"{name}: row differs from independent query split")


def _prediction_arrays(
    table: pd.DataFrame, local_class_map: Mapping[str, Sequence[int]], *, context="target",
) -> tuple[list[np.ndarray], list[int]]:
    arrays, label_columns = [], []
    for index, row in table.iterrows():
        classes = tuple(local_class_map[row[context]])
        if row["true_label"] not in classes:
            raise ValueError(f"prediction row {index}: true_label absent from local class map")
        try:
            logits = np.asarray(json.loads(row["logits"]), dtype=np.float64)
            probabilities = np.asarray(json.loads(row["probabilities"]), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction row {index}: numeric JSON logits/probabilities required") from exc
        if (logits.shape != (len(classes),) or probabilities.shape != logits.shape
                or not np.isfinite(logits).all() or not np.isfinite(probabilities).all()):
            raise ValueError(f"prediction row {index}: finite vectors matching local class map required")
        if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-7, rtol=0):
            raise ValueError(f"prediction row {index}: nonnegative probabilities summing to one required")
        shifted = logits - logits.max()
        exponential = np.exp(shifted)
        expected = exponential / exponential.sum()
        if not np.allclose(probabilities, expected, atol=1e-7, rtol=1e-6):
            raise ValueError(f"prediction row {index}: probabilities disagree with logits")
        arrays.append(logits)
        label_columns.append(classes.index(row["true_label"]))
    return arrays, label_columns


def evaluate_query_predictions(
    predictions_file: str | Path, *, records_file: str | Path,
    support_file: str | Path, query_file: str | Path,
    expected_windows_file: str | Path,
    local_class_map: Mapping[str, Sequence[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the whole declared result before returning window and group NLL.

    Records are unique by (dataset, recording_id). Support/query CSVs contain
    SPLIT_COLUMNS and one row per episode-record. The predeclared window CSV
    contains WINDOW_COLUMNS, expanded over every planned method and seed.
    Checkpoint and split IDs must exactly match those declarations. These are
    independently supplied inputs, not inventories generated from predictions.
    NLL averages windows within each group/seed, then averages seeds per group.
    All files must exist and be distinct. No partial metric is returned on error.
    """
    paths = [Path(p).resolve() for p in (
        predictions_file, records_file, support_file, query_file, expected_windows_file,
    )]
    if len(set(paths)) != len(paths):
        raise ValueError("predictions and independent record/split/window files must be distinct")
    predictions = _read_inventory(paths[0], WINDOW_COLUMNS + ("logits", "probabilities"))
    records = _read_inventory(paths[1], RECORD_COLUMNS)
    support = _read_inventory(paths[2], SPLIT_COLUMNS)
    query = _read_inventory(paths[3], SPLIT_COLUMNS)
    expected = _read_inventory(paths[4], WINDOW_COLUMNS)

    _unique(records, ("dataset", "recording_id"), "records")
    _unique(support, EPISODE_KEY, "support split")
    _unique(query, EPISODE_KEY, "query split")
    _unique(expected, PREDICTION_KEY, "predeclared windows")
    _unique(predictions, PREDICTION_KEY, "predictions")
    if not support["role"].eq("support").all() or not query["role"].eq("query").all():
        raise ValueError("split role differs from its independently declared support/query file")
    split_context = ("target", "fold", "episode", "dataset", "support_split_id", "query_split_id")
    if _keys(support, split_context) != _keys(query, split_context):
        raise ValueError("support/query split contexts differ")
    for name, table in (("support split", support), ("query split", query),
                        ("predeclared windows", expected), ("predictions", predictions)):
        _match_records(table, records, name)
    _validate_query_rows(query, query, support, "query split")
    _validate_query_rows(expected, query, support, "predeclared windows")
    _validate_query_rows(predictions, query, support, "predictions")
    if _keys(expected, EPISODE_KEY) != _keys(query, EPISODE_KEY):
        raise ValueError("predeclared windows do not cover every query record")
    context_key = ["target", "fold", "episode", "dataset"]
    window_key = ["recording_id", "channel", "window_start", "window_end"]
    for context, declared in expected.groupby(context_key, sort=False):
        query_context = query
        for column, value in zip(context_key, context):
            query_context = query_context.loc[query_context[column] == value]
        reference = None
        for _, run in declared.groupby(["method", "seed"], sort=False):
            current = _keys(run, window_key)
            if set(run["recording_id"]) != set(query_context["recording_id"]):
                raise ValueError("each declared method/seed must cover every query record")
            if reference is not None and current != reference:
                raise ValueError("declared methods/seeds must share the same query window inventory")
            if run["checkpoint"].nunique() != 1:
                raise ValueError("each declared method/seed must use one frozen checkpoint")
            reference = current
    for name, table in (("predeclared windows", expected), ("predictions", predictions)):
        if (table["window_end"] <= table["window_start"]).any():
            raise ValueError(f"{name}: window_end must exceed window_start")
    if _keys(predictions, WINDOW_COLUMNS) != _keys(expected, WINDOW_COLUMNS):
        raise ValueError("predictions differ from complete predeclared query window inventory")

    if set(local_class_map) != set(query["target"]):
        raise ValueError("local class maps must cover exactly the predeclared targets")
    for target, classes in local_class_map.items():
        checked = validate_identifiers(classes, f"{target}:local class map")
        if (len(checked) < 2 or len(set(checked)) != len(checked)
                or any(not isinstance(c, (int, np.integer)) or isinstance(c, (bool, np.bool_))
                       or c < 0 for c in checked)):
            raise ValueError(f"{target}: unique nonnegative integer local class map required")
        if not set(query.loc[query["target"] == target, "true_label"]) <= set(checked):
            raise ValueError(f"{target}: query label absent from local class map")
    arrays, labels = _prediction_arrays(predictions, local_class_map)

    losses = []
    for logits, label_column in zip(arrays, labels):
        centered = logits - logits.max()
        losses.append(float(logsumexp(centered) - centered[label_column]))
    if not np.isfinite(losses).all():
        raise ValueError("logits produce nonfinite NLL; no group metric may be reported")
    windows = predictions.copy()
    windows["nll"] = losses
    group_key = ["target", "fold", "method", "episode", "dataset", "group"]
    per_seed = windows.groupby(group_key + ["seed"], sort=False, dropna=False).agg(
        nll=("nll", "mean"), window_count=("nll", "size"),
    ).reset_index()
    groups = per_seed.groupby(group_key, sort=False, dropna=False).agg(
        nll=("nll", "mean"), seed_count=("seed", "size"),
        prediction_count=("window_count", "sum"),
    ).reset_index()
    return windows, groups


SOURCE_KEY = ('dataset', 'recording_id', 'channel', 'window_start', 'window_end')
SOURCE_COLUMNS = SOURCE_KEY + ('group', 'role', 'file_id', 'true_label')


def evaluate_source_predictions(predictions_file, *, expected_windows_file,
                                local_class_map, checkpoint, intervention):
    """Descriptive source-validation risk, not a held-out transfer estimator.

    Expected windows include BOTH source_train and source_val, saved by the
    native Data Factory before fit. No split is derived from predictions.
    The native CE label-to-column mapping must be contiguous per local head.
    """
    if Path(predictions_file).resolve() == Path(expected_windows_file).resolve():
        raise ValueError('predictions cannot define their own expected population')
    expected = _read_inventory(Path(expected_windows_file), SOURCE_COLUMNS)
    predictions = _read_inventory(Path(predictions_file), SOURCE_COLUMNS + (
        'checkpoint', 'method', 'seed', 'intervention', 'logits', 'probabilities'))
    _unique(expected, SOURCE_KEY, 'source window inventory')
    _unique(predictions, SOURCE_KEY, 'source predictions')
    if not expected.role.isin(['source_train', 'source_val']).all():
        raise ValueError('source inventory cannot contain query/target roles')
    if not predictions.role.eq('source_val').all():
        raise ValueError('source predictions must contain source_val only')
    if (not predictions.checkpoint.eq(str(checkpoint)).all()
            or not predictions.intervention.eq(intervention).all()
            or not predictions.method.eq('support').all() or not predictions.seed.eq(0).all()):
        raise ValueError('source predictions disagree with the one-model evaluation condition')
    for source, rows in expected.groupby('dataset', sort=False):
        train = rows[rows.role == 'source_train']
        val = rows[rows.role == 'source_val']
        if train.empty or val.empty or set(train.group) & set(val.group):
            raise ValueError('source train/validation require nonempty disjoint groups')
        if set(train.recording_id) & set(val.recording_id):
            raise ValueError('source recording leakage across roles')
        identities = rows[['recording_id', 'group', 'file_id', 'true_label', 'role']].drop_duplicates()
        if identities.recording_id.duplicated().any():
            raise ValueError('record identity changes group, file, label or role across windows')
    if (expected.window_end <= expected.window_start).any():
        raise ValueError('source window_end must exceed window_start')
    val = expected[expected.role == 'source_val']
    if _keys(predictions, SOURCE_COLUMNS) != _keys(val, SOURCE_COLUMNS):
        raise ValueError('source predictions omit or alter predeclared validation windows')
    if set(local_class_map) != set(expected.dataset):
        raise ValueError('class map must cover every selected source exactly')
    for source, classes in local_class_map.items():
        if (len(classes) < 2 or any(type(c) is not int for c in classes)
                or list(classes) != list(range(len(classes)))):
            raise ValueError('native source head requires ordered contiguous local class columns')
        if not set(expected.loc[expected.dataset == source, 'true_label']) <= set(classes):
            raise ValueError('source label absent from its fixed class map')
    arrays, labels = _prediction_arrays(predictions, local_class_map, context='dataset')
    windows = predictions.copy()
    windows['nll'] = [float(logsumexp(x-x.max()) - (x-x.max())[y]) for x, y in zip(arrays, labels)]
    windows['correct'] = [int(x.argmax() == y) for x, y in zip(arrays, labels)]
    if not np.isfinite(windows.nll).all():
        raise ValueError('nonfinite source risk')
    groups = windows.groupby(['dataset', 'group'], sort=False).agg(
        nll=('nll', 'mean'), accuracy=('correct', 'mean'), windows=('nll', 'size')).reset_index()
    metrics = groups.groupby('dataset', sort=False).agg(
        nll=('nll', 'mean'), accuracy=('accuracy', 'mean'), groups=('group', 'size'),
        windows=('windows', 'sum')).reset_index()
    metrics['scope'] = 'checkpoint_selection_source_validation'
    metrics['intervention'] = intervention
    return windows, groups, metrics


def analyze_source_acceptance(output):
    """Analyze retained observations only. No model, loader or optimizer import."""
    output = Path(output)
    run = json.loads((output/'run.json').read_text())
    if run.get('mode') != 'one_model_acceptance_20' or run.get('model_fits_completed') != 1:
        raise ValueError('analysis requires one completed bounded fit')
    if not run.get('export_completed'):
        raise ValueError('full source prediction export is incomplete')
    classes = json.loads((output/'local_class_map.json').read_text())
    checkpoint = run['best_checkpoint']
    metrics_by_mode, groups_by_mode = {}, {}
    for intervention, filename in (
        ('full', 'source_validation_predictions.csv'),
        ('mask_increment', 'source_validation_masked_predictions.csv'),
    ):
        windows, groups, metrics = evaluate_source_predictions(output/filename,
            expected_windows_file=output/'expected_source_windows.csv', local_class_map=classes,
            checkpoint=checkpoint, intervention=intervention)
        metrics_by_mode[intervention], groups_by_mode[intervention] = metrics, groups
        windows.to_csv(output/f'{intervention}_window_metrics.csv', index=False)
        groups.to_csv(output/f'{intervention}_group_metrics.csv', index=False)
        metrics.to_csv(output/f'{intervention}_source_metrics.csv', index=False)
    paired = groups_by_mode['full'].merge(groups_by_mode['mask_increment'],
        on=['dataset', 'group'], suffixes=('_full', '_mask'), validate='one_to_one')
    paired['delta_mask_minus_full_nll'] = paired.nll_mask-paired.nll_full
    paired.to_csv(output/'increment_mask_group_differences.csv', index=False)
    gradients = pd.read_csv(output/'audit/source_gradients.csv', dtype={'dataset': str})
    updates = pd.read_csv(output/'audit/joint_updates.csv')
    sources = set(classes)
    expected_pairs = {(r, s) for r in range(1, 21) for s in sources}
    if (gradients.duplicated(['round', 'dataset']).any()
            or set(zip(gradients['round'], gradients.dataset)) != expected_pairs):
        raise ValueError('gradient observations must include every source at each of 20 real updates')
    norm_columns = ['encoder_grad_norm', 'own_head_grad_norm', 'other_head_grad_norm']
    if not np.isfinite(gradients[norm_columns].to_numpy()).all() or (gradients[norm_columns] < 0).any().any():
        raise ValueError('gradient observations must be finite nonnegative norms')
    if not gradients.other_head_grad_norm.eq(0).all() or not gradients.windows.eq(32).all():
        raise ValueError('source-head gradient contamination or unequal source exposure')
    for source in sources:
        rows = gradients[gradients.dataset == source]
        if rows.encoder_grad_norm.max() <= 0 or rows.own_head_grad_norm.max() <= 0:
            raise ValueError(f'source {source} has no observed gradient reaching shared encoder or local head')
        if any(len(json.loads(x)) != 32 for x in rows.groups):
            raise ValueError('source group exposure is incomplete')
        if any(sum(json.loads(x).values()) != 32 for x in rows.class_counts):
            raise ValueError('source class exposure is incomplete')
    delta_columns = ['joint_encoder_parameter_delta'] + [f'head_{s}_joint_delta' for s in sorted(sources)]
    if not set(delta_columns) <= set(updates):
        raise ValueError('joint update observations must include the encoder and every local head')
    if (updates['round'].duplicated().any() or set(updates['round']) != set(range(1, 21))
            or not np.isfinite(updates[delta_columns].to_numpy()).all()
            or (updates[delta_columns] < 0).any().any()
            or updates.joint_encoder_parameter_delta.max() <= 0):
        raise ValueError('actual joint optimizer update evidence is incomplete/nonfinite/zero')
    report = dict(scope='descriptive_source_validation_not_independent_test',
        model_fits_completed=1, source_validation_nll=float(metrics_by_mode['full'].nll.mean()),
        masked_source_validation_nll=float(metrics_by_mode['mask_increment'].nll.mean()),
        actual_joint_updates=20, observed_sources=len(sources), windows_per_source=640,
        gradient_head_isolation=True, source_gradients_reach_shared_encoder=True,
        joint_update_observed=True, transfer_delta=None, pooling_interaction=None,
        uncertainty='not estimated: one selected model and checkpoint-selection validation population',
        mechanism_boundary='masking measures frozen-model sensitivity, not retrained information value; joint deltas are not per-source causal updates')
    (output/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
