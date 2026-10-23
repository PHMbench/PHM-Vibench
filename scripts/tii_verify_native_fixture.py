"""Verify completed native M0 fixture checkpoints without training again.

The input is the two successful public CLI runs made from native.yaml. Outputs
describe constructed source fixtures, never industrial targets or query scores.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytorch_lightning as pl
import torch

from phmfactory.config import analyze_config
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task


def assert_same(left, right) -> None:
    """Compare a saved tensor/container state without numerical tolerance."""
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right, (left, right)


def native_objects(config: Path, arm: str):
    resolved = analyze_config(config, override_values=[f'model.token_organization={arm}'])
    args = {key: SimpleNamespace(**value) for key, value in resolved.runtime_config().items()
            if isinstance(value, dict)}
    data, model_args = args['data'], args['model']
    assert data.evidence_kind == 'tensor_fixture'
    assert data.rounds == 20 and args['environment'].seed == 0
    assert model_args.source_rms == 'source_train'
    assert (model_args.num_patches, model_args.patch_size_L) == (data.num_patches, data.patch_size)
    factory = build_data(data, args['task'])
    model_args.source_rms = factory.source_rms
    model = build_model(model_args, metadata=factory.get_metadata())
    task = build_task(args_task=args['task'], network=model, args_data=data,
                      args_model=model_args, args_trainer=args['trainer'],
                      args_environment=args['environment'], metadata=factory.get_metadata())
    return factory, task.eval()


def prediction_path(task, batch) -> tuple[torch.Tensor, ...]:
    network = task.network
    x, p, a = (batch[key] for key in ('x', 'incremental', 'availability'))
    with torch.no_grad():
        tokens = network.embedding(x, p, a)
        with patch.object(network, '_head', side_effect=AssertionError('encoder invoked a source head')):
            features = network.encode(x, incremental=p, availability=a)
        logits = task(batch)
    return tokens, features, logits


def verify_arm(root: Path, arm: str) -> tuple[dict, list[dict]]:
    stdout = (root / f'{arm}_stdout.txt').read_text().splitlines()
    assert 'run=completed' in stdout
    paths = [line.split('=', 1)[1] for line in stdout if line.startswith('best_checkpoint=')]
    assert len(paths) == 1
    checkpoint = Path(paths[0])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['global_step'] == 20
    assert saved['hyper_parameters']['model']['token_organization'] == arm
    assert len(saved['optimizer_states']) == 1 and not saved['lr_schedulers']
    source, task = native_objects(root / 'native.yaml', arm)
    assert source.source_rms == saved['hyper_parameters']['model']['source_rms']
    recorded_rms = pd.read_csv(checkpoint.parent / 'source_rms.csv').source_rms.iloc[0]
    assert abs(recorded_rms - source.source_rms) < 1e-15
    pd.testing.assert_frame_equal(pd.DataFrame(source.window_inventory),
                                  pd.read_csv(checkpoint.parent / 'source_windows.csv'))
    task.load_state_dict(saved['state_dict'], strict=True)
    assert task.network.embedding.source_rms.item() == source.source_rms
    optimizer = task.configure_optimizers()
    optimizer.load_state_dict(saved['optimizer_states'][0])
    assert_same(optimizer.state_dict(), saved['optimizer_states'][0])
    assert len(optimizer.state) == len(list(task.network.parameters()))
    assert {int(value['step']) for value in optimizer.state.values()} == {20}

    # Round-trip the actual trained state, then independently construct all
    # three factories. No optimizer step or Trainer.fit occurs in this script.
    stream = io.BytesIO()
    torch.save({'task': task.state_dict(), 'optimizer': optimizer.state_dict()}, stream)
    stream.seek(0)
    replay = torch.load(stream, map_location='cpu', weights_only=True)
    restored_source, restored = native_objects(root / 'native.yaml', arm)
    restored.load_state_dict(replay['task'], strict=True)
    restored_optimizer = restored.configure_optimizers()
    restored_optimizer.load_state_dict(replay['optimizer'])
    assert_same(optimizer.state_dict(), restored_optimizer.state_dict())
    assert_same(task.state_dict(), restored.state_dict())
    assert source.source_rms == restored_source.source_rms
    assert source.window_inventory == restored_source.window_inventory
    for round_index in range(20):
        assert_same(source.train_dataset[round_index], restored_source.train_dataset[round_index])
    expected_heads = set(map(str, task.sources))
    assert set(task.network.task_head.mutiple_fc) == expected_heads
    assert set(restored.network.task_head.mutiple_fc) == expected_heads

    rows = []
    task.on_validation_epoch_start()
    for batch_index, (batch, restored_batch) in enumerate(zip(source.val_dataset, restored_source.val_dataset)):
        assert_same(batch, restored_batch)
        before, after = prediction_path(task, batch), prediction_path(restored, restored_batch)
        assert_same(before, after)
        task.validation_step(batch, batch_index)
        source_windows = [row for row in source.window_inventory
                          if row['dataset'] == batch['source'] and row['role'] == 'source_val']
        # Fixture has one complete batch/source; reject a different population.
        assert len(source_windows) == len(batch['y'])
        for index, window in enumerate(source_windows):
            assert window['file_id'] == batch['file_id'][index]
            assert window['recording_id'] == batch['recording_id'][index]
            logits = before[2][index]
            rows.append(dict(evidence_kind='tensor_fixture', method=arm, seed=0,
                **window, true_label=int(batch['y'][index]),
                availability=int(batch['availability'][index]),
                logits=json.dumps(logits.tolist()), probabilities=json.dumps(logits.softmax(-1).tolist()),
                checkpoint=str(checkpoint)))
    assert len(source.val_dataset) == len(restored_source.val_dataset) == 2
    assert len(rows) == 16
    selection = next(value for value in saved['callbacks'].values() if value.get('monitor') == 'val_group_nll')
    assert selection['best_model_path'] == str(checkpoint)
    val_nll, val_acc = task.validation_risk()
    assert abs(val_nll - selection['best_model_score'].item()) < 1e-7
    assert set(task.network.task_head.mutiple_fc) == expected_heads
    assert set(restored.network.task_head.mutiple_fc) == expected_heads
    return dict(method=arm, status='PASS', checkpoint=str(checkpoint), global_step=20,
        source_rms=source.source_rms, source_heads=sorted(expected_heads),
        optimizer_parameter_states=len(optimizer.state), optimizer_steps=[20],
        sampler_rounds_bitwise_equal=20, source_validation_windows=len(rows),
        tokens_features_logits_bitwise_equal=True, rms_optimizer_bitwise_equal=True,
        source_val_group_nll=val_nll, source_val_group_acc=val_acc,
        checkpoint_score_matches_recomputed_source_validation=True), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    torch.set_num_threads(1)
    results, predictions = [], []
    for arm in ('ordinary', 'support'):
        result, rows = verify_arm(args.fixture.resolve(), arm)
        results.append(result)
        predictions.extend(rows)
    table = pd.DataFrame(predictions)
    assert not table.duplicated(['method', 'dataset', 'recording_id', 'channel', 'window_start', 'window_end']).any()
    assert set(table.role) == {'source_val'}
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / 'native_source_fixture_predictions.csv', index=False)
    report = dict(status='PASS', evidence_kind='tensor_fixture', config=str((args.fixture/'native.yaml').resolve()),
        command=' '.join(sys.orig_argv), python=platform.python_version(), torch=torch.__version__,
        pytorch_lightning=pl.__version__, device='cpu', elapsed_seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, arms=results,
        limitations=['Constructed source fixtures; no industrial qualification, target head or query evaluation.',
                     'Replays saved states and the fixed round sampler; does not resume Trainer loops or take a new optimizer step.',
                     'Only CPU bitwise equality is established.'])
    (args.output / 'native_recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Native M0 fixture recovery', '', 'PASS — constructed source fixtures, not J1-C industrial evidence.', '',
             f"Command: `{report['command']}`", '',
             f"CPU verification: {report['elapsed_seconds']:.3f} s; peak RSS {report['peak_rss_kib']} KiB.", '',
             'Both completed CLI checkpoints have global_step=20, 16 populated AdamW parameter states and only source heads 1/2.',
             'Fresh native Data/Model/Task Factory reconstruction preserves all 20 sampled rounds, source RMS, optimizer state,',
             'and tokens/features/logits bitwise across save/load. Source validation checkpoint scores match recomputation.', '',
             'The 32 rows in native_source_fixture_predictions.csv are source_val windows (16 per arm), not target queries.', '',
             *report['limitations']]
    (args.output / 'native_recovery.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
