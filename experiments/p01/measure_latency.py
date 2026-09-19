"""Measure actual frozen prediction paths on one source-validation window.

No fitting, assessment or test access. Inputs are already on the GPU during
timing; filesystem reads, construction and checkpoint loading are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from experiments.p01.fusion_data import read_records
from experiments.p01.fusion_deployment import load_model, log_predictions
from experiments.p01.window_io import window_record


def deployed_prediction(model, x, deployment):
    """Time the coefficient's real path, including the zero-correction fast path."""
    alpha = float(deployment['alpha'])
    if alpha == 0.0:
        return F.log_softmax(model.reference(x) / float(model.reference_temperature), -1)
    if deployment['kind'] == 'temperature':
        raw = model.reference(x) / float(model.reference_temperature)
        candidate = F.log_softmax(raw / float(deployment['temperature']), -1)
        if alpha == 1.0:
            return candidate
        return torch.logaddexp(F.log_softmax(raw, -1) + np.log1p(-alpha),
                               candidate + np.log(alpha))
    # Fusion.forward consumes the coefficient restored from its deployment
    # bundle. A plain supervised baseline has no candidate-head alpha buffer.
    if hasattr(model, 'alpha'):
        if not math.isclose(float(model.alpha), alpha, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError('Restored model coefficient differs from its deployment metadata.')
        return model(x)
    if alpha == 1.0:
        return F.log_softmax(model.candidate(x), -1)
    return log_predictions(model, x, 'model', 1.0, alpha)[2]


def direct_prediction(model, x, deployment):
    if deployment['kind'] == 'temperature':
        logits = model.reference(x) / (float(model.reference_temperature) * float(deployment['temperature']))
    elif hasattr(model, 'candidate'):
        logits = model.candidate(x)
    else:
        logits = model.forward_details(x)['candidate_logits']
    return F.log_softmax(logits, -1)


@torch.inference_mode()
def measure(bundle: Path, data_path: Path, dataset_name: str, output: Path) -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not torch.cuda.is_available():
        raise RuntimeError('Latency protocol requires physical GPU0 only: CUDA_VISIBLE_DEVICES=0.')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    data = yaml.safe_load(data_path.read_text())
    dataset = next(d for d in data['datasets'] if d['name'] == dataset_name)
    sources = set(map(str, dataset['source_domains']))
    records = read_records(dataset, data)
    record = next(r for r in records if r['split'] == 'validation' and r['domain'] in sources)
    x = window_record(record, dataset, data['data'])[:1].to('cuda:0')
    model, saved = load_model(bundle, 'cuda:0')
    deployment = saved['deployment']
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    paths = {
        'p0': lambda: F.log_softmax(model.reference(x) / float(model.reference_temperature), -1),
        'direct_q': lambda: direct_prediction(model, x, deployment),
        'deployed': lambda: deployed_prediction(model, x, deployment),
    }
    rows = []
    for name, predict in paths.items():
        for _ in range(20):
            predict()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = predict()
            torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
            if not torch.isfinite(result).all():
                raise FloatingPointError('Nonfinite latency predictor output.')
        q1, median, q3 = np.quantile(durations, [.25, .5, .75])
        rows.append(dict(path=name, median_ms=float(median), q1_ms=float(q1),
                         q3_ms=float(q3), iqr_ms=float(q3-q1),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         repeats_ms=durations))
    if any(not torch.equal(value, model.state_dict()[key].detach().cpu()) for key, value in before.items()):
        raise RuntimeError('Inference modified model parameters or buffers.')
    output.mkdir(parents=True)
    (output / 'latency.json').write_text(json.dumps(dict(
        bundle=str(bundle.resolve()), data_config=str(data_path.resolve()), dataset=dataset_name,
        device='physical GPU0 / cuda:0', gpu=torch.cuda.get_device_name(0),
        dtype=str(x.dtype), input_shape=list(x.shape), acquisition_id=record['acquisition_id'],
        group_id=record['unit_id'], domain=record['domain'], split='validation',
        window_id='0', warmup=20, repeats=100, deployment=deployment,
        io_boundary='input already on GPU; excludes construction/checkpoint/data I/O',
        total_parameters=sum(p.numel() for p in model.parameters()), paths=rows), indent=2))
    with (output / 'latency.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != 'repeats_ms'])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != 'repeats_ms'} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--data-config', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measure(args.bundle, args.data_config, args.dataset, args.output)


if __name__ == '__main__':
    main()
