"""Apply an explicitly selected saved joint policy without refitting."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import box_bounds, class_certificate, max_box_average


def predict(model: Path, data: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    with np.load(model, allow_pickle=False) as saved, np.load(data, allow_pickle=False) as archive:
        required = {'feature_names', 'mean', 'std', 'centers', 'scales', 'q',
                    'rule_cost', 'radius', 'threshold', 'certified'}
        if not required <= set(saved.files) or not {'x', 'feature_names'} <= set(archive.files):
            raise ValueError('joint checkpoint or input feature arrays are missing')
        raw = archive['x']
        names, mean, std, centers = saved['feature_names'], saved['mean'], saved['std'], saved['centers']
        # Check the raw shape before subtraction: N x 1 can broadcast to N x D.
        if (raw.ndim != 2 or raw.dtype.kind not in {'i', 'u', 'f'}
                or not np.isfinite(raw).all() or names.ndim != 1
                or mean.shape != (raw.shape[1],) or std.shape != mean.shape
                or len(names) != raw.shape[1] or centers.ndim != 2
                or centers.shape[1] != raw.shape[1]):
            raise ValueError('raw x must be a finite N x D matrix matching checkpoint feature dimensions')
        if (not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('checkpoint mean/std must be finite and std positive')
        if not np.array_equal(names, archive['feature_names']):
            raise ValueError('feature names/order differ from training')
        if (saved['certified'].shape != () or saved['certified'].dtype != bool
                or saved['threshold'].shape != () or not np.isfinite(saved['threshold'])):
            raise ValueError('checkpoint certified must be a scalar boolean and threshold a finite scalar')
        x = (raw - mean) / std
        lo, hi, weights = box_bounds(x, centers, saved['scales'], float(saved['radius']))
        probability = weights @ saved['q']
        prediction = probability.argmax(axis=1)
        stable = class_certificate(lo, hi, saved['q'], prediction)
        score = max_box_average(lo, hi, saved['rule_cost'])
        accepted = bool(saved['certified']) & stable & (score <= float(saved['threshold']))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            np.savez_compressed(stream, probabilities=probability, prediction=prediction,
                                accepted=accepted, score=score, stable=stable, weights=weights)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(predict(**vars(args)))


if __name__ == '__main__':
    main()
