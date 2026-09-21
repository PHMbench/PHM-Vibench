"""Source-validation figures from saved CSV only; never load a model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed acceptance directory')
    args = parser.parse_args(argv)
    directory = args.output
    run = json.loads((directory/'run.json').read_text())
    if run.get('status') != 'completed_source_acceptance_not_transfer':
        raise ValueError('plot requires completed export, restore and selection-score checks')
    report = json.loads((directory/'analysis.json').read_text())
    if report.get('scope') != 'descriptive_source_validation_not_independent_test':
        raise ValueError('wrong result scope for source-validation plot')
    metrics = pd.read_csv(directory/'full_source_metrics.csv', dtype={'dataset': str})
    differences = pd.read_csv(directory/'increment_mask_group_differences.csv', dtype={'dataset': str})
    if (metrics.empty or metrics.dataset.duplicated().any() or not np.isfinite(metrics.nll).all()
            or set(metrics.dataset) != set(differences.dataset)):
        raise ValueError('complete finite source CSVs required')
    out = directory/'figures'
    out.mkdir(exist_ok=True)
    # Editable text; no embedded image, checkpoint, inference or invented values.
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 10})
    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout='constrained')
    ax.scatter(metrics.dataset, metrics.nll)
    ax.set(xlabel='Source dataset ID', ylabel='Group-balanced NLL (nat)',
           title='Selected-model source validation (not transfer)')
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(out/f'source_validation.{suffix}', dpi=300)
    plt.close(fig)
    values = differences.groupby('dataset', sort=False).delta_mask_minus_full_nll.mean()
    if not np.isfinite(values).all():
        raise ValueError('nonfinite masking differences')
    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout='constrained')
    ax.scatter(values.index, values.values)
    ax.axhline(0, linewidth=0.7, linestyle='--')
    ax.set(xlabel='Source dataset ID', ylabel='Masked minus full NLL (nat)',
           title='Frozen-model sensitivity (not retraining ablation)')
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(out/f'increment_mask_sensitivity.{suffix}', dpi=300)
    plt.close(fig)
    (out/'README.md').write_text(
        'Descriptive source-validation plots; no CI or target-transfer claim.\n'
        'Sources: ../full_source_metrics.csv and ../increment_mask_group_differences.csv.\n'
        'Regenerate: python -m scripts.tii_plot_source --output <run-directory>\n')


if __name__ == '__main__':
    main()
