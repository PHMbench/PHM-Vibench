"""CSV-only source learning curves. No model, checkpoint or inference imports."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output
    report = json.loads((root/'run.json').read_text())
    if (report.get('mode') != 'one_model_fit95' or report.get('model_fits_completed') != 1
            or not report.get('export_completed') or not report.get('selection_score_recomputed')):
        raise ValueError('complete native fit, restored exports and metric recomputation required')
    table = pd.read_csv(root/'source_fit_curve.csv', dtype={'dataset': str})
    figures = root/'fit_figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 9})
    # Distinct plots, default matplotlib colors, no predicted/expected values.
    for source, rows in table.groupby('dataset', sort=True):
        for metric, label in [('accuracy','Group-balanced accuracy'), ('nll','Group-balanced NLL')]:
            stem = f'source_{source}_{metric}'
            if all((figures/f'{stem}.{ext}').is_file() and (figures/f'{stem}.{ext}').stat().st_size
                   for ext in ('svg','pdf','png')):
                continue
            fig, ax = plt.subplots(figsize=(5.2, 3.5))
            for role, part in rows.groupby('role', sort=True):
                part = part.sort_values('global_step')
                ax.plot(part.global_step, part[metric], marker='o', markersize=3,
                        label='Training' if role == 'source_train' else 'Selection validation')
            if metric == 'accuracy':
                ax.axhline(.95, linestyle=':', linewidth=1, label='Training fit target')
                ax.set_ylim(0, 1.02)
            ax.set(xlabel='Optimizer updates', ylabel=label, title=f'Source {source}')
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            for ext in ('svg','pdf','png'):
                fig.savefig(figures/f'{stem}.{ext}', dpi=300)
            plt.close(fig)
    (figures/'README.md').write_text(
        '# Source fitting curves\n\nSource: `../source_fit_curve.csv`.\n'
        'Training is resubstitution; validation participates in checkpoint selection. '
        'The 0.95 line is a user-specified fit target, not measured model performance. '
        'These plots are not industrial target-transfer results.\n'
        'Regenerate with `python -m scripts.tii_plot_fit95 --output RUN_DIR`.\n')


if __name__ == '__main__':
    main()
