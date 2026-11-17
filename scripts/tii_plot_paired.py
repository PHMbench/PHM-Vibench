"""Plot saved paired-group contrasts and matched clean/intervention NLLs.

The inputs are computed CSVs; this command does not load data or models, choose
hyperparameters, recompute uncertainty, or change the supplied populations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--clean', type=Path, required=True)
    parser.add_argument('--interventions', type=Path, required=True)
    parser.add_argument('--intervention', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()

    paired = pd.read_csv(args.paired)
    clean = pd.read_csv(args.clean)
    altered = pd.read_csv(args.interventions)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                      gridspec_kw={'width_ratios': [1.0, 1.18]})
    choices = [('single', 'raw'), ('joint', 'raw'),
               ('single', 'source_rms'), ('joint', 'source_rms')]
    colors = {'raw': '#276c92', 'source_rms': '#ac542d'}
    for y, (scope, mode) in enumerate(reversed(choices)):
        row = paired[(paired['contrast'] == scope + '_S_minus_U') & (paired['mode'] == mode)]
        if len(row) != 1:
            raise ValueError('one precomputed paired interval per requested contrast required')
        p = row.iloc[0]
        if not np.isfinite([p['estimate'], p['ci95_low'], p['ci95_high']]).all() or p['ci95_low'] > p['ci95_high']:
            raise ValueError('finite ordered paired interval required')
        left.plot([p['ci95_low'], p['ci95_high']], [y, y], color=colors[mode], linewidth=1.2)
        left.plot(p['estimate'], y, color=colors[mode], marker='o', markersize=4)
    left.axvline(0., color='0.55', linewidth=.8, zorder=0)
    left.set_yticks(range(4), ['Joint / feature RMS', 'Single / feature RMS',
                               'Joint / raw', 'Single / raw'])
    left.set_xlabel('Paired S − U query NLL (nat)')
    left.set_title('a  Token organization', loc='left', weight='bold')
    left.grid(axis='x', color='0.9', linewidth=.6)

    names = [('S', 'S', '#276c92'), ('U', 'U', '#ac542d'),
             ('N', 'Ordinary-N', '#39815c'), ('HSE', 'Native-HSE', '#83549a'),
             ('envelope', 'Envelope-Linear', '#595959')]
    for offset, (model, label, color) in enumerate(names):
        c = clean[(clean['scope'] == 'joint') & (clean['model'] == model)
                  & (clean['mode'] == 'source_rms')].sort_values('seed')
        v = altered[(altered['scope'] == 'joint') & (altered['model'] == model)
                    & (altered['mode'] == 'source_rms') & (altered['variant'] == args.intervention)].sort_values('seed')
        if len(c) == 0 or len(c) != len(v) or not np.array_equal(c['seed'].to_numpy(), v['seed'].to_numpy()):
            raise ValueError(f'clean and intervention seeds do not match for {model}')
        a, b = c['nll'].to_numpy(dtype=float), v['nll'].to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(), b.min()) <= 0:
            raise ValueError('log-scale NLL points must be finite and positive')
        x = np.array([0., 1.]) + (offset - 2) * .028
        right.plot(x, [a.mean(), b.mean()], color=color, marker='o', markersize=3.5,
                   linewidth=1.1, label=label)
        right.scatter(np.full(len(a), x[0]), a, color=color, s=9, alpha=.35, linewidth=0)
        right.scatter(np.full(len(b), x[1]), b, color=color, s=9, alpha=.35, linewidth=0)
    right.set_yscale('log')
    right.set_xticks([0, 1], ['Clean', 'Resonance +15%'])
    right.set_ylabel('Query NLL (nat; log scale)')
    right.set_title('b  Joint-source frequency shift', loc='left', weight='bold')
    right.grid(axis='y', which='major', color='0.9', linewidth=.6)
    right.legend(frameon=False, fontsize=7, loc='center left', bbox_to_anchor=(1.01, .5))
    fig.tight_layout(pad=.8, w_pad=1.7)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(args.output_prefix.with_suffix('.' + extension), dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
