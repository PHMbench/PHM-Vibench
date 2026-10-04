"""Make two claim-led figures from actual run tables, with editable text."""
import argparse
import csv
import json
from pathlib import Path
from .summarize import load_predictions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-diagnostic', action='store_true')
    a = p.parse_args(argv)
    declared = json.loads((a.run / 'config.json').read_text(encoding='utf-8'))
    methods = tuple(declared.get('methods', ()))
    config, _, context, _, _ = load_predictions(a.run, methods, a.allow_diagnostic)
    synthetic = config['data_kind'] == 'synthetic'
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('use a new, empty figure directory')
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    kind = (f"Diagnostic: {config['data_kind']}, {context['evaluation_role']} observations"
            if context['diagnostic'] else 'Held-out test observations')
    def save(fig, name):
        for extension in ('svg', 'pdf', 'png'):
            fig.savefig(a.output / f'{name}.{extension}', dpi=300)
        plt.close(fig)
    curves = read_rows(a.run / 'risk_coverage.csv')
    # Mechanism question: does the certificate improve on the identical raw rule score?
    fig, ax = plt.subplots(figsize=(4.4, 3.2), layout='constrained')
    for method, line in (('fuzzy_rule_score', '--'), ('joint', '-')):
        seeds = sorted({r['seed'] for r in curves if r['method'] == method}, key=int)
        for seed in seeds:
            rows = [r for r in curves if r['method'] == method and r['seed'] == seed and r['risk']]
            rows.sort(key=lambda r: float(r['coverage']))
            ax.plot([float(r['coverage']) for r in rows], [float(r['risk']) for r in rows],
                    linestyle=line, marker='.', linewidth=1, label=f'{method}, seed {seed}')
    ax.axhline(config['alpha'], linestyle=':', linewidth=.8, label='Risk target')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Unit-balanced coverage',
           ylabel='Accepted error rate', title=kind)
    ax.legend(fontsize=6, loc='upper right')
    save(fig, 'risk_coverage_mechanism')
    # Compare every method, paired by seed and by attainable empirical coverage.
    rows = read_rows(a.run / 'matched_coverage.csv')
    lookup = {(r['method'], r['seed'], r['target_coverage']): r for r in rows}
    methods = list(dict.fromkeys(r['method'] for r in rows if r['method'] != 'joint'))
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, .29 * len(methods) + 1)), layout='constrained')
    unavailable = 0
    for target, marker, offset in ((.5, 'o', -.15), (.7, 's', 0.), (.9, '^', .15)):
        xx, yy = [], []
        for index, method in enumerate(methods):
            for r in rows:
                if r['method'] != method or float(r['target_coverage']) != target:
                    continue
                joint = lookup.get(('joint', r['seed'], r['target_coverage']))
                if not r['risk'] or not joint or not joint['risk']:
                    unavailable += 1
                    continue
                xx.append(float(r['risk']) - float(joint['risk']))
                yy.append(index + offset)
        ax.scatter(xx, yy, marker=marker, s=18, label=f'Coverage {target:.1f}')
    ax.axvline(0, linestyle='--', linewidth=.8)
    ax.set_yticks(range(len(methods)), methods)
    ax.set(xlabel='Risk difference: control minus joint (each point is one seed)', title=kind)
    ax.legend(fontsize=7, loc='best')
    save(fig, 'paired_risk_all_controls')
    undefined = sum(not r['risk'] for r in curves)
    (a.output / 'figure_notes.md').write_text(
        f'# Figure interpretation\n\n{kind}.\n\n'
        'The mechanism curve compares joint certification with its identical-model rule score. '
        'The paired plot includes every control and all defined seed/coverage comparisons. '
        f'{undefined} zero-coverage curve rows have undefined risk; '
        f'{unavailable} paired comparisons are unattainable and are not extrapolated.\n\n'
        'Matched coverage uses label-blind fractional boundary acceptance, not deployment thresholds. '
        'Seeds are algorithmic repetitions, not independent physical specimens; no seed-based '
        'inferential confidence intervals are drawn. Positive differences favor joint.\n\n'
        'SVG text is editable. Both figures are single panels; multipanel alignment is inapplicable. '
        'Final manuscript composition requires rendered collision inspection at publication size.\n', encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
