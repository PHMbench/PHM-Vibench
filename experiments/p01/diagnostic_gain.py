"""Competence and paired diagnostic-gain reports from saved probabilities.

Development qualification is not a population guarantee. Confirmation bounds
require frozen functions and independent groups under the declared sampling law.
No predictor, dataset, coefficient or decision is selected by this report.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.p01.fusion_assessment import grouped_moments, radius


def report(raw, candidates, labels, groups, acquisitions, domains, *, stage,
           independent_groups=False, competence=.8, minimum_gain=0.,
           failure=.05, method='hoeffding'):
    """Condition-wise accuracy and Brier on the same independent physical groups.

    K includes every reported candidate/seed, not just the best-looking one.
    Allocate failure across D*(1+2K) events: baseline accuracy, paired accuracy
    gain and paired Brier excess. This allocation is for this report, not for
    other datasets or an existing adoption procedure.
    """
    if stage not in {'development', 'confirmation'}:
        raise ValueError('Declare development or confirmation explicitly.')
    if stage == 'confirmation' and independent_groups is not True:
        raise ValueError('Confirmation requires the independent-group sampling declaration.')
    if not (math.isfinite(competence) and .8 <= competence <= 1 and
            math.isfinite(minimum_gain) and 0 <= minimum_gain < 1 and
            math.isfinite(failure) and 0 < failure < 1):
        raise ValueError('Competence must be at least .8; declare a finite gain and error budget.')
    if method not in {'hoeffding', 'bernstein'}:
        raise ValueError('Predeclare one radius family; no outcome-dependent minimum.')
    moments = grouped_moments(raw, candidates, labels, groups, acquisitions, domains)
    p, q = np.asarray(raw, float), np.asarray(candidates, float)
    if q.ndim == 2:
        q = q[None]
    y = np.asarray(labels)
    buckets = defaultdict(list)
    for i, key in enumerate(zip(domains, groups, acquisitions)):
        buckets[key].append(i)
    per_group = defaultdict(list)
    for (domain, group, _), rows in sorted(buckets.items()):
        if len(set(y[rows].tolist())) != 1:
            raise ValueError('An acquisition must have one class before probability averaging.')
        target = int(y[rows[0]])
        predictions = np.r_[p[rows].mean(0).argmax(), q[:, rows].mean(1).argmax(-1)]
        cm = np.zeros((len(q)+1, p.shape[1], p.shape[1]))
        cm[np.arange(len(q)+1), target, predictions] = 1
        per_group[(str(domain), str(group))].append(cm)
    K, D = len(q), len(moments)
    u = failure / (D * (1 + 2*K))
    output = []
    for domain, (b, a) in moments.items():
        part = [np.mean(per_group[key], axis=0) for key in sorted(per_group) if key[0] == domain]
        cm = np.stack(part)
        # Physical group weighting agrees with grouped_moments' sorted groups.
        accuracies = np.trace(cm, axis1=2, axis2=3)
        gain = accuracies[:, 1:] - accuracies[:, :1]
        delta = a - 2*b
        mean_cm = cm.mean(0)
        diagonal = np.diagonal(mean_cm, axis1=1, axis2=2)
        den = mean_cm.sum(1) + mean_cm.sum(2)
        macro = np.divide(2*diagonal, den, out=np.zeros_like(diagonal), where=den != 0).mean(1)
        support = sorted(set(y[np.asarray(domains) == domain].tolist()))
        all_classes = support == list(range(p.shape[1]))
        n = len(b)
        base = float(accuracies[:, 0].mean())
        row = dict(domain=domain, independent_groups=n, class_support=support,
                   baseline_accuracy=base, baseline_macro_f1=float(macro[0]),
                   development_competence_pass=bool(all_classes and base >= competence), candidates=[])
        if stage == 'confirmation' and n < 2:
            raise ValueError('The declared radius needs at least two independent groups per condition.')
        base_lower = None
        if stage == 'confirmation':
            base_lower = base - float(radius(accuracies[:, 0].var(ddof=1), n, 1., u, method))
        row['baseline_accuracy_lower'] = base_lower
        for k in range(K):
            g, d = float(gain[:, k].mean()), float(delta[:, k].mean())
            lower, upper = None, None
            if stage == 'confirmation':
                lower = g - float(radius(gain[:, k].var(ddof=1), n, 2., u, method))
                upper = d + float(radius(delta[:, k].var(ddof=1), n, 4., u, method))
            row['candidates'].append(dict(candidate=k, accuracy=float(accuracies[:, k+1].mean()),
                macro_f1=float(macro[k+1]), accuracy_gain=g, brier_excess=d,
                accuracy_gain_lower=lower, brier_excess_upper=upper,
                gain_supported=bool(stage == 'confirmation' and all_classes and
                                    base_lower >= competence and lower > minimum_gain and upper <= 0)))
        output.append(row)
    return dict(stage=stage, competence_threshold=competence, minimum_accuracy_gain=minimum_gain,
        radius_method=method if stage == 'confirmation' else None,
        failure_budget=failure if stage == 'confirmation' else None,
        one_sided_event_budget=u if stage == 'confirmation' else None,
        candidate_count=K, condition_count=D, by_domain=output,
        development_competence_pass=all(row['development_competence_pass'] for row in output),
        all_candidate_gains_supported=bool(stage == 'confirmation' and all(
            c['gain_supported'] for row in output for c in row['candidates'])),
        scope='Conditional on frozen predictors, independent within-condition groups and declared evaluation population. No arbitrary unseen-domain guarantee; development scores are selection-conditional.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, nargs='+', help='Aligned saved raw/candidate NPZs; include every prespecified candidate/seed.')
    parser.add_argument('--stage', required=True, choices=['development', 'confirmation'])
    parser.add_argument('--independent-groups', action='store_true')
    parser.add_argument('--competence', type=float, default=.8)
    parser.add_argument('--minimum-gain', type=float, default=0.)
    parser.add_argument('--failure-budget', type=float, default=.05)
    parser.add_argument('--method', choices=['hoeffding', 'bernstein'], default='hoeffding')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # Read stored arrays only; no H5, checkpoint, model or plotting invocation.
    arrays = []
    for path in args.predictions:
        with np.load(path, allow_pickle=False) as saved:
            arrays.append({key: saved[key] for key in (
                'raw_probs','candidate_probs','labels','group_ids','acquisition_ids','domains',
                'window_ids','raw_class_names','candidate_class_names')})
    first = arrays[0]
    names=first['raw_class_names']
    if names.ndim!=1 or names.dtype.kind not in 'US' or len(names)!=first['raw_probs'].shape[1] or len(set(names))!=len(names):
        raise ValueError('Unique ordered class names must match the probability columns.')
    if not np.array_equal(first['raw_class_names'], first['candidate_class_names']):
        raise ValueError('Raw and candidate class order differ.')
    for other in arrays:
        for key in ('labels','group_ids','acquisition_ids','domains','window_ids','raw_class_names','candidate_class_names'):
            if not np.array_equal(first[key], other[key]):
                raise ValueError(f'Aligned observation/class identity required: {key}.')
        if not np.allclose(first['raw_probs'], other['raw_probs'], atol=1e-8, rtol=1e-7):
            raise ValueError('Reported candidates do not share the same reference probabilities.')
    identities = list(zip(*(first[key].tolist() for key in ('domains','group_ids','acquisition_ids','window_ids'))))
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated window identity; windows cannot be counted twice.')
    result = report(first['raw_probs'], np.stack([p['candidate_probs'] for p in arrays]), first['labels'],
        first['group_ids'], first['acquisition_ids'], first['domains'], stage=args.stage,
        independent_groups=args.independent_groups, competence=args.competence,
        minimum_gain=args.minimum_gain, failure=args.failure_budget, method=args.method)
    result['prediction_files'] = [str(Path(path).resolve()) for path in args.predictions]
    with Path(args.output).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(args.output)


if __name__ == '__main__':
    main()
