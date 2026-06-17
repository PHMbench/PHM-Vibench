# 01. Paper-Readiness Gap Analysis

## Goal

Support a comprehensive PHM generative benchmark paper, not just a runnable
demo.  A paper-grade repository needs:

```text
real datasets
strict protocol
multiple baselines
reproducible train/sample/eval stages
quality/utility/leakage/efficiency metrics
manifested evidence
paperpack
submission-readiness gate
```

## Current readiness score

| Area | Status | Score |
|---|---|---:|
| Entry path and preflight | Mostly ready | 8/10 |
| Generative pipeline | Usable, train/sample/eval only | 7/10 |
| CFM baseline | Usable | 7/10 |
| Rectified Flow baseline | Thin wrapper, needs tests | 5/10 |
| DDPM baseline | Runtime exists, needs path tests | 5/10 |
| Score SDE | Exploratory | 3/10 |
| One-step methods | Exploratory placeholders | 2/10 |
| UNet/DiT backbones | Smoke-ready, not paper-proven | 5/10 |
| Manifest chain | Strong sample manifest, weak stage linkage | 6/10 |
| Metrics | Good smoke bundle, weak utility protocol | 5/10 |
| Six-dataset matrix | Strong plan, needs real-run evidence | 6/10 |
| Paperpack | Exists, needs artifact linkage | 6/10 |
| Submission draft | Conservative and useful | 7/10 |

Overall: **paper-infrastructure 65/100, paper-evidence 35/100**.

## Minimum paper claims the repo can safely support today

Safe:

```text
- PHM-Vibench now has a config-first generative benchmark scaffold.
- CFM, Rectified Flow, and DDPM can be represented as factory tasks.
- Synthetic outputs are evidence-gated and default to exploratory.
- Paperpack/draft scripts refuse to invent missing evidence.
```

Unsafe until real runs:

```text
- "method X outperforms method Y"
- "benchmark is submission-ready"
- "six datasets completed"
- "MeanFlow / Drifting are implemented faithfully"
- "TSTR improvement is validated by downstream training"
```

## Promotion gates for the paper

A method/dataset/seed row is paper-eligible only if:

```text
1. train run completed.
2. sample run completed with checkpoint, not untrained smoke.
3. sample payload contains samples + fault_label + domain_id.
4. synthetic_data_manifest exists.
5. normalization params are recorded.
6. protocol hash and config hash are recorded.
7. dependency lock hash is recorded.
8. leakage check passed or recorded as failed with row excluded.
9. eval metrics exist with status/reason fields.
10. method/dataset/seed row appears in benchmark_effect_summary.csv.
11. paperpack has source paths back to metrics and manifest.
```

## What v0.3 changes relative to v0.2

v0.2 proposed the paper-ready architecture.  v0.3 assumes much of it already
exists and focuses on stabilization:

```text
v0.2: "build the scaffolding"
v0.3: "freeze, link, validate, and promote only evidence-backed rows"
```
