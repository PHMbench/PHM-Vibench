# PU D1 execution record — 2026-09-19

This is the curated aggregate record of one completed local PU D1 execution,
not a new experiment plan, a manuscript result section, or a claim of superiority.
All 30 source-training runs completed. The frozen evaluation exported all 25
declared test predictors. Negative contrasts, near-zero differences, zero
adoption, blocked FIRNet slots, and software-validation failures remain visible.
No D2/D3 experiment is represented here.

## Data, execution, and versions

The data were the existing PHM-Vibench `RM_027_PU` H5 and companion metadata.
The grouping unit was the physical bearing specimen, not a window, acquisition,
or bearing-condition pair: 32 bearings were partitioned globally into 15 source-fit,
5 source-selection, 5 independent-assessment, and 7 permanent-test bearings.
Source conditions were `0, 2, 3`; condition `1` was unseen. The same physical
bearing appearing under several conditions remains one group. Actual configuration
snapshots are in [configs/experiments/p01/d1_pu_20260919](../../../../configs/experiments/p01/d1_pu_20260919/).

| Execution boundary | Recorded revision |
| --- | --- |
| Original parent paper checkout | `0a8b7c2e25a649445e32500f60b5e5b12a32e01e` |
| Parent Benchmark gitlink at start | `376fc875533442de5879622e8aa82282acd75748` |
| Original TSPN reference `p0` training | `4f87b5b60ecc9bf13ebbfa632e690038d75b0742` |
| Core, diagnostic, and ResNet source training | `19ae593bca674ed021662ba92383cb467e583279` |
| F1 source freeze and F2 assessment freeze | `72874750be3e53cf8f78f27e1991ed4749c9a4f9` |
| Integration/CI snapshot before this evidence addition | `d12f65d02ac09123cc150f82d0291df5f08d3768` |

The 30 completed source runs comprise one original TSPN reference, 15 core runs
(MLP16/O/UO/RO/RC at seeds 42, 123, 456), 11 seed-42 source diagnostics, and three
ResNet1D runs. Each training invocation used 1,000 optimizer steps under the
declared recipe. The three FIRNet slots remained blocked: a complete method or
legal original implementation was unavailable; MCN/FIR-CCF were not substituted.
See [source_completion.json](source_completion.json),
[source_comparison_all_seeds.json](source_comparison_all_seeds.json), and
[firnet_status.json](firnet_status.json).

`source_completion.json` is a **pre-assessment source-completion snapshot**.
Its `assessment_or_test_accessed: false` describes that earlier time, not the
subsequent completed frozen assessment and test export. Source comparison checks
describe recorded schedules, reference vectors, and selection/access equality;
they are not performance conclusions.

The 25 frozen test predictors were the raw reference, 15 direct core predictors,
three direct ResNet predictors, the temperature candidate, four source-selected
candidates, and one independently adopted predictor. Source diagnostics were
not selected for additional test evaluation. Training used `LQ_signal` and
physical GPU0 (`cuda:0`, RTX 4090), without CPU-training fallback or GPU2 use.
[execution_environment.json](execution_environment.json) records software and
hardware details. Private original paths below
`/tmp/p01-d1-20260919-1irHPs/artifacts/d1_pu_20260919` are provenance references,
not portable public inputs or redistributed artifacts.
The complete private run was also backed up in the parent workspace's ignored
`results/d1_pu_20260919/`; the original private run directory remains retained.

## Saved outcomes and their limits

The following are rounded displays of the saved three-seed means in
[test/contrast_seed_summary.csv](test/contrast_seed_summary.csv). They refer to
the **permanent-test partition**, separately averaged over source conditions
and the unseen condition, not to source-validation performance. A negative
Brier difference favors the named treatment. Full-precision values, CE,
seed dispersion, and the saved intervals remain in the CSV files.

| Saved contrast | Treatment minus control | Source Brier difference | Unseen Brier difference |
| --- | --- | ---: | ---: |
| `Delta_pipe` | O − MLP16 | +0.05306736 | +0.15632025 |
| `Delta_loss` | RC − O | +0.04490826 | +0.18190057 |
| `Delta_prior` | RO − UO | +0.05786594 | +0.15336291 |
| `Delta_response` | RC − RO | +8.9138e-10 | −1.3870e-9 |

These recorded losses do not show an improvement from the first three
treatments; RC and RO have only numerical-scale differences here. `Delta_pipe`
is a processing-pipeline comparison, not an isolated causal representation
effect. Nothing in this package promotes a manuscript claim or establishes
generalization beyond this run.

[source_selection.csv](source_selection.csv) records seed-42 source selection:
the temperature candidate has `T=1.5, alpha=1`; MLP16/O/RC have `T=1, alpha=0`.
This is distinct from the independent assessment in [decision.json](decision.json):
the moments/Bernstein rule used `source_mixture`, four candidates, three source
conditions, `delta_total=0.05`, and `delta_shift=0`. There were only five independent
assessment groups per condition, not 15 independent specimens across conditions.
Every candidate received `alpha=0` and `accepted_nonzero=false`. The selected
temperature candidate therefore returned the frozen reference through zero
mixing; it did **not** deploy the `T=1.5, alpha=1` source-selected correction.
The recorded zero upper excess is the identity-reference outcome, not evidence
of a useful nonzero correction, empirical coverage, or an unseen-condition
risk guarantee. The decision retains its conditional-assessment status and
assumptions rather than claiming an unconditional deployment guarantee.

Losses average windows within acquisitions, acquisitions within physical groups,
groups within conditions, and then the named population's conditions equally.
Saved 95% intervals are descriptive paired global-physical-group bootstrap
intervals: 2,000 draws, analysis seed 20260919, condition-incidence stratification,
and shared resampling across arms and training seeds. They are conditional on
the frozen predictors and unstable with few groups. Training seeds are not
independent specimens. The separate ResNet `baseline_seed_summary.csv` files
contain finite three-seed means and sample SD only: `lower`/`upper` are blank and
`interval_status=not_computed`. No interval was made by averaging endpoints.
Figures are diagnostic previews, not publication-validated figures; their
captions preserve these statistical boundaries.

## Labels, time, memory, and latency

[costs/label_budget.csv](costs/label_budget.csv) counts each shared label pool
once, not once per model, seed, view, or repeated window visit:

| Pool | Global physical groups | Acquisitions / labels | Windows |
| --- | ---: | ---: | ---: |
| Source fit | 15 | 900 | 1,800 |
| Source selection | 5 | 300 | 600 |
| Independent assessment | 5 | 300 | 600 |
| Permanent test | 7 | 559 | 1,118 |
| Distinct accessed union | 32 | 2,059 | 4,118 |

[costs/notes.json](costs/notes.json) records 30/30 completed source runs and
`missing_artifacts=[]`; this does not mean every timer was available.
[costs/cost.csv](costs/cost.csv) retains stage-level blanks and measurement
status. For example, `p0` training took 11.573788720939774 s, but its process
and source-materialization timers were not recorded; its filesystem-observed
interval is not a replacement monotonic-process timer. The saved shared-stage
times were 107.19613257999299 s for source freeze/selection,
28.108324697008356 s for assessment/adoption, and 165.10304714599624 s for the
test-export invocation. Process, filesystem, and component intervals overlap
and must not be summed into an invented total; blank cost fields are not zero.

[costs/latency.csv](costs/latency.csv) preserves all saved timing and parameter
values while removing **only** `group_id`, `acquisition_id`, and `window_id`.
Those three identifiers remain in the private original table. Each measurement
used one source-validation input, 20 warmups, and 100 timing repetitions, with
input already on GPU and model construction, checkpoint loading, and data I/O
excluded. Repetitions are not independent specimens. For example, the adopted
zero-correction path has saved median 1.0066469840239733 ms and IQR
0.16423172201029956 ms. GPU0 was shared with other projects during parts of
execution; these are observed shared-device costs, not exclusive-device speed
claims. Executed, stored, and loaded parameter counts have different meanings;
resident-reference memory must not be attributed to a direct ResNet alone.

The training feature/mechanism summaries average recorded batches along changing
optimization states. They are not frozen-population estimates or independent
assessment evidence. Native feature units are not energy-matched; summaries of
within-batch medians are not pooled-window medians. No waveforms, checkpoints,
prediction arrays, or latency measurements were reopened or rerun to prepare
this public package.

## Software checks and retained failures

Local focused implementation checks passed (98 tests before the final analysis
extension; 34 final analysis/plot/cost tests in 8.48 s at research revision
`72874750be3e53cf8f78f27e1991ed4749c9a4f9`; 11 source-comparison tests in the
source worktree). These scopes overlap and are not summed as unique tests.
Docs and maintained public-config checks passed. The one maintained-suite
attempt, with one explicitly excluded out-of-scope real-DIRG test, failed during
collection because `streamlit` was absent: exit 2, no suite tests executed.
No dependency was installed and no test was altered to hide that failure.
The exact status/stdout/stderr/exit status are retained under
[failures/maintained_suite/](failures/maintained_suite/). This is not a full-suite
pass. A separate earlier preflight was blocked by missing complete reference
history; its failure and initial decision are retained under
[failures/preflight_initial/](failures/preflight_initial/), not presented as the
final frozen decision.

[PR #270](https://github.com/PHMbench/PHM-Vibench/pull/270) had 11/11 successful
CI checks at implementation HEAD `d12f65d02ac09123cc150f82d0291df5f08d3768`.
That snapshot does not claim CI coverage of this subsequent evidence addition,
completion of the locally blocked suite, or scientific validation.

## Public contents

Only aggregate tables, diagnostic plots/captions, environment/execution summaries,
and the two stated failure records are included. No physical-bearing,
acquisition, or window identifiers, raw metadata/protocol, private development
histories, per-sample prediction arrays, NPZ/PT/H5 files, or checkpoints are
distributed. Path strings referencing private artifacts are retained solely as
provenance. Apart from the explicit three-column latency redaction and harmless
terminal newlines in two JSON files, copied results retain the original content.
CSV line endings are normalized to LF and generated SVG line-end spaces removed
for Git; field values, SVG coordinates, and PNG/PDF files are unchanged.
The complete file list is:

```text
README.md
decision.json
execution_environment.json
firnet_status.json
source_comparison_all_seeds.json
source_completion.json
source_selection.csv
costs/
  cost.csv
  label_budget.csv
  latency.csv
  notes.json
  training_features_summary.csv
  training_mechanism_summary.csv
failures/
  preflight_initial/
    decision.md
    failure.json
  maintained_suite/
    exit_status.txt
    status.json
    stderr.log
    stdout.log
source_validation/
  contrast_seed_summary.csv
  mechanism.csv
  metrics.csv
  paired_contrasts.csv
  seed_summary.csv
  plots/
    CAPTIONS.md
    main_contrasts.pdf
    main_contrasts.png
    main_contrasts.svg
    seed_dispersion.pdf
    seed_dispersion.png
    seed_dispersion.svg
source_frozen/
  baseline_seed_summary.csv
  contrast_seed_summary.csv
  mechanism.csv
  metrics.csv
  paired_contrasts.csv
  seed_summary.csv
test/
  baseline_seed_summary.csv
  contrast_seed_summary.csv
  mechanism.csv
  metrics.csv
  paired_contrasts.csv
  seed_summary.csv
  plots/
    CAPTIONS.md
    adoption_mechanism.pdf
    adoption_mechanism.png
    adoption_mechanism.svg
    main_contrasts.pdf
    main_contrasts.png
    main_contrasts.svg
    seed_dispersion.pdf
    seed_dispersion.png
    seed_dispersion.svg
```
