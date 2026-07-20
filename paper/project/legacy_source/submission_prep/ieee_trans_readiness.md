# IEEE Transactions Readiness Evidence

Status date: 2026-05-11

This note is the strict-reviewer gate for the current TII operator-attention
package. It records runnable evidence and blocks any unsupported submission or
SOTA claim.

## Canonical Package

- Canonical manuscript entrypoint: `manuscript/final_tex/main.tex`
- Source manuscript consumed by canonical entrypoint: `bare_jrnl_new_sample4.tex`
- Existing PDF: `bare_jrnl_new_sample4.pdf`
- Parent-facing reproduction contract: `VIBENCH.md`
- Minimal PHM-Vibench config: `configs/vibench/min.yaml`
- Main synthetic validation entrypoint: `python code/synthetic_verification.py`
- Command-bound baseline/ablation matrix:
  `submission_prep/baseline_ablation_matrix.yaml`
- Rejection-recovery evidence contract:
  `submission_prep/rejection_recovery_contract.md`
- Reviewer traceability matrix:
  `submission_prep/reviewer_traceability_matrix.md`

## Verified Commands

Synthetic operator-selection validation:

```bash
python code/synthetic_verification.py
```

Result on 2026-05-11:

- Exit code: 0
- Signal classes covered: 8
- Mean physics consistency: 0.999
- Mean explainability score: 0.261
- Output figures:
  - `figures/synthetic_signals.png`
  - `figures/operator_weights_heatmap.png`
  - `figures/explainability_comparison.png`
- Output data:
  - `results/synthetic_validation_results.json`
  - `doc/synthetic_verification_report.md`

Minimal PHM-Vibench smoke run:

```bash
eval "$(conda shell.bash hook)"
conda activate LQ_signal
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override trainer.num_epochs=1
```

Result on 2026-05-11:

- Exit code: 0 in `LQ_signal`
- Base environment blocker: `ModuleNotFoundError("No module named 'pytorch_lightning'")`
- Runtime device: CPU fallback; `CUDA_VISIBLE_DEVICES=0` was set, but PyTorch reported GPU unavailable in this sandbox
- Dummy-data output root:
  `results/uxfd/pilot/TII_operator_attention/metadata_dummy.csv/M_NSN/`
- Representative test metric from the run:
  `test_loss=0.7221537828445435`, `test_acc_Dummy_Data=0.0`

The smoke run proves the entrypoint wiring only. It does not prove industrial
accuracy, SOTA performance, or submission readiness.

Canonical TeX compile gate:

```bash
pdflatex -interaction=nonstopmode manuscript/final_tex/main.tex
bibtex main
pdflatex -interaction=nonstopmode manuscript/final_tex/main.tex
pdflatex -interaction=nonstopmode manuscript/final_tex/main.tex
```

Result on 2026-05-11:

- Exit code: 0 for all four commands
- Output: `main.pdf`
- Final pass: no undefined citation/reference warnings observed
- Remaining warnings: routine layout warnings and IEEEtran language
  hyphenation warnings; no `empty year` BibTeX warnings remained after adding
  explicit `year` fields for the affected entries.
- Scope: compileability of the normalized IEEE entrypoint only; the manuscript
  still needs evidence-table updates before submission.

## Current Gate Status

| Gate | Status | Evidence or blocker |
|---|---|---|
| Canonical TeX entrypoint | partial pass | `manuscript/final_tex/main.tex` compiles from the submodule root and consumes `bare_jrnl_new_sample4.tex`; evidence tables still require command-backed updates before submission. |
| Synthetic validation | partial pass | Script runs, writes submodule-local artifacts, and covers 8 signal classes; this satisfies the synthetic signal-count gate but remains simulated operator-selection evidence only. |
| Industrial runnable entrypoint | partial | `configs/vibench/min.yaml` runs as dummy-data smoke in `LQ_signal`; no industrial protocol proof yet. |
| 6+ baseline suite | command-bound with dummy-smoke pass | `submission_prep/baseline_ablation_matrix.yaml` declares seven baseline commands and all seven dummy smokes pass in `LQ_signal`; same-protocol industrial runs/artifacts are still missing. |
| Ablation suite | command-bound with dummy-smoke pass | `submission_prep/baseline_ablation_matrix.yaml` declares six ablation commands and all six dummy smokes pass in `LQ_signal`; same-protocol industrial artifacts are still missing. |
| TOP recent-work representative | blocked | Parent goal requires TimeMixer, SARAD, CATCH, DADA, PGRFNet, GTM, CSLSTM, and TSPulse representatives; only the parent queue GTM proxy is mapped, and no accepted same-protocol command/log/artifact package exists yet. |
| SOTA diagnosis claim | blocked | No same-protocol industrial-data evidence beats the declared baselines. Use theory/interpretable-mechanism wording only. |
| Rejection-recovery trace | partial | `revision/review_response_plan.md`, `submission_prep/rejection_recovery_contract.md`, and `submission_prep/reviewer_traceability_matrix.md` exist; they still need command-backed accepted evidence links. |

## Next Execution Milestone

1. Run the command-bound baseline and ablation matrix on the accepted
   industrial protocol with local GPU metadata and multi-seed statistics.
2. Bind the TOP recent-work representatives to local commands or explicitly
   mark exact reproduction as `resource-blocked` under the two RTX 4090 budget.
3. Update the normalized manuscript package after the evidence tables are
   generated.
