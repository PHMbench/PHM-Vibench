# Paper 06 Neural-Symbolic Theory IEEE Transactions Readiness

Date: 2026-05-11

This checkpoint converts the Neural-Symbolic Theory evidence plan into a
machine-readable comparison, proposition, mapping, and ablation matrix. It does
not make the paper submission-ready.

## Current Evidence

- Matrix: `submission_prep/baseline_ablation_matrix.yaml`
- Manuscript checkpoint: `manuscript/final_tex/main.tex`
- Base config: `configs/vibench/min.yaml`
- Existing gate: `report/T045_evidence_readiness.md`
- Evidence level: six PHM-Vibench baseline dummy smokes plus paper-local
  proposition hooks, scripted mapping, source-backed sibling-submodule mapping,
  non-accepted mapping-ablation smoke hooks, and an evidence-bound IEEEtran
  manuscript checkpoint
- Compute policy: local RTX 4090 GPUs `0,1`; runnable PHM-Vibench commands bind
  `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1`

## Dummy-Smoke Summary

The proposed constrained NSN/TSPN_UXFD smoke and six baseline commands completed
in `LQ_signal` on dummy data with CPU fallback because the current environment
reported `GPU available: False` and `Can't initialize NVML`.

| ID | Role | Status |
|---|---|---|
| P00 | constrained NSN/TSPN_UXFD logic-slot model | pass; dummy only |
| B01/A01 | no-symbolic NSN/TSPN_UXFD | pass; dummy only |
| B02 | ResNet | pass; dummy only |
| B03 | SincNet | pass; dummy only |
| B04 | TFN | pass; dummy only |
| B05 | WKN | pass; dummy only |
| B06 | ConvTransformer | pass; dummy only |
| A03-A04 | logic `logit_scale` sweep | pass; dummy only |

## Proposition And Mapping Hooks

- `python simple_validation_demo.py` passes as a script but records P1/P3 as
  supported and P2 as failed in `results/theory_validation/validation_summary.json`.
- `python experiments/proposition2_simple.py` writes a synthetic P2 robustness
  artifact in `experiments/results/proposition2_12_14/simple_results.json`.
  This is a scope-limited positive synthetic hook. Per
  `submission_prep/p2_evidence_contract.md`, it does not override the P2
  boundary/failure outcome in the aggregate validation demo.
- `python code/validate_mapping.py` writes
  `report/mapping_validation_report.json` and
  `manuscript/figures/mapping_validation.png`; this scripted mapping is now
  paired with `python scripts/build_source_backed_mapping.py`, which writes
  `report/source_backed_mapping_report.json` and
  `report/source_backed_mapping_report.md` by checking sibling submodule
  `VIBENCH.md`, matrix, config, and source files. This is source-introspection
  evidence only and does not prove mapping impact.
- `python scripts/run_mapping_ablation_smoke.py --condition no_mapping` writes
  non-accepted `run_meta.yaml` and `metrics.json` smoke artifacts for the
  remove-mapping ablation surface. This does not prove train/eval impact.

## Manuscript Checkpoint

- `manuscript/final_tex/main.tex` now uses `IEEEtran`, has a concrete title,
  abstract, evidence-state table, blocked-claim section, and the existing
  `manuscript/figures/mapping_validation.png` figure.
- It no longer contains the placeholder title/body or the missing
  `../../figures/example.pdf` reference.
- Compile command from the submodule root:
  `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/uxfd_paper06_tex manuscript/final_tex/main.tex`
- Two `pdflatex` passes generated `/tmp/uxfd_paper06_tex/main.pdf`.
- This is a manuscript checkpoint only; final evidence-bearing tables and
  submission wording still require accepted run artifacts.

## Remaining Gaps

- Full CWRU/XJTU multi-seed baseline matrix with mean/std/95% CI.
- Real-data P1/P2/P3 validation with separate artifact IDs and failure cases.
- Real train/eval impact for the mapping module against 1D-2D, MoE, Fuzzy-XFD,
  Toolkit, LLM, and Operator Attention submodules.
- Complete strict local GPU metadata from devices `0,1`.
- TOP representative artifacts for TimeX++/SARAD/CFCBM/IFCBM or local faithful
  proxies under the 2x4090 budget.
- Final evidence-bearing manuscript expansion after accepted baseline,
  ablation, TOP representative, and GPU metadata artifacts exist.
- SOTA gate.

## Allowed Manuscript Wording

The manuscript may state that the repository now exposes runnable
neural-symbolic baseline, ablation, proposition, scripted mapping,
source-backed sibling-submodule mapping, and non-accepted mapping-ablation
smoke hooks, and an evidence-bound IEEEtran manuscript checkpoint. It must not
claim final proposition support, same-protocol superiority, TOP-method
reproduction, accepted mapping impact, GPU feasibility, or SOTA from this
checkpoint.
