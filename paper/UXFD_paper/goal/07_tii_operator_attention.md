# Paper Goal: TII Operator Attention

## Target

- Submodule: `paper/UXFD_paper/TII_operator_attention`
- Default journal: IEEE Transactions on Signal Processing
- Alternate journals: IEEE Transactions on Industrial Electronics or IEEE Transactions on Industrial Informatics
- Contribution: SOTA-capable operator-attention diagnosis with rigorous operator-level interpretability and repaired theory-experiment linkage.

## Canonical Package

- Canonical manuscript entrypoint: `manuscript/final_tex/main.tex`
- Source manuscript consumed by canonical entrypoint: `bare_jrnl_new_sample4.tex`
- Reproduction contract: `VIBENCH.md` and `configs/vibench/min.yaml`
- Current manuscript status: normalized IEEE entrypoint compiles; final
  evidence-table updates, accepted industrial artifacts, TOP representative
  evidence, GPU metadata, and SOTA gate remain blocked.

## Required Evidence

- Formal operator-space definitions and assumptions.
- Theorem/proposition proof appendix with precise limitations.
- Synthetic signal validation for operator selection behavior.
- Industrial-data experiments across the same datasets, splits, seeds, and metrics as all declared baselines.
- Operator explanation metrics: OAS, OSS, and OCS.

## Baseline Suite

- `X_model.NSN` or `TSPN_UXFD` without operator attention.
- `CNN.ResNet1D`.
- `X_model.Sincnet`.
- `X_model.TFN`.
- `X_model.WKN`.
- `Transformer.PatchTST` or `Transformer.ConvTransformer`.
- Standard feature/self-attention variant using the same backbone.

## Ablation Suite

- Remove operator attention.
- Remove sparse/L1 operator selection.
- Remove physics-consistency regularization.
- Operator subset sweep: identity, FFT, HT, wavelet, and combined operators.
- Operator attention versus feature/self-attention.
- Sensitivity to sparsity weight, operator count, and attention temperature.

## SOTA Optimization Gate

- Paper 07 may claim SOTA diagnostic performance only if the optimized operator-attention model beats all declared baselines on the primary metric under the accepted industrial-data protocol.
- If performance remains below strong baselines, SOTA language is blocked and the paper must reposition to theory/interpretable-mechanism contribution with explicit limitations.
- Any interpretability SOTA claim must beat the declared explanation baselines on OAS, OSS, OCS, faithfulness, stability, or an explicitly defined operator-level metric.

## Rejection-Recovery Focus

- Repair prior rejection risks: weak performance, unclear innovation, missing recent/SOTA baselines, shallow ablations, and theory-experiment mismatch.
- Innovation upgrade target: Dynamic Sparse Operator Attention v2 with learnable operator selection and physics-consistency regularization.
- The revised contribution must explicitly distinguish operator attention from standard feature attention and transformer attention.
- The manuscript must include a reviewer-response style trace from each major prior concern to new evidence or a scoped limitation.

## TOP Recent-Work Quota

- RWTOP2024-TIMEMIXER: `representative-runnable` multiscale temporal baseline for operator-space decomposition.
- RWTOP2024-SARAD: `representative-runnable` spatial/association diagnosis baseline for operator explanations.
- RWTOP2025-CATCH: `representative-runnable` frequency/channel baseline for operator-attention comparison.
- RWTOP2025-DADA: `representative-runnable` adaptive bottleneck anomaly baseline for rejection-recovery SOTA positioning.
- RWTOP2026-PGRFNET: `representative-runnable` prototype/relational diagnostic comparator for operator evidence.
- RWTOP2026-GTM: `representative-runnable` frequency-attention representation comparator for operator selection.
- RWTOP2026-CSLSTM: `representative-runnable` contextual/seasonal anomaly comparator for signal behavior.
- RWTOP2026-TSPULSE: `representative-runnable` compact pretrained comparator under the 2x4090 budget.

## Compute Budget

- Available devices: local RTX 4090 GPUs `0,1` only; commands must bind `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` by default.
- Default execution: one GPU per operator-attention seed/config; at most two concurrent jobs across baseline, ablation, and sensitivity sweeps.
- Runtime tier: rejection-recovery evidence must be feasible on 2x4090; exact TOP baselines that exceed this budget are `resource-blocked` and cannot support exact SOTA claims.
- Required metadata: device ID, GPU model, seed, batch size, precision, runtime, operator count, attention temperature, sparsity weight, and any OOM/failure reason must be captured with accepted artifacts.

## Strict-Reviewer Risks

- Prior reviewer concerns targeted weak theoretical substantiation and weak industrial performance.
- The paper must not claim SOTA diagnosis accuracy unless industrial-data evidence beats all declared baselines.
- Theory-to-experiment linkage must be explicit for every major claim.
- The innovation will still look incremental unless DSOA v2, physics consistency, and operator metrics are shown together.

## Acceptance Gates

- Canonical TeX entrypoint is selected and compiles.
- Theory claims map to proofs, synthetic validation, or blocked status.
- Industrial performance claims satisfy the SOTA optimization gate or are scoped conservatively.
- At least six baselines, all required ablations, and the rejection-recovery trace are present in accepted artifacts.
- Submodule-local commit records the accepted paper package milestone.

## Stage-2 Follow-up Task Binding

- Evidence task: `P07-A`.
- Queue step: `Q1`; this remains the first paper queue because it is the
  rejected-paper recovery target.
- Dependencies: `T02`, `T03`, `T04`, `T05`, and `T06`.
- Accepted evidence required: industrial same-protocol proposed model results,
  at least seven accepted baselines, at least six accepted ablations,
  TOP-Q1-GTM representative evidence, local 2x4090 metadata, and
  reviewer-traceable rejection-recovery artifacts.
- Verification:
  `python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage`
  and `python -m scripts.uxfd_sota_gate --format markdown`.

Paper07 may be optimized toward SOTA, but SOTA wording is allowed only after
matched-seed accepted aggregate evidence beats every declared baseline and
runnable TOP representative. Otherwise the contribution must be reframed as a
bounded operator-attention or interpretability improvement.
