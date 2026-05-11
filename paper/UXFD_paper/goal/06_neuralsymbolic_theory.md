# Paper Goal: Neural-Symbolic Theory

## Target

- Submodule: `paper/UXFD_paper/Neuralsymbolic_theory`
- Default journal: IEEE Transactions on Neural Networks and Learning Systems
- Alternate journal: IEEE Transactions on Artificial Intelligence
- Contribution: a formal neural-symbolic framework with verifiable propositions and cross-method mappings for explainable fault diagnosis.

## Canonical Package

- Manuscript entrypoint: `manuscript/final_tex/main.tex`
- Reproduction contract: `VIBENCH.md` and `configs/vibench/min.yaml`
- Current compile blocker: placeholder TeX content and missing `../../figures/example.pdf`.

## Required Evidence

- Proposition-level validation: each major proposition has a script, command, and artifact.
- Cross-method mapping for 1D-2D, MoE, and Fuzzy mechanisms.
- Boundary-condition or counterexample reporting where propositions fail.
- CWRU and XJTU evidence where claims extend to data behavior.

## Baseline Suite

- `ISFM.M_01_ISFM`.
- `X_model.NSN` or `TSPN_UXFD` without neural-symbolic constraints.
- `CNN.ResNet1D`.
- `Transformer.PatchTST` or `Transformer.ConvTransformer`.
- `X_model.Sincnet`.
- `X_model.TFN`.
- Unconstrained neural model versus symbolic-constraint variant.

## Ablation Suite

- Remove symbolic constraints.
- Remove physical-consistency constraint.
- Remove cross-method mapping module.
- Validate each proposition independently.
- Vary constraint strength and report boundary conditions.
- Compare neural-only, symbolic-only, and neural-symbolic variants.

## SOTA Optimization Gate

- The Neuralsymbolic paper may claim SOTA trustworthiness or constrained diagnosis only if it beats all declared baselines on the stated diagnostic and consistency metrics.
- If the main contribution is theoretical unification rather than raw accuracy, the manuscript must state the exact metric axis where it is SOTA or avoid SOTA language.

## TOP Recent-Work Quota

- RWTOP2024-TIMEXPP: `representative-runnable` time-series explanation baseline for faithfulness metrics.
- RWTOP2024-SARAD: `representative-runnable` association-based diagnosis baseline for symbolic mapping.
- RWTOP2025-CFCBM: `literature-only` counterfactual concept baseline until symbolic concepts are defined.
- RWTOP2025-IFCBM: `literature-only` concept-bottleneck comparator for theory positioning.

## Compute Budget

- Available devices: local RTX 4090 GPUs `0,1` only; commands must bind `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` when GPU is used.
- Default execution: proposition validation and neural-symbolic variants should run as one-GPU or CPU jobs; at most two GPU-backed jobs may run concurrently.
- Runtime tier: exact TOP concept/counterfactual baselines are `resource-blocked` if concept labels, model scale, or intervention search exceed the 2x4090 budget.
- Required metadata: device ID when used, GPU model, seed, batch size, precision, runtime, constraint strength, and any OOM/failure reason must be captured with accepted artifacts.

## Strict-Reviewer Risks

- A theory paper cannot rely on conceptual diagrams without formal definitions and runnable validation.
- Cross-method mapping must be executable, not only narrative.
- Failed proposition cases should be framed as boundary conditions, not hidden.

## Acceptance Gates

- Canonical TeX compiles with proposition, proof, and validation artifacts linked.
- Each proposition has verified or blocked evidence status.
- Cross-method mapping artifacts are accepted.
- At least six baselines and the neural-symbolic ablations above are present in accepted artifacts.
- Any SOTA trustworthiness or constrained-diagnosis claim satisfies the SOTA optimization gate.
- Submodule-local commit records the accepted paper package milestone.
