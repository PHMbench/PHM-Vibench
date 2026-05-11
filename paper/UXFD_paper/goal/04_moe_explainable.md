# Paper Goal: MOE Explainable Fault Diagnosis

## Target

- Submodule: `paper/UXFD_paper/MOE_explainable`
- Default journal: IEEE Transactions on Neural Networks and Learning Systems
- Alternate journal: IEEE Transactions on Industrial Informatics
- Contribution: physics-constrained experts and auditable routing for stable, explainable fault diagnosis.

## Canonical Package

- Manuscript entrypoint: `manuscript/final_tex/main.tex`
- Reproduction contract: `VIBENCH.md` and `configs/vibench/min.yaml`
- Current compile status: prior Slice 4 compile gate passed for the final TeX entrypoint.

## Required Evidence

- Multi-seed CWRU and XJTU performance with confidence intervals.
- Expert-count ablation for at least 3, 5, and 8 experts or an explicitly justified reduced matrix.
- Route entropy, path signature, and expert activation distribution.
- Stability analysis with CV and failure-mode explanation if CV exceeds threshold.

## Baseline Suite

- `ISFM.M_01_ISFM`.
- `X_model.NSN` or `TSPN_UXFD` without MoE.
- `CNN.ResNet1D`.
- `CNN.TCN`.
- `X_model.Sincnet`.
- `X_model.TFN`.
- Dense ensemble or equal-weight experts without learned routing.

## Ablation Suite

- Experts removed and replaced with a single backbone.
- Learned router replaced by uniform routing.
- Statistical/physics features removed from router input.
- Expert-count sweep for 3, 5, and 8 experts.
- Route sparsity or entropy regularization removed.
- Path-signature explanation module removed.

## SOTA Optimization Gate

- The optimized MoE must beat all declared baselines on the primary diagnostic metric while also improving or matching route stability before SOTA is claimed.
- If accuracy wins but routing is unstable, the paper may claim performance improvement but not stable explainable MoE.

## TOP Recent-Work Quota

- RWTOP2025-TIMEMOE: `representative-runnable` sparse MoE foundation baseline for scaling and route efficiency.
- RWTOP2025-MOIRAIMOE: `representative-runnable` token-level sparse expert baseline for automatic specialization.
- RWTOP2024-MOMENT: `representative-runnable` dense/foundation representation comparator.
- RWTOP2024-TIMEXPP: `representative-runnable` explanation-quality comparator for route interpretations.

## Compute Budget

- Available devices: local RTX 4090 GPUs `0,1` only; commands must bind `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` by default.
- Default execution: one GPU per MoE seed/config; expert-count sweeps must be queued rather than assuming more than two GPUs.
- Runtime tier: exact Time-MoE/Moirai-MoE-scale reproduction is `resource-blocked` unless it fits `CUDA_VISIBLE_DEVICES=0,1`; local MoE proxies must record activated parameters and routing artifacts.
- Required metadata: device ID, GPU model, seed, batch size, precision, runtime, expert count, activated experts, and any OOM/failure reason must be captured with accepted artifacts.

## Strict-Reviewer Risks

- MoE performance variance can undermine the contribution if route stability is weak.
- Expert interpretability must be shown through route artifacts, not only architecture diagrams.
- Claims of physical homology need evidence tying experts to signal features or fault mechanisms.

## Acceptance Gates

- Canonical TeX compiles and all route/evidence tables are populated.
- Expert ablation artifacts are accepted or blocked with reasons.
- Route-level claims map to generated artifacts.
- At least six baselines and the MoE ablations above are present in accepted artifacts.
- Any SOTA MoE claim satisfies the SOTA optimization gate.
- Submodule-local commit records the accepted paper package milestone.
