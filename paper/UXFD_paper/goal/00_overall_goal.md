# Overall Goal: Seven IEEE Transactions UXFD Submissions

## Objective

Prepare the seven UXFD paper submodules as independent IEEE Transactions
submission packages that can withstand a strict reviewer: every claim must be
traceable, every result must be reproducible or explicitly blocked, and every
manuscript must compile from a canonical entrypoint.

## Papers

| Paper | Submodule | Default Target | Alternate |
|---|---|---|---|
| Explainable FD Toolkit | `Explainable_FD_Toolkit` | IEEE TII | IEEE TAI |
| 1D-2D Fusion | `1D-2D_fusion_explainable` | IEEE TII | IEEE TIE / Information Fusion |
| LLM Explainable FD Toolkit | `LLM_Explainable_FD_Toolkit` | IEEE TII | IEEE THMS |
| MOE Explainable | `MOE_explainable` | IEEE TNNLS | IEEE TII |
| Fuzzy-XFD | `Paper_fuzzy_XFD` | IEEE TFS | IEEE TII |
| Neuralsymbolic Theory | `Neuralsymbolic_theory` | IEEE TNNLS | IEEE TAI |
| TII Operator Attention | `TII_operator_attention` | IEEE TSP | IEEE TIE/TII |

## Common Submission Contract

Each paper must provide:

- Canonical manuscript entrypoint and compile command.
- IEEE Transactions target and template decision.
- Claim-to-evidence map for figures, tables, metrics, baselines, and major text claims.
- Reproduction command set rooted in `VIBENCH.md` and `configs/vibench/min.yaml`.
- Evidence package with commands, configs, logs, artifacts, and accepted result tables.
- At least six fair baselines under the same dataset split, seed protocol, metric definitions, and preprocessing.
- A top-venue recent-work citation map covering 2024-2026 and identifying which recent works are exact-runnable, representative-runnable, literature-only, or blocked.
- Paper-specific ablation studies that remove or vary the exact claimed innovation.
- Statistical protocol for multi-seed, confidence interval, and ablation reporting.
- Compute budget declaration for every accepted run, using only the local GPUs `0,1`.
- SOTA optimization gate: the proposed method is optimized toward SOTA, but SOTA may be claimed only after same-protocol evidence beats every declared baseline.
- Limitations and failure-case section that does not hide known weak points.
- Submodule-local commit for every accepted paper-specific milestone.

## Baseline, Ablation, And SOTA Gate

Every paper must declare a baseline suite and ablation suite before manuscript claims are accepted.

- Baseline minimum: six methods per paper, with at least three strong diagnostic model baselines, at least two recent or competitive architecture baselines, and at least one interpretability or paper-specific baseline.
- Registry-backed baseline candidates include `ISFM.M_01_ISFM`, `X_model.NSN`/`TSPN_UXFD`, `CNN.ResNet1D`, `X_model.Resnet`, `X_model.Sincnet`, `X_model.TFN`, `X_model.WKN`, `CNN.TCN`, `Transformer.PatchTST`, and `Transformer.ConvTransformer`.
- Recent-work baselines must be drawn from the accepted TOP method pool in `08_recent_work_citation_readme.md` and must carry an exact-run or representative-run status.
- Each paper's six-baseline suite must include at least two accepted TOP conference/journal methods or faithful PHM-Vibench representatives of those methods.
- Fairness rule: every baseline table must use the same datasets, splits, seeds, preprocessing, metric definitions, and report format.
- Ablation rule: each ablation must remove or vary the contribution claimed by that paper, not an unrelated hyperparameter.
- Claim rule: if the optimized method does not beat the declared baselines under the accepted protocol, the paper must report the result honestly and revise the contribution framing instead of claiming SOTA.

## Top-Venue Recent-Work Gate

The seven papers must use current, high-quality methods for related work,
baselines, and SOTA positioning.

- Accepted sources: NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV, KDD, AAAI, IJCAI, ACL/EMNLP/NAACL, SIGIR, WWW, IEEE TPAMI, IEEE TNNLS, IEEE TKDE, IEEE TCYB, IEEE TFS, IEEE TII, IEEE TIE, IEEE TSP, Information Fusion, Mechanical Systems and Signal Processing, and Pattern Recognition.
- Excluded sources for core claims: Scientific Reports, publisher-level MDPI journals, IEEE Transactions on Instrumentation and Measurement, IEEE Access, Applied Sciences, Electronics, Sensors, Mathematics, and similar low-tier or application-only venues.
- Excluded sources may be mentioned only as intentionally rejected context; they must not appear in the accepted method pool, baseline tables, or SOTA comparisons.
- Each paper must map at least three 2024-2026 TOP-source methods to its contribution and at least one must be `exact-runnable` or `representative-runnable` before submission.
- Paper 07 must rebuild its novelty and rejection-recovery argument from TOP-source operator, attention, anomaly, and time-series methods, not from low-tier bearing-fault-diagnosis papers.

## Compute Resource Gate

The only available accelerator resources are local GPUs `0` and `1`, both RTX
4090-class cards. No goal, baseline, ablation, or SOTA plan may assume cloud
GPUs, A100/H100 hardware, multi-node execution, or more than two GPUs unless a
later human-approved resource change updates this goal package.

- Required device declaration: every runnable command must record `CUDA_VISIBLE_DEVICES=0,1` or the single selected device `CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=1`.
- Default scheduling policy: one GPU per experiment and at most two concurrent single-GPU jobs.
- Multi-GPU policy: a run may use both GPUs only when the command explicitly binds `CUDA_VISIBLE_DEVICES=0,1` and records why two GPUs are needed.
- Artifact metadata: every accepted experiment must record device IDs, GPU model, GPU count, seed, batch size, precision, runtime, and any OOM or resource failure.
- Feasibility rule: TOP methods that exceed the 2x4090 budget must be labelled `resource-blocked` for exact reproduction and may count only as `representative-runnable` if a local faithful proxy is run.
- Claim rule: SOTA wording is blocked if the winning comparison depends on a baseline that cannot be run or represented under the 2x4090 budget.

## Strict-Reviewer Rubric

The paper is not submission-ready if any of these are true:

- A numerical claim lacks an accepted artifact or documented external source.
- A figure/table exists but its generation path is unknown.
- Fewer than six baselines are declared or the baseline comparison lacks matching data, seed, split, or metric protocol.
- Recent work from the last two years is missing, uncited, not from an accepted TOP source, or counted as a reproduced baseline without a runnable command/log.
- A low-tier source is used to establish novelty, baseline strength, or SOTA positioning.
- A run, baseline, or SOTA claim assumes compute beyond local GPUs `0,1` without an explicit approved resource change.
- A large TOP method is counted as exact-runnable even though exact reproduction is `resource-blocked` under the 2x4090 budget.
- Ablation coverage does not test the claimed innovation.
- A SOTA claim is written before same-protocol evidence proves it.
- The manuscript contains placeholders, old paths, unverified TODO claims, or synthetic-only claims
  promoted as final real-data evidence.
- TeX fails to compile or contains unresolved references/citations that change interpretation.
- Submodule changes are dirty but not committed inside the owning paper repo.

## Claude Code Team Usage

Use Claude Code Team for parallel read-only review before major writing or evidence milestones.
Codex remains lead-of-record and must verify the reports before accepting them.

Default team:

- Evidence Auditor: claim-to-artifact and result provenance.
- LaTeX Submission Auditor: canonical TeX, figures, references, and IEEE compliance.
- Method Reviewer: Toolkit, 1D-2D, MoE, and Fuzzy method sufficiency.
- Theory/Application Reviewer: Neuralsymbolic, Operator Attention, and LLM evidence.
- Strict Reviewer #3: adversarial review for overclaiming and weak experiments.

Run directory:

- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/`

## Handoff Usage

Every milestone handoff must record:

- Completed papers and current blockers.
- Submodule commit SHA for accepted paper-specific changes.
- Parent gitlink intent.
- Commands run and validation status.
- Next paper-specific tasks.

Current handoff:

- `.claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md`
