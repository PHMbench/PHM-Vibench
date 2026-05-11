# Contract: UXFD IEEE Transactions Submission Readiness

## Scope

This contract governs parent-level readiness coordination for seven independent UXFD papers. It does not authorize manuscript rewrites, experiment execution, or submodule commits by itself.

## Required Parent Artifacts

The following files must exist:

- `paper/UXFD_paper/goal/README.md`
- `paper/UXFD_paper/goal/00_overall_goal.md`
- `paper/UXFD_paper/goal/01_explainable_fd_toolkit.md`
- `paper/UXFD_paper/goal/02_1d2d_fusion.md`
- `paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md`
- `paper/UXFD_paper/goal/04_moe_explainable.md`
- `paper/UXFD_paper/goal/05_fuzzy_xfd.md`
- `paper/UXFD_paper/goal/06_neuralsymbolic_theory.md`
- `paper/UXFD_paper/goal/07_tii_operator_attention.md`
- `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`
- `paper/UXFD_paper/goal/99_submission_readiness_matrix.md`

## Paper Readiness Rules

Each paper goal file must include:

- target and alternate IEEE Transactions journal;
- contribution statement;
- canonical manuscript entrypoint or blocker;
- required evidence package;
- at least six fair baselines;
- TOP recent-work quota with at least three accepted 2024-2026 TOP-source methods;
- compute budget restricted to local RTX 4090 GPUs `0,1`;
- contribution-specific ablation suite;
- SOTA optimization gate and claim policy;
- strict-reviewer risks;
- acceptance gates;
- submodule commit rule.

## TOP Recent Work Contract

- The goal package must include a recent related-work README covering 2024-2026 TOP-source works relevant to UXFD papers.
- Accepted core sources are TOP journals and computer-science top conferences only.
- Scientific Reports, publisher-level MDPI journals, IEEE Transactions on Instrumentation and Measurement, IEEE Access, Applied Sciences, Electronics, Sensors, Mathematics, and similar low-tier sources must not enter the accepted method pool, baseline tables, or SOTA comparisons.
- Each accepted recent work must include venue tier, venue, URL, UXFD relevance, reproduction status, and representative run policy.
- Each recent work must be labelled as `exact-runnable`, `representative-runnable`, `literature-only`, or `blocked`.
- Literature-only and blocked works may support related-work writing but must not count as reproduced performance baselines.
- Representative-runnable works must be labelled as representative and tied to a local PHM-Vibench command before being used in comparison tables.
- Exact-runnable works require command, config, log, and artifact paths before SOTA comparison.
- Each paper must have at least one exact-runnable or representative-runnable TOP-source method before submission.
- TOP methods that exceed local GPUs `0,1` must be marked `resource-blocked` for exact reproduction.

## Compute Budget Contract

- Available accelerators are only local RTX 4090 GPUs `0` and `1`.
- Commands must record `CUDA_VISIBLE_DEVICES=0`, `CUDA_VISIBLE_DEVICES=1`, or `CUDA_VISIBLE_DEVICES=0,1`.
- Default scheduling is one GPU per experiment and at most two concurrent single-GPU jobs.
- No claim may assume cloud GPUs, A100/H100 hardware, multi-node execution, or more than two GPUs unless a later approved resource update changes the contract.
- Accepted artifacts must record device IDs, GPU model, GPU count, seed, batch size, precision, runtime, and OOM/failure reason if any.

## Evidence Rules

- Minimal `configs/vibench/min.yaml` root runs are necessary but not sufficient for submission readiness.
- Claims are accepted only when mapped to generated artifacts or documented external sources.
- Missing evidence, missing entrypoints, compile failures, and placeholders remain blockers.
- Fewer than six baselines blocks performance claims.
- Missing ablations block innovation claims.
- SOTA wording is blocked unless same-protocol evidence beats all declared baselines.
- Low-tier literature blocks novelty, baseline-strength, and SOTA claims when used as core evidence.
- Compute-infeasible exact reproduction blocks exact baseline and exact SOTA claims.

## Claude Code Team Rules

- Default mode is read-only review/plan.
- The task spec must include roles, target paths, out-of-scope actions, acceptance checks, and report paths.
- Claude teammates must not push, deploy, publish, delete, read secrets, or edit files unless a later implementation phase explicitly partitions ownership.
- Codex must verify Claude reports before accepting findings.

## Handoff Rules

- Every major milestone must produce a handoff under `.claude/handoffs/`.
- Handoff must state current feature, changed files, decisions, blockers, validation status, and next actions.
- Paper-specific milestone handoffs must include submodule commit SHA and parent gitlink intent.
