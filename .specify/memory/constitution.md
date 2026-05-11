<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- Template Principle 1 -> I. Configuration-First Execution
- Template Principle 2 -> II. Factory-First Runtime Boundaries
- Template Principle 3 -> III. Evidence-Gated Benchmark Validity
- Template Principle 4 -> IV. PHM Data Leakage And Signal Integrity
- Template Principle 5 -> V. Minimal, Auditable Change Sets
Added sections:
- Generative Benchmark Constraints
- Development Workflow And Quality Gates
Removed sections:
- Template placeholder sections
Templates requiring updates:
- ✅ reviewed: .specify/templates/plan-template.md; generated feature plan includes PHM gates
- ✅ reviewed: .specify/templates/spec-template.md; generated feature spec includes evidence requirements
- ✅ reviewed: .specify/templates/tasks-template.md; generated feature tasks include goal-sized PHM gates
Follow-up TODOs:
- None
-->

# PHM-GenBench Constitution

## Core Principles

### I. Configuration-First Execution

Every maintained experiment MUST be runnable through:

```bash
python main.py --config <yaml>
```

Configs are the experiment contract. Maintained configs MUST preserve the
5-block structure: `environment`, `data`, `model`, `task`, and `trainer`.
Pipeline selection MUST come from the top-level YAML `pipeline:` field and MUST
be checked against an explicit whitelist. Malformed configs, missing required
blocks, illegal pipeline names, and unresolved presets MUST fail fast.

### II. Factory-First Runtime Boundaries

Runtime extensions MUST use the existing factory architecture:

- data loading under `src/data_factory/`
- models and backbones under `src/model_factory/`
- tasks, losses, samplers, metrics, and manifests under `src/task_factory/`
- trainer wiring under `src/trainer_factory/`

Generative benchmark code MUST stay in the established locations:

- `src/Pipeline_06_generative.py`
- `src/model_factory/generative_model/`
- `src/task_factory/task/generative/`
- `src/task_factory/Components/generative/`

New code MUST NOT create a parallel `src/phm_factory/` runtime or bypass the
existing registry/factory contracts.

### III. Evidence-Gated Benchmark Validity

Synthetic outputs MUST be labeled as exactly one of:

- `benchmark-valid`
- `exploratory`
- `docs-only`

`benchmark-valid` MUST require complete evidence: config hash, protocol hash,
normalization artifact and hash, condition counts, source split, leakage checks,
and metric status/reason reporting. If any required evidence is missing, the run
MUST fail explicitly or be visibly downgraded to `exploratory`; hidden fallback
logic is not allowed.

### IV. PHM Data Leakage And Signal Integrity

Generative training and synthetic-source manifests MUST NOT use `val`, `valid`,
`validation`, `test`, or `target_test` as source splits. Test references may be
used only in explicitly marked evaluation mode with a visible opt-in flag.

FFT, envelope, spectral, leakage, and downstream utility metrics are eval-only
evidence in the V0/V1 generative benchmark. They MUST NOT be silently introduced
as training losses unless a later goal explicitly defines and validates that
objective.

### V. Minimal, Auditable Change Sets

One goal SHOULD map to one reviewable PR. Each goal MUST state objective, scope,
out-of-scope items, required behavior, acceptance criteria, and validation
commands. Changes SHOULD be small, deterministic, and traceable to the stated
goal. New dependencies MUST be optional or clearly justified by a promotion
goal.

## Generative Benchmark Constraints

Core runtime methods MAY include CFM, Rectified Flow or FlowTS-style baselines,
DDPM or Diffusion-TS-style baselines, Score-SDE or TimeFlow-style baselines, and
backbones such as UNet1D, DiT-style 1D transformers, and stateless Mamba/SSM
adapters.

Frontier one-step or experimental methods, including MeanFlow, improved
MeanFlow, Drifting, Transition Flow Matching, and ODE-free Neural Flow Matching,
MAY enter core only behind explicit experimental labeling. They MUST default to
`validity_status: exploratory` and MUST NOT be benchmark-valid until they pass
the same evidence gates as established baselines.

Mamba and SSM modules are backbones, not losses. Sampling steps MUST be
stateless unless a later goal introduces state handling with tests and manifest
evidence.

Official generative benchmark documentation lives under `docs/phm_generative/`.
Research-only demos may live under `demos/generative/`, but they MUST declare
that their outputs are not benchmark-valid.

## Development Workflow And Quality Gates

Speckit artifacts SHOULD be produced in this order:

1. Constitution
2. Specification
3. Clarification
4. Implementation plan
5. Requirements-quality checklist
6. Tasks
7. Cross-artifact analysis
8. Implementation

Before implementation, the active feature MUST have a `spec.md`, `plan.md`, and
`tasks.md`. Checklist and analysis findings that identify critical constitution
or evidence-gate violations MUST be resolved before `/speckit-implement`.

Minimum validation gates for meaningful generative benchmark changes are:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m pytest test/
```

If `--preflight-only` is not yet implemented, the goal that introduces it MUST
define temporary validation commands and then replace them with the public
preflight commands once available.

## Governance

This constitution supersedes conflicting generative benchmark practices in
plans, goals, prompts, and implementation notes. Amendments require a Speckit
constitution update, a version bump, and a sync impact report.

Versioning policy:

- MAJOR: backward-incompatible governance changes or principle removals.
- MINOR: new principles, new mandatory evidence gates, or materially expanded
  workflow requirements.
- PATCH: wording clarifications that do not change obligations.

Every PR or goal that touches generative benchmark runtime, configs, metrics, or
paperpack artifacts MUST review compliance with this constitution.

**Version**: 1.0.0 | **Ratified**: 2026-05-10 | **Last Amended**: 2026-05-10
