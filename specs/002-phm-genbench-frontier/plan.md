# Implementation Plan: PHM-GenBench Frontier

**Branch**: `002-phm-genbench-frontier` | **Date**: 2026-05-10 | **Spec**: `specs/002-phm-genbench-frontier/spec.md`
**Input**: Feature specification from `specs/002-phm-genbench-frontier/spec.md`

## Summary

Upgrade the current PHM generative prototype into a paper-grade benchmark path by
first enforcing governance and evidence contracts, then integrating frontier
generative families through the existing factory architecture with exploratory
defaults.

## Technical Context

**Language/Version**: Python 3.10-compatible project code  
**Primary Dependencies**: PyTorch, PyTorch Lightning, pandas, pydantic, existing PHM-Vibench factories  
**Storage**: YAML configs, CSV metrics/tables, JSON manifests, PyTorch sample tensors  
**Testing**: `pytest`, `scripts.validate_docs`, `scripts.validate_configs`, CLI smoke commands  
**Target Platform**: Local Linux research environment with CPU smoke support and optional CUDA  
**Project Type**: configuration-first research benchmark repository  
**Performance Goals**: CPU smoke paths complete on dummy data; paper runs report NFE, wall-clock, throughput, and memory  
**Constraints**: no hidden fallback, no mandatory new compiled dependency, no parallel `src/phm_factory`, no test split as synthetic source  
**Scale/Scope**: full P0-P3 roadmap from governance through paperpack and multi-family generative baselines

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Maintains `python main.py --config <yaml>` as the public execution path.
- PASS: Preserves the 5-block config model.
- PASS: Uses `src/Pipeline_06_generative.py` and existing factories.
- PASS: Requires evidence-gated validity before benchmark claims.
- PASS: Keeps frontier methods exploratory until promoted.
- PASS: Separates PHM paperpack evidence from hidden fallback behavior.

## Project Structure

### Documentation (this feature)

```text
specs/002-phm-genbench-frontier/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── generative-benchmark-contract.md
├── checklists/
│   ├── requirements.md
│   └── benchmark-readiness.md
├── analysis/
│   └── m2-cross-artifact-analysis.md
├── m2/
│   ├── README.md
│   └── goals.md
├── reviews/
│   ├── claude-team/
│   └── codex/
├── handoffs/
├── paper/
└── tasks.md
```

### Source Code (repository root)

```text
main.py
configs/
├── base/model/
├── base/task/
├── demo/10_generative/
└── paper/phm_generative/
src/
├── Pipeline_06_generative.py
├── config_schema/
├── model_factory/generative_model/
└── task_factory/
    ├── task/generative/
    └── Components/generative/
scripts/
├── generative_sweep.py
└── paperpack_generative.py
test/
└── generative/
docs/
└── README.md
demos/
└── generative/
```

**Structure Decision**: Keep the existing PHM-Vibench factory layout. Add new
models/tasks/configs only through current registries and keep research demos
separate from benchmark-valid runtime claims.

**Documentation Decision**: Keep project-level docs as an index. Put
module-specific PHM generative guidance in the README next to the owning module
or config. Put process, review, handoff, and paper-readiness artifacts under the
active Speckit feature.

**Process Artifact Decision**: Keep `.specify/goals/v2/` as the goal-contract
queue. Store M2 development-process artifacts under
`specs/002-phm-genbench-frontier/`, including reviews, handoffs, validation
logs, cross-artifact analysis, and paper readiness notes. `.codex/` and
`.claude/` are tool scratch or mirrors only.

## Review And Handoff Artifacts

Claude Code Teams are advisory reviewers, not implementation owners or final
approval. Default mode is read-only `plan` or `review`; launch only after the
configured endpoint is approved for the scoped workspace content. A blocked
Claude review is evidence of non-execution, not independent approval.

Canonical M2 artifacts:

- Claude task spec:
  `reviews/claude-team/2026-05-11-phm-genbench-m2-six-dataset/TASK_SPEC.md`
- Claude status files:
  `reviews/claude-team/2026-05-11-phm-genbench-m2-six-dataset/`
- Codex verification: `reviews/codex/2026-05-11-m2-verification.md`
- Handoff: `handoffs/2026-05-11-m2-six-dataset.md`

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Core-fast experimental methods | User explicitly selected fast core integration for frontier families | Docs-only or demo-only would not satisfy the requested integration target |
| Multi-family roadmap | Top PHM paper target requires comparisons beyond CFM | A single CFM hardening PR is necessary but insufficient for the paper-level roadmap |

## Phase 0 Research

Research decisions are recorded in `research.md`. The accepted model ladder is:

1. Harden CFM infrastructure.
2. Promote Rectified Flow / FlowTS and DDPM / Diffusion-TS.
3. Add TimeFlow / Score-SDE and backbone variants.
4. Add one-step frontier methods as exploratory core candidates.

## Phase 1 Design

Design artifacts:

- `data-model.md`: entities and validation rules.
- `contracts/generative-benchmark-contract.md`: CLI/config/artifact contracts.
- `quickstart.md`: validation and paper evidence loop.

## Phase 2 Task Generation

Tasks are grouped by user story and PR-sized goals in `tasks.md`. P0 tasks block
all later model-family work.

## Post-Design Constitution Check

- PASS: No design artifact requires a parallel runtime.
- PASS: Evidence artifacts are mandatory for validity promotion.
- PASS: One-step frontier families remain exploratory by default.
- PASS: Test split usage remains eval-only with explicit opt-in.
