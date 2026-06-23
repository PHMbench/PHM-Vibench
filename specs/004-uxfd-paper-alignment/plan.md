# Implementation Plan: UXFD Paper Alignment

**Branch**: `004-uxfd-paper-alignment` | **Date**: 2026-05-10 | **Spec**: `specs/004-uxfd-paper-alignment/spec.md`
**Input**: Feature specification from `specs/004-uxfd-paper-alignment/spec.md`

## Summary

Align the seven UXFD paper submodules with PHM-Vibench evidence contracts: audit
`VIBENCH.md` plus minimal configs, run or explicitly skip minimal evidence gates,
map LaTeX figure/table/result claims to artifacts or blockers, and compile selected
entrypoints only after the real TeX entrypoints and toolchain are known.

This slice is a paper-evidence and submission-readiness slice. It must not move
paper-specific scripts into core code, silently verify unsupported claims, or record
parent gitlink changes without a submodule-local commit.

## Technical Context

**Language/Version**: Python 3.x for root CLI gates; LaTeX toolchain as available in the environment
**Primary Dependencies**: Existing PHM-Vibench runtime dependencies, shell tools, optional TeX tools discovered during implementation
**Storage**: UXFD submodule files, YAML configs, paper artifacts, LaTeX sources/logs, filesystem run artifacts
**Testing**: Focused pytest/docs checks, root CLI smoke commands, selected LaTeX compile commands
**Target Platform**: Local Linux research workstation / CI-compatible shell where possible
**Project Type**: Parent benchmark repo with paper submodules
**Performance Goals**: Evidence gates should fail with the exact submodule/config/claim blocker before paper claims are marked verified
**Constraints**: Do not recursively read large paper results or ignored thesis workspaces; keep paper-specific changes inside submodules; no silent paper-claim verification; no accidental parent gitlink changes
**Scale/Scope**: Seven UXFD submodules, their `VIBENCH.md` files, `configs/vibench/min.yaml` files, selected LaTeX entrypoints, and named artifacts only

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Config-first contract is preserved for parent-facing UXFD minimal configs.
- PASS: Factory and registry wiring remains in core; paper-specific assets stay in
  submodules.
- PASS: Fail-fast behavior is explicit for missing configs, artifacts, entrypoints,
  toolchains, and unsupported claims.
- PASS: Evidence-backed reproducibility is the main purpose of this slice.
- PASS: Minimal correct change is enforced by auditing real entrypoints and blockers
  instead of inventing paper assets.

Post-design re-check:

- PASS: `research.md`, `data-model.md`, `contracts/uxfd-paper-alignment-contract.md`,
  and `quickstart.md` keep the same constraints and do not introduce broad paper or
  core-runtime refactors.

## Project Structure

### Documentation (this feature)

```text
specs/004-uxfd-paper-alignment/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── uxfd-paper-alignment-contract.md
└── checklists/
    └── requirements.md
```

### Source Code And Paper Assets (repository root)

```text
paper/README.md
paper/README_SUBMODULE.md
paper/UXFD_paper/README.md
paper/UXFD_paper/*/VIBENCH.md
paper/UXFD_paper/*/configs/vibench/min.yaml
paper/UXFD_paper/*/manuscript/
paper/UXFD_paper/*/paper_draft/
paper/UXFD_paper/results/
scripts/
test/
```

**Structure Decision**: keep parent changes to indexes/spec evidence unless a paper
submodule itself owns the edited file. Do not create parent-level paper mapping docs
that duplicate each submodule's `VIBENCH.md`.

## Phase Plan

### Phase 0: Research

Resolve current behavior from source of truth:

- UXFD family index and submodule rules: `paper/UXFD_paper/README.md` and
  `paper/README_SUBMODULE.md`;
- submodule reproduction contracts: each `VIBENCH.md` and `configs/vibench/min.yaml`;
- LaTeX entrypoint discovery: actual `main.tex` or paper-specific TeX files;
- artifact expectations: VIBENCH docs, Slice 1 artifact contract, and named result
  artifacts only;
- compile tooling: available TeX commands discovered during implementation.

Output: `research.md`.

### Phase 1: Design And Contracts

Define:

- data model for UXFD Submodule Contract, Minimal Evidence Gate, LaTeX Entry Point,
  Claim Evidence Link, Compile Gate, and Submodule Pointer State in `data-model.md`;
- UXFD contract, evidence, claim, compile, and submodule-pointer rules in
  `contracts/uxfd-paper-alignment-contract.md`;
- validation quickstart in `quickstart.md`;
- AGENTS context pointer to this plan.

### Phase 2: Task Generation

Generate tasks that first audit contracts and existing evidence, then patch only
uncovered gaps. Expected task groups:

- submodule contract and min-config inventory checks;
- minimal root CLI evidence gates and artifact recording;
- claim-to-evidence mapping for selected LaTeX entrypoints;
- compile-gate discovery and execution;
- submodule dirty-state and parent gitlink safety checks.

## Complexity Tracking

No constitution violations are planned.
