# Feature Specification: UXFD Paper Alignment

**Feature Branch**: `004-uxfd-paper-alignment`
**Created**: 2026-05-10
**Status**: Draft
**Input**: User description: "Slice 4 from `.specify/goals/phm-vibench-full-phm-experiment-platform.md`: align UXFD submodule contracts, experiment evidence, and LaTeX claims."

## Clarifications

### Session 2026-05-10

- Q: Which paper files are in scope for final LaTeX alignment? -> A: Planning must
  discover actual LaTeX entrypoints in the seven UXFD submodules; missing or
  non-final entrypoints are documented as blockers rather than invented.
- Q: How should submodule edits be handled? -> A: Paper-specific changes stay
  inside the owning submodule and require a submodule-local commit before the
  parent records any gitlink pointer change.

## User Scenarios & Testing

### User Story 1 - Audit UXFD Reproduction Contracts (Priority: P1)

A parent-repo maintainer can audit the seven UXFD submodules and see whether each
has a parent-facing `VIBENCH.md`, a minimal config, a maintained smoke command, and
expected artifacts.

**Why this priority**: Paper alignment cannot start from claims; it must start from
reproducible submodule contracts.

**Independent Test**: Inspect the UXFD family index and submodule contracts; every
submodule has a status for `VIBENCH.md`, `configs/vibench/min.yaml`, smoke command,
and expected artifacts.

**Acceptance Scenarios**:

1. **Given** the UXFD family index, **When** the maintainer audits contracts, **Then**
   each of the seven submodules has a status and blocker reason if any required
   contract is missing.
2. **Given** a submodule `VIBENCH.md`, **When** its maintained command is checked,
   **Then** the command uses the root `python main.py --config ...` contract or is
   recorded as paper-local only.

---

### User Story 2 - Run Minimal UXFD Evidence Gates (Priority: P1)

A maintainer can run or explicitly skip each UXFD minimal config and record whether
it produces the expected Slice 1 artifacts plus any paper-specific artifacts.

**Why this priority**: Figure and table claims must trace to actual run evidence or
known blockers.

**Independent Test**: Run selected minimal configs through the maintained root CLI
where feasible and record pass/fail/skipped status with artifact paths or blocker
reasons.

**Acceptance Scenarios**:

1. **Given** a submodule minimal config, **When** the smoke command runs, **Then** it
   completes through the root CLI and records required runtime artifacts.
2. **Given** a submodule minimal config that cannot run, **When** the gate is
   evaluated, **Then** the blocker identifies the missing config, dependency, data,
   or paper-local script requirement.

---

### User Story 3 - Align LaTeX Claims With Evidence (Priority: P1)

A paper author can trace each selected UXFD figure, table, result claim, and
baseline statement in the final LaTeX entrypoint to generated artifacts or a
documented unresolved blocker.

**Why this priority**: Submission-ready paper text must not contain unsupported
claims or placeholder figures/tables.

**Independent Test**: Build a claim-to-evidence map for selected LaTeX entrypoints;
every claim has an artifact, external source, or blocker status.

**Acceptance Scenarios**:

1. **Given** a final LaTeX entrypoint, **When** figure and table references are
   audited, **Then** each referenced artifact exists or has a blocker.
2. **Given** a performance or baseline claim, **When** evidence is checked, **Then**
   it maps to a run artifact, config, command, or unresolved blocker.

---

### User Story 4 - Compile Selected Paper Entrypoints (Priority: P2)

A paper maintainer can compile the selected UXFD LaTeX entrypoints or receive
actionable compile blockers.

**Why this priority**: A paper can be evidence-aligned but still not submission-ready
if LaTeX entrypoints fail to compile.

**Independent Test**: Run the planned compilation command for each selected
entrypoint and record pass/fail/skipped status with log paths and the first
actionable error.

**Acceptance Scenarios**:

1. **Given** a selected LaTeX entrypoint with available bibliography tooling, **When**
   compilation runs, **Then** the PDF is produced without fatal errors or the log
   names the first actionable blocker.
2. **Given** no final entrypoint or missing TeX toolchain, **When** the compile gate
   is evaluated, **Then** the skip reason and impact are recorded.

### Edge Cases

- UXFD submodule has `VIBENCH.md` but no runnable root CLI command.
- UXFD submodule has minimal config but it depends on private data or missing
  optional dependencies.
- Paper README contains historical paths or old CLI flags that conflict with
  `VIBENCH.md`.
- Submodule worktree is dirty before alignment starts.
- LaTeX entrypoint exists outside the expected `manuscript/final_tex/main.tex`
  pattern.
- A figure or table file exists but cannot be traced to a run artifact.
- A result claim references a baseline from Slice 3 that is blocked or unverified.
- TeX toolchain or bibliography command is unavailable.
- Parent repo gitlink changes are present without a corresponding submodule commit.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST audit all seven UXFD submodules listed in the UXFD family index for `VIBENCH.md`, `configs/vibench/min.yaml`, maintained smoke command, expected artifacts, and status.
- **FR-002**: UXFD minimal configs MUST use the maintained root runtime contract or be explicitly labeled paper-local/unverified.
- **FR-003**: Each UXFD submodule MUST have an evidence status: smoke-passed, blocked, skipped, paper-local-only, or unverified.
- **FR-004**: Minimal evidence gates MUST record actual commands, pass/fail/skipped status, artifact paths, and blocker reasons.
- **FR-005**: Paper-specific configs, scripts, and outputs MUST remain inside the owning submodule unless they are reusable core PHM-Vibench code.
- **FR-006**: Submodule file edits MUST be committed inside the submodule before any parent gitlink update is treated as intentional.
- **FR-007**: The alignment plan MUST discover actual LaTeX entrypoints and selected bibliography tooling before compile tasks are generated.
- **FR-008**: Each selected figure, table, metric, baseline, and result claim in a selected LaTeX entrypoint MUST map to a run artifact, documented external source, or unresolved blocker.
- **FR-009**: Placeholder or unsupported paper claims MUST be removed, marked unresolved, or blocked; they MUST NOT be silently retained as verified.
- **FR-010**: Compilation gates MUST record the exact command, output PDF path, log path, and first actionable error when compilation fails.
- **FR-011**: Missing TeX toolchain or missing final entrypoint MUST be recorded as a skipped/blocked gate with impact.
- **FR-012**: Parent-level paper docs MUST remain navigation and boundary documents; detailed paper roadmaps and evidence stay inside submodules.

### Key Entities

- **UXFD Submodule Contract**: A submodule's `VIBENCH.md`, minimal config, smoke
  command, and expected artifacts.
- **Minimal Evidence Gate**: A root CLI or paper-local command with pass/fail/skipped
  status and artifact paths.
- **LaTeX Entry Point**: A selected TeX file intended to compile into a paper PDF.
- **Claim Evidence Link**: A trace from paper figure, table, metric, baseline, or
  text claim to an artifact, external source, or blocker.
- **Compile Gate**: A command, log, and PDF status for a selected LaTeX entrypoint.
- **Submodule Pointer State**: Parent gitlink status plus submodule-local commit
  status for any paper-specific edits.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All seven UXFD submodules have recorded contract status for `VIBENCH.md`, minimal config, smoke command, and expected artifacts.
- **SC-002**: Each attempted UXFD minimal config records command, result, and artifact paths or blocker reason.
- **SC-003**: Every selected LaTeX figure/table/result claim maps to an artifact, source, or blocker.
- **SC-004**: Selected LaTeX entrypoints have compile pass/fail/skipped status with commands and logs recorded.
- **SC-005**: Any submodule gitlink change is backed by a submodule-local commit or explicitly recorded as not intentional.
- **SC-006**: Unsupported claims, missing artifacts, missing toolchains, and missing entrypoints are documented as blockers rather than verified outputs.

## Assumptions

- Slice 1 provides the runtime artifact contract for root CLI runs.
- Slice 2 and Slice 3 provide task/model/baseline support status used by paper claims.
- Some UXFD paper submodules may contain historical paths or local scripts; `VIBENCH.md`
  and `configs/vibench/min.yaml` are the parent-facing reproduction contract.
- This slice may inspect named result artifacts but must not recursively read large
  paper results or ignored local thesis workspaces.
- Existing user work in submodules or the parent worktree must not be reverted while
  implementing this slice.
