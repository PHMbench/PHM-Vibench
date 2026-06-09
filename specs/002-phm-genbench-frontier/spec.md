# Feature Specification: PHM-GenBench Frontier

**Feature Branch**: `002-phm-genbench-frontier`  
**Created**: 2026-05-10  
**Status**: Draft  
**Input**: User description: "搜索网上最前沿的生成模型，整合到当前仓库，优化 goal 到 .specify/goals，并结合 handoff、claude-code-teams 和 Speckit 工作流，最终达到生成模型到 PHM 应用的顶级论文水平"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Govern Benchmark Validity (Priority: P1)

As a PHM benchmark maintainer, I need enforceable governance for generative
experiments so that synthetic data cannot be presented as benchmark-valid
without config, protocol, normalization, condition, leakage, and metric evidence.

**Why this priority**: Without validity governance, new frontier models can make
the repository look advanced while producing non-auditable paper results.

**Independent Test**: A reviewer can read the constitution and the active goals
and decide whether a generated dataset is `benchmark-valid`, `exploratory`, or
`docs-only` without inspecting implementation internals.

**Acceptance Scenarios**:

1. **Given** a generative run lacks normalization evidence, **When** its manifest
   is reviewed, **Then** it cannot be accepted as `benchmark-valid`.
2. **Given** a frontier model is added, **When** no promotion goal exists, **Then**
   it remains experimental or demo-only.

---

### User Story 2 - Preflight And Evidence Loop (Priority: P2)

As a PHM researcher, I need a strict train/sample/eval/paperpack loop so that I
can reproduce synthetic-signal experiments and compare model families fairly.

**Why this priority**: The paper claim depends on reproducible evidence, not just
model definitions.

**Independent Test**: A researcher can run preflight, one smoke train, one
condition-grid sample, one eval, and one paperpack command on the dummy
generative config.

**Acceptance Scenarios**:

1. **Given** a malformed config, **When** preflight runs, **Then** the command
   fails before training starts.
2. **Given** a condition grid is requested, **When** sample mode runs, **Then**
   generated samples and manifest condition counts cover every requested
   fault/domain pair.
3. **Given** eval metrics are not computable, **When** paperpack is generated,
   **Then** missing values include status and reason fields.

---

### User Story 3 - Integrate Frontier Model Families (Priority: P3)

As a generative-method integrator, I need current flow, diffusion, one-step, and
sequence-backbone families integrated through existing factories so that the
repository can support paper-grade PHM comparisons without parallel runtimes.

**Why this priority**: The project needs credible frontier coverage, but only
after the governance and evidence loop protect benchmark claims.

**Independent Test**: Each promoted family has a runnable demo config,
factory registration, schema coverage, CPU smoke test, manifest compatibility,
and exploratory default validity.

**Acceptance Scenarios**:

1. **Given** a Rectified Flow or FlowTS-style baseline config, **When** it runs
   through `main.py`, **Then** it uses the existing generative pipeline and
   writes compatible evidence artifacts.
2. **Given** a MeanFlow/iMF, Drifting, Transition Flow Matching, or OT-NFM method
   is integrated quickly, **When** it is sampled, **Then** its manifest remains
   `exploratory` until a later benchmark-valid promotion goal passes.

---

### User Story 4 - Produce Paper-Grade Review Artifacts (Priority: P4)

As a paper reviewer, I need tables, figure sources, missing-metric explanations,
and reproducibility statements so that I can audit the PHM generative benchmark
without relying on informal experiment notes.

**Why this priority**: A top PHM application paper needs reproducibility and
negative-result visibility.

**Independent Test**: A reviewer can inspect a paperpack directory and trace each
table row back to a run, config hash, manifest, condition policy, and metric
status.

**Acceptance Scenarios**:

1. **Given** multiple seeds are present, **When** paperpack runs, **Then** it
   outputs mean/std tables and a run index.
2. **Given** some metrics are missing, **When** paperpack runs, **Then** it
   reports missing reasons in an appendix artifact.

### Edge Cases

- Frontier papers may be newer than stable time-series baselines; such methods
  must start as exploratory even if they are core-integrated.
- Test or target-test data must not become synthetic source data.
- Optional dependencies such as true Mamba kernels must not become mandatory for
  normal smoke validation.
- A model may run successfully but still be non-benchmark-valid if evidence is
  incomplete.
- Metrics may be non-computable because labels, domains, sample counts, or
  numerical preconditions are missing; these cases require explicit reasons.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The project MUST maintain a PHM-GenBench constitution that defines
  configuration-first execution, 5-block configs, factory-first extension, and
  evidence-gated validity.
- **FR-002**: The project MUST expose strict preflight validation for maintained
  configs before any training, sampling, or evaluation starts.
- **FR-003**: Generative sample mode MUST support explicit condition sampling
  policies for repeated metadata, grids, train-distribution sampling, and
  explicit condition lists.
- **FR-004**: Generative runs MUST record normalization parameter artifacts and
  hashes when benchmark validity is requested.
- **FR-005**: Synthetic manifests MUST record config/protocol hashes, source
  split, condition counts, normalization evidence, leakage checks, and validity
  status.
- **FR-006**: Generative metrics MUST report missing value status and reason
  fields when metrics are not computable.
- **FR-007**: Paperpack generation MUST aggregate multi-seed runs and produce
  reproducibility statements, table CSVs, figure-source CSVs, run index, manifest
  completeness, and missing-metric appendix artifacts.
- **FR-008**: Frontier model families MUST integrate through existing factories
  and configs rather than a parallel runtime.
- **FR-009**: Core-fast experimental methods MUST default to exploratory
  validity and require later promotion before benchmark-valid claims.
- **FR-010**: Claude Code Teams MUST be used first in read-only plan/review mode
  for frontier research, runtime-contract, and paperpack review before broad
  implementation.
- **FR-011**: Handoff documents MUST capture active feature directory, goal
  queue, research anchors, validation status, open risks, and next task IDs.
- **FR-012**: Development-process artifacts for goal execution MUST be stored
  or indexed under the active Speckit feature directory, with `.codex/` and
  `.claude/` treated only as tool scratch or mirrors.
- **FR-013**: Goals that use Claude Code Teams MUST encode teammate roles,
  subagent/teammate acceleration scopes, read-only default mode, endpoint export
  approval, Codex verification, and blocked-review handling in the goal
  contract.
- **FR-014**: Module-specific PHM generative documentation MUST live in the
  README of the owning module or config directory. Project-level `docs/` MUST
  remain an index and MUST NOT accumulate a separate PHM generative docs tree.
- **FR-015**: Benchmark-effect aggregation MUST record configured, observed,
  missing, and unexpected dataset coverage in a manifest so paper claims use
  actual metric evidence rather than matrix definitions alone.

### Key Entities *(include if feature involves data)*

- **Generative Method Family**: A flow, diffusion, one-step, or sequence-backbone
  family integrated as a benchmark candidate.
- **Synthetic Dataset Manifest**: The provenance and validity record for a
  generated dataset.
- **Condition Policy**: The rule used to choose `fault_label` and `domain_id`
  during sample mode.
- **Normalization Evidence**: Per-channel statistics and hash needed to interpret
  generated vibration signals.
- **Paperpack**: The reproducibility bundle containing tables, figure sources,
  appendices, and run indexes.
- **Benchmark Effect Manifest**: The aggregation-level evidence record that
  maps configured datasets to observed metric records and records coverage gaps.
- **Goal File**: A PR-sized implementation contract under `.specify/goals/`.
- **Process Artifact**: Speckit plans, tasks, reviews, handoffs, validation
  logs, blocked-review notes, and paper readiness notes stored under the active
  feature directory.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reviewer can classify a synthetic dataset validity status from
  its manifest and constitution in under five minutes.
- **SC-002**: Both dummy default and dummy generative configs pass preflight
  without starting trainer execution.
- **SC-003**: Condition-grid sample mode produces manifest counts matching every
  requested fault/domain pair.
- **SC-004**: Paperpack aggregates at least two seeds into mean/std tables and
  preserves source paths for all rows.
- **SC-005**: At least five generative families are represented in goals and at
  least four are planned for factory-integrated smoke configs.
- **SC-006**: No frontier method can be marked benchmark-valid without complete
  manifest, normalization, leakage, and metric evidence.
- **SC-007**: Claude Teams review outputs identify unresolved risks before any
  broad implementation begins.
- **SC-008**: Six-dataset readiness checks can identify missing configured
  datasets, unexpected observed datasets, and missing source-path evidence from
  aggregation artifacts without inspecting run directories manually.

## Assumptions

- The target publication is a PHM application/benchmark paper, not a pure
  generative-model theory paper.
- The first implementation phase prioritizes benchmark infrastructure over
  maximizing model count.
- New compiled or CUDA-specific dependencies remain optional and guarded.
- Module-specific documentation for generative work remains in the README next
  to the owning module. Process, review, handoff, and paper-readiness artifacts
  remain under the active Speckit feature directory.
