# Feature Specification: Agent Context Cleanup

**Feature Branch**: `001-clean-agent-context`  
**Created**: 2026-05-09  
**Status**: Draft  
**Input**: User description: "清除会污染智能体的上下文"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Identify Canonical Agent Context (Priority: P1)

As an AI-assisted maintainer, I need a clear whitelist of files that are safe to
read first so that I do not load stale plans, archived notes, generated reports,
or tool-private material as primary project truth.

**Why this priority**: The main failure mode is not missing information; it is
reading too much non-authoritative context and treating it as current guidance.

**Independent Test**: A new maintainer can list the canonical context whitelist
and explain which files are non-authoritative without opening archived reports
or tool-private directories.

**Acceptance Scenarios**:

1. **Given** a new agent starts in the repository root, **When** it asks what to
   read first, **Then** it is directed to the canonical context whitelist.
2. **Given** archived reports or old planning files exist, **When** the agent
   encounters them, **Then** it can tell they are secondary evidence and not
   primary operating instructions.

---

### User Story 2 - Archive Context-Polluting Material (Priority: P2)

As a repository maintainer, I need old examples, reports, schemas, and project
notes to live under documented archive locations so that the repository root
stays clean while historical evidence remains discoverable.

**Why this priority**: Cleaning by deletion can lose evidence, while leaving
root-level noise encourages agents to read the wrong material.

**Independent Test**: A maintainer can verify that moved material has an indexed
destination under `docs/` and that no maintained source-of-truth directory was
left at the repository root.

**Acceptance Scenarios**:

1. **Given** a legacy root-level report directory exists, **When** it is cleaned,
   **Then** the material is archived under a documented `docs/` path with an
   index or migration note.
2. **Given** a future contributor wants to add a report or schema, **When** they
   check the guidance, **Then** they know to place it under `docs/` or a
   configured output directory, not as a new root-level context source.

---

### User Story 3 - Review Context Hygiene (Priority: P3)

As a reviewer, I need a simple acceptance rule for context hygiene so that I can
block changes that introduce new confusing context surfaces.

**Why this priority**: Context hygiene needs a repeatable review rule, not a
one-time manual cleanup.

**Independent Test**: A reviewer can inspect the repository root and context
guidance and decide whether the change passes or fails without understanding
the generative benchmark implementation.

**Acceptance Scenarios**:

1. **Given** a PR adds a new root-level context directory, **When** the reviewer
   checks it against the context hygiene rule, **Then** the PR is rejected unless
   the directory is explicitly allowed.
2. **Given** a PR updates assistant guidance, **When** the reviewer checks the
   canonical whitelist, **Then** the guidance remains consistent across the
   onboarding and agent-facing documents.

### Edge Cases

- If an archived file is still referenced by existing documentation, the
  reference must point to the new archive path or a maintained index.
- If a tool-private directory contains useful evidence, it remains non-canonical
  unless a maintainer promotes the evidence into a documented `docs/` location.
- If a future feature needs additional canonical context, it must amend the
  whitelist rather than relying on agents to discover the file opportunistically.
- If a generated report is needed for a paper or review, it must be linked from
  an index and must not become primary agent instructions by existing at root.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST define a canonical agent context whitelist
  consisting of `README.md`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
  `.specify/memory/constitution.md`, and the active feature's `spec.md`,
  `plan.md`, and `tasks.md` when present.
- **FR-002**: The repository MUST distinguish canonical context from archived
  evidence, generated reports, tool-private material, and local experiment
  outputs.
- **FR-003**: The repository MUST document that root-level `examples`,
  `metrics_reports`, `projects`, `reports`, and `schemas` directories are not
  valid maintained context destinations.
- **FR-004**: Historical material that is still useful MUST be archived under a
  documented `docs/` path and remain discoverable through an index, README, or
  migration note.
- **FR-005**: The cleanup guidance MUST preserve evidence rather than deleting
  material by default.
- **FR-006**: Review guidance MUST provide a pass/fail rule for new context
  surfaces introduced at the repository root.
- **FR-007**: Tool-private directories such as local assistant caches, reviewer
  workspaces, and virtual environments MUST NOT be treated as canonical context.
- **FR-008**: The feature MUST avoid changing runtime behavior, benchmark
  metrics, model code, data loading, or generative pipeline behavior.

### Key Entities *(include if feature involves data)*

- **Canonical Context Whitelist**: The small set of files an agent should read
  first and treat as current project guidance.
- **Archived Evidence**: Historical reports, examples, schemas, notes, and
  reviewer outputs that may be useful but are not primary operating context.
- **Context Surface**: Any root-level file or directory likely to be discovered
  by an agent during onboarding or repository exploration.
- **Archive Index**: A maintained document that explains where moved material
  went and what authority it has.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reviewer can identify the canonical context whitelist in under
  two minutes from the repository root.
- **SC-002**: The repository root contains zero maintained `examples`,
  `metrics_reports`, `projects`, `reports`, or `schemas` directories after
  cleanup.
- **SC-003**: Every archived context-polluting root directory has a documented
  destination under `docs/` or an explicit note that it is intentionally ignored.
- **SC-004**: A reviewer can determine pass or fail for a new root-level context
  directory using the documented hygiene rule without reading runtime code.
- **SC-005**: The cleanup can be validated without changing model, task,
  pipeline, data, or metric behavior.

## Assumptions

- The intended audience is AI-assisted maintainers and reviewers working from
  the repository root.
- The default cleanup action is archive plus index, not deletion.
- Existing tool-private directories may remain present locally, but they are not
  canonical context.
- This feature governs context hygiene only; PHM generative benchmark evidence
  and paperpack rules remain governed by the project constitution.
