# Feature Specification: PHM 2025+ Literature Integration

**Feature Branch**: `005-phm-2025-literature-integration`  
**Created**: 2026-05-11  
**Status**: Ready for implementation  
**Input**: User description: "执行 .specify/goals/phm-vibench-full-phm-experiment-platform.md 并联网搜索最新工作加入到本仓库体系中，至少加入25年之后的50个工作，在对应readme 中给出参考文献，并且跑通模块"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Curated Recent PHM Work Inventory (Priority: P1)

A PHM-Vibench maintainer can inspect a source-backed inventory of at least 50
PHM works published in 2025 or later, grouped by task family and method family,
without reading ad hoc notes scattered across the repository.

**Why this priority**: The user explicitly requested at least 50 post-2025 works
and references. Without the inventory, later benchmark mapping has no evidence.

**Independent Test**: Run the literature inventory module and verify that it
loads at least 50 unique entries with publication year >= 2025, required source
fields, and PHM task/method labels.

**Acceptance Scenarios**:

1. **Given** the repository checkout, **When** a maintainer runs the literature
   inventory module, **Then** the module reports at least 50 valid 2025+ entries.
2. **Given** an entry in the inventory, **When** the maintainer inspects it,
   **Then** it includes title, year, venue or source, URL or DOI, PHM task family,
   method family, and repository mapping status.

---

### User Story 2 - Repository-System Mapping (Priority: P2)

A benchmark maintainer can see how each recent work maps into the existing
PHM-Vibench task/model/loss/baseline taxonomy as represented by an existing
surface, candidate-baseline, dependency-blocked, unsupported, or literature-only.

**Why this priority**: The goal requires the works to be added to the repository
system, but the constitution forbids marking unverified work as supported.

**Independent Test**: Run the literature inventory module and confirm every
entry has a non-empty task family, method family, repository surface, and status.

**Acceptance Scenarios**:

1. **Given** the curated inventory, **When** the maintainer filters by
   `fault_diagnosis`, `rul`, `domain_generalization`, `few_shot`, or
   `explainability`, **Then** the mapped entries show relevant existing or
   candidate PHM-Vibench surfaces.
2. **Given** a method not implemented in PHM-Vibench, **When** it appears in the
   inventory, **Then** its status is not `smoke-tested` unless runtime evidence
   exists.

---

### User Story 3 - README References And Validation Gate (Priority: P3)

A reader can find the references from the corresponding documentation README and
a maintainer can include the inventory check in local validation without network
access.

**Why this priority**: The user requested references in the corresponding README
and asked that modules run.

**Independent Test**: Run documentation validation, the literature inventory
module, and focused tests for the new inventory.

**Acceptance Scenarios**:

1. **Given** a reader starts at `docs/README.md`, **When** they follow the
   literature reference entry, **Then** they reach a README containing at least
   50 post-2025 PHM references.
2. **Given** the repository is offline after curation, **When** the maintainer
   runs the inventory module, **Then** validation does not require live network
   access.

### Edge Cases

- Duplicate titles or URLs MUST be rejected by the validation module.
- Entries older than 2025 MUST fail validation.
- Entries without a URL or DOI MUST fail validation.
- Entries with unsupported methods MUST remain explicitly marked as
  `literature-only`, `candidate-baseline`, `dependency-blocked`, or `unsupported`.

## Clarifications

### Session 2026-05-11

- No critical ambiguity worth stopping for user input: "25年之后" is interpreted as publication year >= 2025, because the current date is 2026-05-11 and the request asks for latest work after 2025.
- The phrase "加入到本仓库体系" is implemented as a validated literature/method inventory plus README references and mapping to existing registries; it does not mean adding 50 unverified model implementations.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST include a repository-local 2025+ PHM literature inventory with at least 50 unique works.
- **FR-002**: Each inventory entry MUST include title, year, source/venue, URL or DOI, PHM task family, method family, repository mapping surface, and support status.
- **FR-003**: Inventory validation MUST fail explicitly for duplicate IDs, duplicate titles, years before 2025, missing links, missing task labels, missing method labels, or invalid support statuses.
- **FR-004**: The corresponding documentation README MUST provide references or a direct reference table for the 2025+ works.
- **FR-005**: The inventory MUST distinguish runtime-supported PHM-Vibench surfaces from literature-only or candidate-baseline entries.
- **FR-006**: The implementation MUST add a runnable module or script that summarizes and validates the literature inventory without network access.
- **FR-007**: Tests MUST cover the minimum entry count, date filter, required fields, status vocabulary, and task/method coverage.
- **FR-008**: Existing config-first experiment execution MUST remain unchanged; no new runtime model may be marked supported without a passing smoke or focused gate.

### Key Entities *(include if feature involves data)*

- **LiteratureEntry**: One 2025+ PHM work with bibliographic fields, task/method taxonomy, repository mapping, and validation status.
- **RepositoryMapping**: The relationship between a literature method family and PHM-Vibench surfaces such as task registry, model registry, loss components, config demos, or unsupported backlog.
- **InventoryReport**: A generated validation summary containing counts by task family, method family, and support status.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The inventory validation module reports at least 50 unique works with year >= 2025.
- **SC-002**: Documentation validation passes with a README path that exposes the reference list.
- **SC-003**: Focused inventory tests pass and fail-fast behavior is covered for malformed sample rows.
- **SC-004**: Existing benchmark validation commands still pass for configs/docs and a smoke experiment run.

## Assumptions

- Publication year >= 2025 satisfies "25年之后".
- Search results are curated manually from web-accessible publisher, journal, DOI,
  arXiv, or proceedings pages; the repository keeps offline metadata afterward.
- The requested "modules" are satisfied by a runnable validation/reporting module
  and existing benchmark smoke modules, not by unverified implementations of all
  50 cited papers.
