# Requirements Document

## Introduction

The HSE contrastive task currently duplicates core Default_task logic, accepts invalid loss parameters, and lacks clear documentation of its supporting components. This effort streamlines the task class, hardens contrastive loss configuration, and clarifies the Components layer so future contributors can extend domain-generalization features without guessing at legacy behavior.

## Alignment with Product Vision

By reducing redundant task code and documenting the Components layer, we keep the PHM-Vibench pipelines modular and transparent, enabling faster experimentation across datasets while minimizing maintenance risk—key goals highlighted in the benchmark roadmap.

## Requirements

### Requirement 1

**User Story:** As a task maintainer, I want the `hse_contrastive` task to delegate shared behavior to `Default_task`, so that updates land in one place and the contrastive extension stays lightweight.

#### Acceptance Criteria

1. WHEN the task initializes THEN it SHALL call `super().__init__` before custom setup and reuse Default_task optimizers unless fine-tune param groups are explicitly configured.
2. IF `_shared_step` runs for any stage THEN the method SHALL call `super()._shared_step` and only layer contrastive metrics on top.
3. WHEN contrastive loss is disabled (weight ≤ 0) THEN the task SHALL skip building contrastive branches and return baseline metrics unchanged.

### Requirement 2

**User Story:** As a machine-learning engineer configuring experiments, I want contrastive loss kwargs to be validated against supported losses, so that bad configs fail fast instead of crashing during training.

#### Acceptance Criteria

1. WHEN constructing the contrastive loss THEN the system SHALL normalize the selected loss name and only pass extra kwargs declared for that loss.
2. IF an unsupported kwarg is provided for the chosen loss THEN the system SHALL raise a `ValueError` during setup with actionable guidance.
3. WHEN the loss constructor raises an internal error THEN the task SHALL re-raise a `RuntimeError` that identifies the loss name and offending parameters.

### Requirement 3

**User Story:** As a new contributor onboarding to the Components layer, I want clear documentation of active versus legacy modules, so that I can understand where to add or retire code without digging through history.

#### Acceptance Criteria

1. WHEN reading `src/task_factory/Components/README.md` THEN the document SHALL describe each module’s purpose, primary APIs, and consumer tasks.
2. IF a component is legacy or unused THEN the README SHALL flag its status and next-step recommendations.
3. WHEN inspecting prompt contrastive helpers THEN docstrings and README references SHALL agree on supported kwargs and tensor shapes.

### Requirement 4

**User Story:** As a roadmap planner for domain-generalization features, I want placeholders to be intentional, so that empty stubs do not ship without defined requirements.

#### Acceptance Criteria

1. WHEN reviewing the Components directory THEN `DG_loss.py` SHALL either be removed or documented as a placeholder with explicit TODOs pending spec approval.
2. IF historical metric-loss variants are unused in the repository imports audit THEN the system SHALL capture a follow-up consolidation task in project docs or backlog.
3. WHEN future domain-generalization work is proposed THEN this spec SHALL surface the open questions around loss registries and DG requirements for stakeholder review.

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: Refactored helpers (e.g., `_apply_contrastive_branch`, `_forward_with_prompts`) SHALL each have focused responsibilities.
- **Modular Design**: New logic SHALL reside in factories or helpers consistent with existing task architecture, avoiding duplicated code paths.
- **Dependency Management**: The task SHALL rely on Lightning logging interfaces instead of Python logging modules.
- **Clear Interfaces**: Helper functions SHALL include type hints and docstrings describing required batch keys and return values.

### Performance
- Contrastive branch execution SHALL not add more than a negligible overhead (≤5%) compared to current runs when contrastive loss is enabled.

### Security
- No new external services shall be invoked; changes operate entirely within the existing training pipeline.

### Reliability
- Unit tests SHALL cover at least two loss types (e.g., InfoNCE and Triplet) to ensure configuration gating works across variants.

### Usability
- Documentation updates SHALL make it straightforward for engineers to configure contrastive losses without referencing source code directly.
