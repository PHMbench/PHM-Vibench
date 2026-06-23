# PHM-GenBench V4 Paper Goals

Date: 2026-06-18

## Purpose

Produce the final high-level PHM generative paper from implemented methods and
honest evidence. The paper should be strong because the method roster is
implemented, reproducible, and clearly scoped, not because exploratory evidence
is overstated.

## Paper Position

Allowed current position:

```text
PHM-GenBench presents a repo-native implementation and exploratory evaluation
of modern PHM generative methods, with reproducible configs and an explicit
claim boundary.
```

Forbidden unless later evidence supports it:

```text
state-of-the-art performance
benchmark winner
method superiority
cross-domain generalization demonstrated
downstream diagnosis improvement
submission-ready benchmark result
```

## GOAL-V4-PAPER-001-PAPER-SCOPE

Owner:
S6 Paper Integration

Objective:
Lock the paper as an implementation-complete, exploratory-evidence PHM
generative benchmark paper.

Required behavior:
- The abstract, introduction, contributions, experiments, and limitations must
  use the same paper position.
- Existing exploratory rows may be retained only with clear wording.
- Method roster promises must match `goal_sota.md`.

Deliverables:
- Paper claim statement.
- Method roster statement.
- List of unsupported claims to remove.

Validation commands:

```bash
python -m scripts.validate_docs
rg -n "state-of-the-art|outperform|benchmark winner|downstream diagnosis improvement" specs/002-phm-genbench-frontier/paper .specify/goals/v4 || true
```

## GOAL-V4-PAPER-002-EVIDENCE-PACK

Owner:
S6 Paper Integration

Objective:
Collect focused evidence for every implemented roster method without requiring
a full benchmark rerun before the first final draft.

Required behavior:
- Record command, config, method status, and result location per method.
- Separate smoke/preflight evidence from exploratory metric evidence.
- Keep old six-dataset evidence as historical exploratory evidence.
- Do not block the paper on heavy statistical promotion gates.

Deliverables:
- `method_evidence_matrix.md`
- command log or manifest paths for each roster method
- explicit missing evidence list

Validation commands:

```bash
python -m scripts.validate_docs
rg -n "implemented|smoke_passed|exploratory_evidence|blocked" specs/002-phm-genbench-frontier/paper .specify/goals/v4
```

## GOAL-V4-PAPER-003-FINAL-DRAFT

Owner:
S6 Paper Integration

Objective:
Write the final paper draft around implemented SOTA methods and claim-safe
exploratory evidence.

Required behavior:
- Every V4 roster method appears in the method section.
- Every method has implementation status and evidence status.
- Main result text avoids superiority claims unless evidence supports them.
- Limitations explain exploratory evidence, sample budget, and future benchmark
  promotion work.
- Reproducibility appendix lists configs and commands.

Deliverables:
- updated paper draft
- updated reproducibility appendix or equivalent paper section
- updated evidence gaps/limitations

Validation commands:

```bash
python -m scripts.validate_docs
rg -n "Score SDE|MeanFlow|Drifting|Transition Flow Matching|OT-NFM" specs/002-phm-genbench-frontier/paper
rg -n "benchmark_valid_row_count=0|exploratory|NOT_SUBMISSION_READY" specs/002-phm-genbench-frontier/paper .specify/goals/v4
```

## GOAL-V4-PAPER-004-REVIEWER-PASS

Owner:
S6 Paper Integration

Objective:
Run the V4 reviewer on the final draft and convert only paper-blocking issues
into follow-up goals.

Required behavior:
- Review focuses on clarity, novelty, reproducibility, and claim safety.
- Reviewer must not demand unnecessary gates before paper production.
- Blocking issues must map to the smallest next paper or implementation goal.

Deliverables:
- review report under `specs/002-phm-genbench-frontier/reviews/v4/`
- reviewer decision block
- next goal if changes are required

Validation commands:

```bash
python -m scripts.validate_docs
rg -n "REVIEW_DECISION|READINESS_SCORE|NEXT_GOAL|FIX_INSTRUCTION" .specify/goals/v4/reviewer.md specs/002-phm-genbench-frontier/reviews/v4 || true
```
