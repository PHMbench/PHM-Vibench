# GOAL-GEN-003: Codex-to-Claude Handoff Protocol

## Goal ID

GOAL-GEN-003

## Objective

Create Codex-to-Claude handoff and Claude review templates under the active
Speckit feature directory, with module README pointers for relevant context.

## Why

Codex should implement small goals. Claude Code should review architecture,
loss correctness, leakage risk, evidence gates, and factory compliance. A
structured handoff prevents vague review and creates a Codex-ready fix loop.
Process artifacts belong under the active Speckit feature, not under `docs/`.

## Current Facts To Verify

Run:

```bash
cat .specify/feature.json
find specs -maxdepth 2 -type f | sort
sed -n '1,220p' src/task_factory/Components/generative/README.md
sed -n '1,220p' src/model_factory/generative_model/README.md
```

Verify the active Speckit feature directory exists before adding Claude Teams
or handoff materials. `.codex/` and `.claude/` are tool scratch locations, not
canonical process-artifact destinations.

## Scope

Allowed to add or update:

- `specs/<active-feature>/reviews/README.md`
- `specs/<active-feature>/reviews/claude-team/<run-id>/TASK_SPEC.md`
- `specs/<active-feature>/reviews/claude-team/<run-id>/report.md`
- `specs/<active-feature>/reviews/claude-team/<run-id>/risks.md`
- `specs/<active-feature>/reviews/claude-team/<run-id>/test-log.md`
- `specs/<active-feature>/handoffs/README.md`
- `specs/<active-feature>/handoffs/<date>-<goal-id>.md`
- `.codex/claude-team-runs/*` and `.claude/handoffs/*` only as tool scratch or
  mirrors, not canonical process-artifact storage.

## Out Of Scope

- Do not run Claude automatically.
- Do not implement runtime.
- Do not modify `main.py`.
- Do not push, deploy, publish, delete files, or read secrets.
- Do not create handoff docs under `docs/`.

## Required Behavior

The handoff must include:

- Goal ID
- Objective
- Files changed
- Runtime behavior changed: yes/no
- Contracts touched
- Validation commands run
- Validation results
- Known risks
- Required Claude reviewers
- Required context files
- Review output format

Required Claude reviewer roles:

- `phm-gen-architect`
- `phm-gen-loss-reviewer`
- `phm-gen-leakage-reviewer`

Claude review guide must require reading module README context:

```bash
cat src/task_factory/task/generative/README.md
cat src/task_factory/Components/generative/README.md
cat src/task_factory/Components/generative/losses/README.md
cat src/task_factory/Components/generative/manifests/README.md
cat src/model_factory/generative_model/README.md
```

Claude output must end with:

```xml
<REVIEW_DECISION>APPROVE | REQUEST_CHANGES | BLOCKING</REVIEW_DECISION>
<BLOCKING_ISSUES>
...
</BLOCKING_ISSUES>
<NON_BLOCKING_ISSUES>
...
</NON_BLOCKING_ISSUES>
<FIX_INSTRUCTION>
Codex-ready patch instruction.
</FIX_INSTRUCTION>
```

Claude Code Teams task spec must use review mode first, three teammates by
default, no edits, and must require `report.md`, `risks.md`, and `test-log.md`.
The canonical task spec and review outputs must be stored or mirrored under the
active Speckit feature directory, not only under `.codex/` or `.claude/`.
Subagent/teammate acceleration is allowed only for bounded, non-blocking
read-only planning or review scopes; Codex must verify all outputs locally and
must not delegate urgent blocking work in this handoff goal.

## Deliverables

- Feature-scoped Codex-to-Claude handoff README or template.
- Feature-scoped Claude review guide.
- Claude Teams review task spec.
- Session handoff.

## Acceptance Criteria

- Handoff doc contains machine-parseable review tags.
- Review guide checks factory placement, CFM target, split guard, FFT eval-only
  rule, synthetic leakage, and research-only frontier models.
- Claude Teams package is prepared but not launched.
- No runtime code is changed.
- No handoff material is added under `docs/`.

## Validation Commands

```bash
python -m scripts.validate_docs
rg -n "REVIEW_DECISION|phm-gen-|src/task_factory/Components/generative" specs/002-phm-genbench-frontier/reviews specs/002-phm-genbench-frontier/handoffs
```

## Failure Handling

Report `VALIDATION_UNAVAILABLE` if Claude CLI is unavailable; do not block docs
creation. Report `SCOPE_VIOLATION` if review setup requires runtime changes.

## Review Checklist

- [ ] Does the handoff record runtime behavior changed yes/no?
- [ ] Does it list required context files?
- [ ] Does it include reviewer roles?
- [ ] Does it force machine-parseable Claude output?
- [ ] Does the Teams task spec forbid push/deploy/delete/secret access?
- [ ] Does it keep process docs under the active Speckit feature?
