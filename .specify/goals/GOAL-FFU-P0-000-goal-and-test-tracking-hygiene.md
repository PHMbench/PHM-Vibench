# GOAL-FFU-P0-000: Goal And Test Tracking Hygiene

## Objective

Make PHM-GenBench Speckit artifacts, goal files, constitution, and new tests
trackable without opening the repository to unrelated generated files.

## Scope

Allowed to modify:

- `.gitignore`
- `.specify/feature.json`
- `.specify/goals/*.md`
- `.specify/memory/constitution.md`
- `specs/002-phm-genbench-frontier/**`

Out of scope:

- Runtime code.
- Test implementation beyond making future test paths trackable.

## Acceptance Criteria

- `.specify/goals/*.md` appears in `git status --short --untracked-files=all`.
- `.specify/memory/constitution.md` appears when modified.
- New `test/**/*.py` files are not ignored.
- Existing local ignored artifacts remain ignored.

## Validation Commands

```bash
git check-ignore -v .specify/goals/GOAL-FFU-P0-001-constitution.md || true
git check-ignore -v .specify/memory/constitution.md || true
git check-ignore -v test/generative/test_preflight.py || true
git status --short --untracked-files=all
```
