# GOAL A Scorecard

Use this scorecard at the end of every cycle.

## 1. Recoverability — 20
- 20: every unique tip is protected by verified archive tags or an explicit no-change decision.
- 15: protection plan exists but remote verification is incomplete.
- 10: some refs are classified but not protected.
- 0: destructive or irreversible action was attempted.

## 2. Runtime correctness — 20
- 20: config-first path and smoke run are preserved or improved.
- 15: no runtime regression is known but tests are partial.
- 10: runtime behavior is unclear.
- 0: main user path is broken.

## 3. Merge minimality — 15
- 15: only logical, reviewable slices are proposed.
- 10: patch is reviewable but mixes concerns.
- 5: large patch with weak justification.
- 0: whole-line merge without triage.

## 4. Public user clarity — 15
- 15: README/docs tell a new PHM user exactly how to start.
- 10: docs mostly clear but duplicate or tool-specific material remains.
- 5: docs are fragmented.
- 0: docs mislead users.

## 5. Validation strength — 15
- 15: smoke, config, docs, atlas, and tests are run or explicitly bounded.
- 10: core commands run but full tests not run.
- 5: only static inspection.
- 0: no validation.

## 6. Release cleanliness — 10
- 10: release path contains only source, configs, tests, docs, and formal metadata.
- 5: minor internal material remains with justification.
- 0: workflow scratch material remains in release path.

## 7. Handoff quality — 5
- 5: B can make decisions from the handoff without redoing work.
- 3: enough context but some ambiguity.
- 0: missing or unverifiable handoff.
