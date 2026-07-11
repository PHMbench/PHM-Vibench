# Branch Governance and Merge Execution Plan (2026-07-09)

This document records the non-destructive execution plan for consolidating historical branches and PR work into `main`.

## Executive decision

Do **not** merge all branches directly into `main`.

`main` is the stable product line. It should accept only small, validated, reproducible changes that preserve the configuration-first workflow:

```bash
python main.py --config <yaml> [--override key=value ...]
```

The rest of the branch inventory should be handled by one of four outcomes:

1. already merged -> delete source branch after provenance is recorded;
2. no-op / stale -> archive or delete after owner confirmation;
3. useful but unvalidated -> salvage into an integration branch and split into small PRs;
4. active feature -> rebuild as a minimal PR against latest `main`.

## Critical refinements to the original plan

### 1. Do not perform destructive cleanup first

Branch deletion, forced ref movement, and immediate merges are postponed until the audit record is merged. The safe order is:

1. document branch decisions;
2. create integration branch(es);
3. split work into reviewable PRs;
4. run validation gates;
5. only then delete or archive stale branches.

### 2. Treat GitHub `mergeable: true` as necessary but insufficient

A PR can be technically mergeable and still be architecturally unsafe. Admission into `main` requires:

- no hard-coded local paths;
- no public CLI breakage;
- no bypassing factory / registry wiring;
- no paper-only dependency in core validation;
- config registry and generated atlas kept in sync;
- smoke/config/docs/tests passing or an explicit reason why a test is not applicable.

### 3. Keep GUI, paper, UXFD, and core runtime separate

The repository already has a configuration-first core path. GUI and paper workflows are allowed, but they must remain optional and must not become a dependency for the core onboarding path.

### 4. Split integration branches by architectural boundary, not by author or date

Large historical branches should be decomposed into vertical slices with independent acceptance tests. The right unit of merge is not "whatever a branch contains"; it is "one coherent capability with a validation gate".

### 5. Do not merge maintained demos before their target runtime exists

The original draft put "config registry + demo skeleton" before "TSPN/UXFD model core". That order is unsafe: a maintained demo in `configs/demo/` must not reference a model that is absent from the runtime/registry. Therefore the UXFD order is revised below: minimal runtime contract first, maintained config promotion second.

## Execution status

- Created `integration/uxfd-main-20260709` from latest `main`.
- Created `feature/branch-governance-20260709` from latest `main`.
- Added and refined this document on `feature/branch-governance-20260709`.

No destructive action has been taken.

## Branch / PR decision matrix

| Item | Current interpretation | Decision | Next action |
| --- | --- | --- | --- |
| PR #3, #4, #5, #7, #12-#20, #36 | Already merged | Cleanup candidate | Delete source branch only after this audit doc is merged |
| PR #6 | Revert of earlier work, closed/unmerged, not directly mergeable | Archive only | Do not merge unless a concrete regression requires it |
| PR #21-#28 | Streamlit GUI series, closed/unmerged, partially salvageable | Salvage only | Rebuild as `feature/streamlit-app-experimental` under `apps/streamlit/` |
| PR #29 | Draft/closed/no effective file changes | Stale/no-op | Archive or delete after owner confirmation |
| PR #30 | Closed/unmerged/no effective file changes | Stale/no-op | Archive or delete after owner confirmation |
| PR #35 | Active TSPN/X_model work, base is not `main`, not mergeable to its current base | Active review | Rebuild as `feature/tspn-x-model-minimal` after UXFD boundary is clarified |
| PR #37 | Closed/unmerged, useful pretrain/fewshot material, contains hard-coded local paths | Salvage only | Extract docs/config ideas after removing machine-specific paths |
| `lq_merge_UXFD` | Diverged integration branch with UXFD/NSN/TSPN/docs/tests mixed together | Split integration | Use `integration/uxfd-main-20260709`, then split into U1-U5 PRs |

## Main admission gates

Every PR into `main` should list the exact commands it ran. The default gate is:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
python -m pytest test/
```

If a PR only touches docs, the author may narrow the gate, but the PR body must explain why runtime tests are not applicable.

## Revised UXFD split plan

### U1: TSPN_UXFD minimal runtime contract

Goal: make the new model importable and instantiable without promoting demos as maintained examples too early.

Scope:

```text
src/model_factory/X_model/TSPN_UXFD.py
src/model_factory/X_model/UXFD/*
src/model_factory/model_factory.py
src/model_factory/model_registry.csv
configs/base/model/tspn_uxfd.yaml
test/test_tspn_uxfd_assembly.py
```

Acceptance:

```bash
python -m pytest test/test_tspn_uxfd_assembly.py
python -m scripts.config_inspect --config <temporary-or-reference-uxfd-config> --dump targets --format yaml
```

Notes:

- The base config may be added in U1 because it is part of the model contract.
- Maintained demos should not be added until the model target exists and at least one assembly test passes.
- If config inspection requires a committed demo, put it under `configs/reference/uxfd/` or `configs/experiments/uxfd/` until U2 promotes it.

### U2: UXFD maintained demos and config registry promotion

Goal: promote UXFD examples to the maintained `configs/demo/` surface only after U1 exists.

Scope:

```text
configs/demo/uxfd/*
configs/config_registry.csv
docs/CONFIG_ATLAS.md
configs/demo/README.md
configs/demo/uxfd/README.md
```

Acceptance:

```bash
python -m scripts.validate_configs
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.config_inspect --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml
python main.py --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml --override trainer.num_epochs=1
```

### U3: NSN / neurosymbolic wrapper extension

Goal: introduce NSN as an explicit wrapper after the UXFD core contract is stable.

Scope:

```text
src/model_factory/X_model/NSN.py
configs/demo/nsn/*
configs/config_registry.csv
docs/CONFIG_ATLAS.md
configs/demo/nsn/README.md
```

Acceptance:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/nsn/00_smoke_nsn_min.yaml
python main.py --config configs/demo/nsn/00_smoke_nsn_min.yaml --override trainer.num_epochs=1
```

If the demo requires non-repository data, it must be downgraded from `configs/demo/` to `configs/experiments/` or `configs/reference/`.

### U4: Run artifacts and plot factory

Goal: add provenance, explainability artifacts, and plotting as optional backend capabilities.

Scope:

```text
src/explain_factory/run_artifacts.py
src/plot_factory/*
scripts/collect_uxfd_runs.py
test/test_collect_uxfd_runs.py
test/test_run_artifacts_contract.py
```

Acceptance:

```bash
python -m pytest test/test_collect_uxfd_runs.py test/test_run_artifacts_contract.py
```

### U5: Documentation and paper notes

Goal: document UXFD/paper context without making paper-only workflows part of core validation.

Scope:

```text
paper/UXFD_paper/*
src/changelog/CHANGELOG.md
CONTRIBUTING.md
README_CN.md
AGENTS.md
```

Acceptance:

```bash
python -m scripts.validate_docs
```

Paper-only content must not become a core validation dependency.

## TSPN_X_model review checklist

Before rebuilding PR #35 as a minimal PR, review these points:

1. Does the TSPN classifier requiring `num_classes` as a dictionary break existing single-dataset workflows?
2. Does fallback to the first classifier head hide missing or incorrect `dataset_id`?
3. Does removing eager `.to(device)` from parameter initialization preserve CPU/CUDA behavior?
4. Is metadata-derived `in_channels` safe for all TSPN-family models, or only for selected models?
5. Should dataset-specific heads live in generic TSPN, or only in a UXFD-specific subclass/config?

## Streamlit app policy

Streamlit work should be rebuilt as an optional app:

```text
apps/streamlit/
```

Rules:

- do not modify `main.py`;
- do not change core pipelines;
- invoke training with `python main.py --config ...`, not `--config_path`;
- keep `streamlit` optional;
- document it as experimental and not part of the core validation gate.

Minimal acceptance:

```bash
python -m py_compile apps/streamlit/*.py
python -m scripts.validate_docs
```

## Destructive cleanup hold list

The following actions are intentionally deferred:

- deleting source branches for already merged PRs;
- deleting stale no-op branches;
- closing or retargeting active PR #35;
- merging `lq_merge_UXFD` wholesale;
- force-moving any branch ref.

These actions should happen only after the corresponding split PRs have been reviewed and merged, or after repository owner confirmation.
