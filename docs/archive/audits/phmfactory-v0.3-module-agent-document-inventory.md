# PHMFactory v0.3 Module Agent Document Inventory

## Purpose

This audit prevents module-level Agent documents from being deleted merely
because their names are similar. Every listed file was first preserved outside
the public upstream from immutable source commit:

```text
a331769d4005018bc833534ecf4efeb5e8a5a78d
```

The private preservation job used `git archive` and verified every destination
file against its source Git blob SHA.

## Disposition rules

| Status | Meaning |
| --- | --- |
| `remove-archived` | Clearly historical or Agent-only; exact archived copy exists and current references are zero. |
| `merge-neutral-first` | Contains potentially useful module knowledge; merge verified facts into a neutral README or developer guide before removal. |
| `protected-review` | Describes a protected runtime area; requires implementation-aware review and cannot be mechanically removed. |

## Inventory

| Path | v0.3 status | Required next action |
| --- | --- | --- |
| `dev/test_history/AGENTS.md` | `remove-archived` | Remove in this batch; maintained testing guidance is `docs/testing.md`. |
| `configs/base/CLAUDE.md` | `merge-neutral-first` | Compare with `configs/README.md`; retain only accurate base-composition guidance. |
| `configs/demo/CLAUDE.md` | `merge-neutral-first` | Compare with `configs/demo/README.md`; retain maintained-demo admission rules. |
| `configs/experiments/CLAUDE.md` | `merge-neutral-first` | Compare with the experiments README; retain only current local-experiment policy. |
| `configs/reference/CLAUDE.md` | `merge-neutral-first` | Record historical/reference status without presenting it as maintained guidance. |
| `src/configs/CLAUDE.md` | `merge-neutral-first` | Compare with `src/configs/README.md` and public configuration docs. |
| `src/data_factory/CLAUDE.md` | `protected-review` | Merge only implementation-backed facts into `src/data_factory/README.md`. |
| `src/data_factory/reader/CLAUDE.md` | `protected-review` | Preserve reader contracts; verify against actual function-based readers before neutral merge. |
| `src/model_factory/CLAUDE.md` | `protected-review` | Compare with registry, factory, and maintained README. |
| `src/model_factory/CNN/CLAUDE.md` | `merge-neutral-first` | Consolidate accurate CNN-family notes into a neutral family README. |
| `src/model_factory/ISFM/CLAUDE.md` | `merge-neutral-first` | Consolidate accurate ISFM notes without changing model code. |
| `src/model_factory/ISFM_Prompt/CLAUDE.md` | `merge-neutral-first` | Retain only implementation-backed prompt-model guidance. |
| `src/model_factory/MLP/CLAUDE.md` | `merge-neutral-first` | Consolidate accurate MLP-family notes. |
| `src/model_factory/NO/CLAUDE.md` | `merge-neutral-first` | Consolidate accurate neural-operator notes. |
| `src/model_factory/RNN/CLAUDE.md` | `merge-neutral-first` | Consolidate accurate RNN-family notes. |
| `src/model_factory/Transformer/CLAUDE.md` | `merge-neutral-first` | Consolidate accurate Transformer-family notes. |
| `src/model_factory/X_model/CLAUDE.md` | `protected-review` | Review experimental/explainability scope before neutral consolidation. |
| `src/task_factory/CLAUDE.md` | `protected-review` | Compare with task registry, factory code, and contribution guide. |
| `src/task_factory/Components/CLAUDE.md` | `merge-neutral-first` | Consolidate component-level facts into neutral module docs. |
| `src/trainer_factory/CLAUDE.md` | `protected-review` | Compare with trainer construction and Lightning compatibility docs. |
| `src/utils/CLAUDE.md` | `merge-neutral-first` | Compare with `src/utils/README.md` and API reference; remove stale claims. |

## Batch 03 decision

Only `dev/test_history/AGENTS.md` is removed in this batch because:

1. it is explicitly a legacy document;
2. no current repository reference targets it;
3. its exact Git blob is preserved;
4. `dev/test_history/README.md` already points to maintained testing guidance;
5. it contains outdated advice that conflicts with current validation gates.

The remaining 21 files are not deleted by this PR.

## Protected-runtime boundary

This inventory does not authorize changes to reader, factory, task, trainer, or
Pipeline implementations. Neutral-document consolidation must remain in
separate, module-scoped PRs and must not include runtime refactors.
