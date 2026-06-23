# Contract: PHM Task Experiment Matrix

## Sources Of Truth

- Task families come from `src/task_factory/task_registry.csv`.
- Runnable maintained config entries come from `configs/config_registry.csv`.
- Human-readable config coverage comes from generated `docs/CONFIG_ATLAS.md`.
- Runtime artifact expectations come from Slice 1.

The matrix must not maintain a second manually curated task inventory.

## Status Contract

Every task family present in the task registry must resolve to exactly one status:

- `smoke-tested`: an offline command or focused test passes.
- `real-data-ready`: a full-matrix command is defined and real-data evidence is
  recorded when data is available.
- `unverified`: source-of-truth support exists, but passing evidence is missing.
- `unsupported`: source-of-truth support is absent or intentionally out of scope.

Unknown or incompatible entries must fail or be skipped with an explicit reason.
They must not fall back to a default task.

## Offline Smoke Contract

Command:

```bash
bash scripts/run_demo_matrix.sh --mode smoke
```

Required behavior:

- Does not require `PHM_VIBENCH_DATA`.
- Runs only offline-safe entries.
- Fails with the matrix entry name and command if an entry fails.
- For completed runs, preserves Slice 1 artifact expectations.

## Real-Data Full Contract

Command:

```bash
PHM_VIBENCH_DATA=<data-root> bash scripts/run_demo_matrix.sh --mode full
```

Required behavior:

- Without `PHM_VIBENCH_DATA`, exits before running experiments and reports the
  missing data root.
- With a valid data root, runs selected DG, CDDG, FS, GFS, and pretrain entries.
- Records per-family pass/fail evidence.
- Does not substitute dummy data for real-data entries.

## Registry Consistency Contract

Matrix validation must check:

- Config entries referencing `task.type` and `task.name` exist in the task registry.
- Registry-backed task families without maintained configs are reported as gaps or
  unverified entries.
- Maintained config changes keep `configs/config_registry.csv` and
  `docs/CONFIG_ATLAS.md` synchronized.

## Task/Data Compatibility Contract

Validation must report the matrix entry and task family when these are missing or
incompatible:

- required batch keys;
- domain ids or system ids;
- class labels or class counts;
- few-shot `n_way` and `k_shot` feasibility;
- objective-specific fields for pretraining variants.

If a compatibility issue can only be discovered during assembly or first batch
loading, the failure still must identify the entry and avoid fallback.
