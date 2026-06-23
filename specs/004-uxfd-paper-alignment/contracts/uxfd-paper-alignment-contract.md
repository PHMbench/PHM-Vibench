# Contract: UXFD Paper Alignment

## Sources Of Truth

- UXFD submodule list and boundaries come from `paper/UXFD_paper/README.md`,
  `paper/README_SUBMODULE.md`, and `.gitmodules`.
- Parent-facing reproduction contract comes from each submodule's `VIBENCH.md` and
  `configs/vibench/min.yaml`.
- Runtime artifacts use the Slice 1 artifact contract.
- Task/model/baseline support comes from Slice 2 and Slice 3 status evidence.

## Submodule Contract Audit

For each of the seven UXFD submodules, record:

- `VIBENCH.md` presence;
- `configs/vibench/min.yaml` presence;
- maintained root CLI command or paper-local command;
- expected artifacts;
- status and blocker reason.

Historical README paths or old CLI flags do not override `VIBENCH.md`.

## Minimal Evidence Gate

Root CLI smoke commands must use:

```bash
python main.py --config <submodule-config> --override trainer.num_epochs=1
```

Completed root CLI runs must record Slice 1 artifacts. Paper-local commands may be
recorded, but they do not replace root CLI evidence unless explicitly accepted as
paper-local-only.

## Claim Evidence Contract

Selected LaTeX figure, table, metric, baseline, and result claims must map to one of:

- generated artifact path;
- documented external source;
- unresolved blocker with reason.

Unsupported claims must not be treated as verified.

## Compile Gate Contract

For each selected LaTeX entrypoint, record:

- exact compile command;
- PDF path when produced;
- log path;
- first actionable error when failed;
- skipped or blocked reason when no toolchain or entrypoint exists.

## Submodule Safety Contract

Paper-specific file edits stay inside the owning submodule. A parent gitlink change
is intentional only when:

- the submodule has a local commit containing the paper changes;
- the parent records the new gitlink pointer intentionally;
- the handoff states the submodule path, commit, and reason.
