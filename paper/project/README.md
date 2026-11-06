# Paper source contracts

This directory records how historical P07 and P09 paper overlays were
accounted for. It is not imported by either runtime, and none of its contents
is experimental or positive-claim evidence. `SOURCE_MAP.yaml` preserves the historical migration mapping, not current
implementation ownership or permission to add paper material.

## P07 XOAN

P07 code, study configurations and tests are maintained in
`AI4Engineering-L/P07-XOAN-Operator-Attention`: active bounded DAG code in
`P4/`, retired XOAN protocol in `legacy/p07_protocol/` and `legacy/p07_runtime/`.
No P07 implementation remains in the shared runtime. The legacy payload from
`paper/p07-xoan-operator-attention` is already preserved in the P07 paper
repository under `legacy/source_snapshot/`, so it is not duplicated here.

## P09 HSE-Prompt GFS

The executable P09 runtime is maintained in `configs/experiments/p09/`. The
commit-pinned GFS configs and launch scripts are preserved by the P09 paper
repository under `legacy/phmfactory_foundation_2dd7da/`. The historical
paper-branch validator is superseded by the maintained
`scripts/validate_configs.py`, so it is not copied.

## Evidence boundary

Do not merge either old paper branch wholesale: they carry obsolete
`.gitmodules` state and historical paper gitlink deletions. Performance and explanation claims require current-source experiments recorded
by the corresponding paper repository: exact configuration, data population,
split, selected checkpoint, estimators, seeds, and results. A source snapshot
or a valid identifier alone does not establish a scientific result.
