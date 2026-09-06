# Paper source contracts

This directory records how historical P07 and P09 paper overlays were
accounted for. It is not imported by either runtime, and none of its contents
is experimental or positive-claim evidence. `SOURCE_MAP.yaml` is the audited
mapping authority.

## P07 XOAN

The executable P07 runtime is maintained in
`configs/experiments/p07_xoan_operator_attention/`. The legacy payload from
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
`.gitmodules` state and historical paper gitlink deletions. Positive claims
still require fresh, hash-bound runs recorded by the corresponding paper
repository.
