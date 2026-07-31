# P09 HSE-Prompt GFS source contract

The executable P09 runtime is maintained in `configs/experiments/p09/` on
this branch. This directory records how the historical Metric-derived source
was accounted for; it is not imported by the runtime and is not accepted
experimental evidence.

The commit-pinned GFS configs and launch scripts are preserved by the P09
paper repository under `legacy/phmfactory_foundation_2dd7da/`. The old
paper-branch validator is superseded by the maintained
`scripts/validate_configs.py`, so it is not copied. See `SOURCE_MAP.yaml` for
the exact audit and exclusions.

Do not merge the old paper branch wholesale: it also carries obsolete
`.gitmodules` state and historical paper gitlink deletions.
