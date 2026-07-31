# P07 XOAN source contract

The executable P07 runtime is maintained in
`configs/experiments/p07_xoan_operator_attention/` on this branch. This
directory records how the historical paper overlay was accounted for; it is
not imported by the runtime and it is not experimental evidence.

The legacy source payload from `paper/p07-xoan-operator-attention` is already
preserved in the P07 paper repository under `legacy/source_snapshot/`. It is
therefore not duplicated here. See `SOURCE_MAP.yaml` for the audited
allowlist, exact commits, and exclusions.

Do not merge the old paper branch wholesale: it also carries obsolete
`.gitmodules` state and historical paper gitlink deletions. Positive claims
still require fresh, hash-bound runs recorded by the P07 paper repository.
