# paper/: Local Memory (Research / Writing)

## vibe research guide and rule
- Paper work lives under `paper/**`. Each paper project declares its own SSOT locally (e.g., `paper/**/CLAUDE.md`).
- For `paper/paper_state_flow/`, treat `paper/paper_state_flow/` as the replication package boundary (configs/scripts/results live
  under that folder).

## Scope guard (paper-only by default)
- For paper-writing / research-doc tasks: only change files under `paper/**` unless explicitly requested otherwise.
- If code changes are required, write a concrete handoff request in `paper/paper_state_flow/core/STATUS.md` (what/where/how to
  validate), instead of changing `src/**` directly.

## Where to write
- Core doc map: `paper/paper_state_flow/core/README.md`
- Backlog: `paper/paper_state_flow/core/09_TODO_and_Backlog.md`

## Optional validation
- Docs links: `python -m scripts.validate_docs` (run from repo root)
