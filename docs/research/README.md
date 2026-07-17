# Research method inventory

This directory summarizes externally published 2025-2026 methods that may be
relevant to PHM-Vibench. The machine-readable source of truth is
`research/2025_2026/method_registry.csv`; the method atlas is generated from it.

Registry presence means that primary publication and code sources were reviewed.
It does not mean that a method is implemented, release-supported, or
benchmark-valid in PHM-Vibench.

```bash
python -m scripts.validate_research_registry
python -m scripts.gen_research_atlas
git diff --exit-code docs/research/2025_2026_METHOD_ATLAS.md
```

Publication state and repository maturity are independent:

- `peer_reviewed` and `accepted` describe publication evidence only.
- `preprint` and `submission` remain at most `research_only`.
- `experimental_candidate` identifies a selected local implementation target.
- `benchmark_candidate` and `benchmark_valid` require repository-local,
  leakage-safe, repeated-run evidence.

Code with an `unknown` or `conflicting` license must not be copied or vendored.
An optional dependency adapter or a clean-room implementation from the paper may
still be considered when documented explicitly.
