# 2026-11-16 — Agent ownership boundary

Clarify in the existing `AGENTS.md` that paper tasks cannot modify upstream source
through editable installs, submodules, symlinks or shared checkouts, alter pins, or
write experiment outputs into upstream. Public standalone capabilities and legitimate
citations remain permitted; keyword matches require semantic review, not deletion.

Add staged/full-PR diff inspection commands and reuse existing boundary checks.
Correct the conflicting English/Chinese contribution guidance that allowed manuscript
variants to be staged in `configs/experiments/`.

This documentation change does not move or remove implementations, alter runtime,
scientific protocols, tests, CI, data, results, dependencies or filesystem permissions.
It does not integrate the separate ownership migration or other pending runtime PRs.
Validation evidence and exact-head CI status are recorded on this change's PR; no
industrial performance or complete historical contamination audit is claimed.
