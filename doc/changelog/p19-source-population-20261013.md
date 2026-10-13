# Source-only physical splits and explicit channels — 2026-10-13

The existing read-only grouped splitter can now consume a declared DG/CDDG population with grow/select roles. The mapped target partition must exactly equal the task's target IDs; every other declared role must contain only source IDs. Roles must cover the requested population with disjoint records and physical groups. Invalid inputs stop without modifying the supplied manifest.

Public configuration exposes the existing manifest mode/partition mapping. `data.selected_channels` optionally selects declared reader channels before per-window normalization; absent selection preserves all channels. Invalid, duplicate or unavailable indices fail rather than substituting another input.

Read-only manifest controls and partition mappings now require explicit `strategy: grouped_metadata` at both the public configuration and splitter boundaries. Legacy or omitted strategy fails before public Pipeline construction or splitter manifest access, instead of silently ignoring frozen roles. The public schema does not expose the pre-existing internal `manifest_sha256` field.

These changes support P19's single vibration channel and source-only population. Focused synthetic tests cover frozen role semantics, native dataset construction and channel values. No real-data split, DG score or baseline qualification is established by these checks.
