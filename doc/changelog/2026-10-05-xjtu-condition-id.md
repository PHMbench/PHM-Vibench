# Preserve XJTU operating-condition IDs

The historical XJTU metadata updater now assigns `40Hz10kN` to domain `2`,
instead of merging it with `37.5Hz11kN` as domain `1`. The documented three
conditions retain IDs `0`, `1`, and `2` and their original descriptions.

A focused temporary-CSV regression checks all three mappings, preservation of
non-XJTU rows, the written output, and an unchanged input file. No real metadata
was regenerated. Lifecycle label semantics are unchanged, and this correction
does not qualify XJTU for closed-set specimen-disjoint condition DG.
