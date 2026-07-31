# PHM-Vibench branch salvage map — 2026-07-28

> Read-only inventory. No branch was deleted, moved, force-pushed, or merged by this task.

- Total remote branches: **108**
- Main SHA: `a331769d4005018bc833534ecf4efeb5e8a5a78d`
- Dev SHA: `cf3c767b5d475308686b966d18a38eb26294bf7c`

## Decision counts

| Decision | Count |
|---|---:|
| `delete_after_ledger` | 20 |
| `hold_content_port_evidence` | 2 |
| `hold_physical_ancestry` | 7 |
| `hold_until_canonical_post_merge` | 1 |
| `manual_salvage_review` | 68 |
| `rebuild_from_dev` | 5 |
| `retain_permanent` | 2 |
| `review_validation_or_staging` | 1 |
| `salvage_by_vertical_slice` | 2 |

## Divergent branches requiring extraction

| Branch | Ahead of dev | Behind dev | Unique files | Required action |
|---|---:|---:|---:|---|
| `Feature_factory-update` | 36 | 207 | 413 | split into generative runtime, registry/config, smoke, evidence tools, and research-material disposition |
| `lq_merge_UXFD` | 12 | 213 | 66 | split into TSPN_UXFD runtime, configs/registry, NSN wrapper, artifact tools, and paper material |

## Rebuild-required PRs

`#35`, `#42`, `#79`, `#80`, `#81`, and `#83` must be rebuilt from current `dev`; their source branches are not direct merge candidates.

## Deletion gate

A branch may be deleted only after its head SHA, ancestry/content-port proof, superseding commit, operator, timestamp, and restore command are recorded in a deletion ledger.

The canonical branch `agent/v030-canonical-integration-r2` is the last v0.3 integration branch eligible for deletion.
