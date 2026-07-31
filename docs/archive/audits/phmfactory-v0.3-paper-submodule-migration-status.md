# PHMFactory v0.3 Paper Gitlink Migration Status

## Policy

A paper gitlink is removable only when its destination repository contains complete
content-level evidence for the exact gitlink commit and the destination change has been
reviewed or retained.

Repository existence, a similar name, or a later working-tree snapshot is not enough.

The machine-readable authority is:

```text
docs/archive/audits/phmfactory-v0.3-paper-submodule-migration-status.yaml
```

Validation commands:

```bash
python tools/repo/check_paper_migration_status.py --mode policy
python tools/repo/check_paper_migration_status.py --mode release
```

Policy mode validates structure and evidence consistency. Release mode remains blocked
until every paper has `safe_to_remove: true` under the review rules.

## Current status

| Paper | Source gitlink | Destination | Coverage | Target review | Safe to remove |
| --- | --- | --- | ---: | --- | --- |
| foundation metric | `2dd7dabe...` | unresolved | not started | not started | no |
| P01 multimodal alignment | `b385b07e...` | `AI4Engineering-L/P01-UXFD-Multimodal-Alignment#2` | 89/89 | workflow approval required | no |
| P02 XFD benchmark toolkit | `379244dc...` | `AI4Engineering-L/P02-XFD-Benchmark-Toolkit#2` | 163/163 | workflow approval required | no |
| P03 evidence-grounded LLM XFD | `08eb944d...` | `AI4Engineering-L/P03-Evidence-Grounded-LLM-XFD#2` | 90/90 | workflow approval required | no |
| P04 physics-informed MoE | `0da06ae3...` | `AI4Engineering-L/P04-Physics-Informed-MoE-XFD` | not started | not started | no |
| P05 neuro-fuzzy safe XFD | `1bedd533...` | `AI4Engineering-L/P05-Neuro-Fuzzy-Safe-XFD` | not started | not started | no |
| P06 neural-symbolic XFD | `ad7dc2e2...` | `AI4Engineering-L/P06-Verifiable-Neural-Symbolic-XFD` | not started | not started | no |
| P07 operator attention | `20f47bac...` | `AI4Engineering-L/P07-XOAN-Operator-Attention` | not started | not started | no |

## Completed content coverage

### P01

```text
source blobs: 89
snapshot exact: 68
bounded overlay exact: 21
uncovered: 0
coverage manifest SHA-256:
82cf569c2a900a9a5f0931a179cc2a5965ba48cf1f1101f407748e9721614780
```

### P02

```text
source blobs: 163
snapshot exact: 142
immutable archive exact: 21
uncovered: 0
coverage manifest SHA-256:
05770099dfd98477b1eaca0a5a1f01db18595b763ef1b529f8f289c1ce61f3c0
```

### P03

```text
source blobs: 90
snapshot exact: 81
immutable archive exact: 9
uncovered: 0
coverage manifest SHA-256:
9c34faa184756cb7a2db5d85ba3bbabac4a7116f21afbd82f3c95230641e3883
```

The archives and overlays are provenance-only. They do not promote legacy metrics,
figures, checkpoints, or claims to verified PaperTrace evidence.

## Current blocker

The latest target heads returned `action_required` for their repository workflows, or
require equivalent target-side human review. Therefore all three completed coverage
items remain `safe_to_remove: false` and their PHMFactory gitlinks remain frozen.

## Update rule

Changing a paper to `safe_to_remove: true` requires all of:

```text
coverage_status: complete
uncovered_count: 0
target_review_status: target_ci_passed | target_reviewed | target_merged
valid target repository, PR, head, source hash, and coverage hashes
```

The actual gitlink deletion must occur in a separate bounded PR that changes only the
approved gitlink, its `.gitmodules` section, the tracker/allowlist state, and migration
audit documentation.
