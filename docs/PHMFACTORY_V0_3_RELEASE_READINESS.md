# PHMFactory v0.3 Release Readiness

This document is the blocking contract for the final PHMFactory v0.3.0 release. It records the current tree, not an assertion that publication is already authorized.

## Current status

```text
status: BLOCKED
release target: v0.3.0
current repository: PHMbench/PHM-Vibench
target repository: PHMbench/phmfactory
package version: 0.3.0.dev0
```

Audit commands:

```bash
python tools/repo/check_submodule_policy.py --mode policy
python tools/repo/check_submodule_policy.py --mode release
python tools/repo/check_release_readiness.py --mode audit
python tools/repo/check_release_readiness.py --mode release
```

The submodule release policy must now pass. The overall release command remains non-zero while any final release finding exists.

## Resolved release areas

The following areas are complete and must not return as blockers:

- PHMFactory README and citation branding;
- public `phmfactory` package, CLI, configuration resolver, and canonical Pipeline names;
- explicit v0.2.0 release-candidate provenance anchored to `a331769d4005018bc833534ecf4efeb5e8a5a78d`;
- P01–P09 content-level migration evidence;
- Foundation 257-path partition with zero unassigned paths;
- removal of every legacy mode-160000 paper, personal, and research gitlink;
- removal of `.gitmodules` after the last gitlink was deleted;
- deny-by-default submodule policy with zero submodule release blockers;
- formal deferral of optional `phm-data-factory` integration to v0.3.1.

The backend decision authority is:

```text
docs/releases/v0.3.0-backend-deferral.yaml
```

A valid deferral means the backend is absent, optional, not imported by the v0.3.0 runtime, not claimed as supported, and not release-blocking.

## Remaining machine-checked blockers

The expected current finding set is exactly:

```text
2 x CWRU_REVISION_FLOATING
2 x CWRU_HASH_MISSING
1 x REPOSITORY_RENAME_PENDING
1 x VERSION_NOT_FINAL
```

Total expected findings:

```text
6
```

No other finding is authorized. In particular, these must remain absent:

```text
PHM_DATA_FACTORY_BACKEND_PENDING
BACKEND_DEFERRAL_INVALID
LEGACY_SUBMODULES_REMAIN
UNKNOWN_SUBMODULES_PRESENT
README_BRAND_PENDING
CITATION_BRAND_PENDING
CITATION_REPOSITORY_PENDING
V020_PROVENANCE_UNRESOLVED
```

## Meaning of the remaining blockers

### CWRU immutable publication

Both Hugging Face and ModelScope currently use floating revisions, and the logical `metadata` and `signals` SHA-256 values are not pinned. This is intentionally deferred from the present change set.

Release requires:

```text
immutable provider revisions
metadata SHA-256
signals SHA-256
byte-identical required files across providers
```

### Repository identity

GitHub still reports `PHMbench/PHM-Vibench`. The final release identity is `PHMbench/phmfactory`. The rename is intentionally deferred from the present change set.

### Final version

`pyproject.toml` and `phmfactory.__version__` must remain `0.3.0.dev0` until the CWRU and repository-identity gates are ready. The final promotion PR changes both to `0.3.0` exactly once, after the other release blockers are cleared.

`VERSION_NOT_FINAL` is therefore a finalization gate, not a request to publish final metadata early.

## Backend boundary

`phm-data-factory` is deferred to v0.3.1 and must not be added to the v0.3.0 tree.

For v0.3.0:

```text
backend gitlink: absent
runtime import: forbidden
silent fallback: forbidden
core dependency: false
release blocker: false
support claim: false
```

A future v0.3.1 integration still requires an organization-owned public repository, compatible license, immutable reviewed commit, bounded adapter PR, explicit missing-backend failure, and proof that core paths pass without backend initialization.

See [PHM_DATA_FACTORY_BACKEND_V0_3.md](PHM_DATA_FACTORY_BACKEND_V0_3.md).

## Final release order

```text
1. keep the merged migration and backend-deferral authorities unchanged
2. publish and verify the dual-source immutable CWRU bundle
3. update the CWRU manifest with immutable revisions and logical-key SHA-256 values
4. rerun release-readiness and confirm only rename/version remain
5. prepare the final promotion PR
6. rename the GitHub repository to PHMbench/phmfactory
7. change 0.3.0.dev0 to 0.3.0 in package metadata
8. update changelog and release-note status from pre-release to final
9. run full CI, wheel/sdist build, clean installation, CLI and smoke validation
10. require release-readiness PASS with 0 blockers
11. create tag v0.3.0 and publish the release
```

## Human review required before tagging

The final tagged commit still requires confirmation that:

- branch protection and required checks are correct for the final repository identity;
- Hugging Face and ModelScope expose the exact immutable bundle revisions;
- required CWRU files are byte-identical across providers;
- release notes accurately separate maintained software behavior from experimental or deferred components;
- wheel and source distribution are built from the exact tagged commit;
- clean installation, CLI entrypoints, offline Dummy smoke, Pipeline 06, UXFD, Streamlit, dependency ownership, repository layout, submodule policy, and CWRU validation all pass.

## Rollback

Before tagging, revert the final promotion commit or retain `0.3.0.dev0`. After publication, do not move or recreate the tag; issue a corrective release instead.
