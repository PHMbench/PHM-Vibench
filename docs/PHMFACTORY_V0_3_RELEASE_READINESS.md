# PHMFactory v0.3 Release Readiness

This page is the blocking checklist for the final PHMFactory v0.3 release. It is an audit contract, not a claim that the release is already ready.

## Current status

```text
status: BLOCKED
release target: v0.3.0
repository target: PHMbench/phmfactory
```

Run the audit with:

```bash
python tools/repo/check_release_readiness.py --mode audit
```

The release command must remain blocked while any finding exists:

```bash
python tools/repo/check_release_readiness.py --mode release
```

Submodule policy can be inspected independently:

```bash
python tools/repo/check_submodule_policy.py --mode policy
python tools/repo/check_submodule_policy.py --mode release
```

## Machine-checked blockers

The checker currently evaluates:

1. `pyproject.toml` and `phmfactory.__version__` agree and equal `0.3.0`;
2. the public README heading uses `PHMFactory`;
3. `CITATION.cff` uses the PHMFactory title and final repository URL;
4. `CHANGELOG.md` contains a v0.3.0 section;
5. `RELEASE_NOTES_v0.3.0.md` exists;
6. Hugging Face and ModelScope CWRU revisions are immutable rather than `main` or `master`;
7. required CWRU bundle SHA-256 values are populated;
8. the governed `phm-data-factory` backend is organization-owned, approved, pinned, and present at the exact gitlink;
9. all legacy paper/research submodules have completed content-level migration and are absent from `.gitmodules`;
10. v0.2 provenance is resolved either by a visible historical tag or by the exact approved release-candidate provenance record;
11. no v0.3.0 tag already exists before the release gate passes;
12. when running in GitHub Actions, the repository has the final `PHMbench/phmfactory` identity.

## Resolved staged records

The staged v0.3 chain now includes:

- PHMFactory README and citation branding;
- a v0.3 changelog section and draft release notes;
- an explicit v0.2.0 release-candidate provenance authority at
  `docs/releases/v0.2.0-rc-provenance.yaml`;
- the immutable v0.2 runtime baseline commit
  `a331769d4005018bc833534ecf4efeb5e8a5a78d`;
- an explicit decision not to create a retroactive final v0.2.0 tag;
- a deny-by-default submodule allowlist with one neutral organization-owned backend target;
- an explicit decision not to merge the old personal-URL backend integration directly.

These records are only effective after their stacked PRs are reviewed and merged.

## Current submodule state

The `phm-data-factory` source tree has a reviewed Apache-2.0 commit, but its neutral
organization-owned repository transfer has not completed. The allowlist therefore
uses:

```text
status: blocked_pending_org_transfer
path: packages/phm-data-factory
target: https://github.com/PHMbench/phm-data-factory.git
```

The remaining paper gitlinks are still frozen because destination repository names
alone do not prove content coverage. Both conditions remain release blockers.

See [PHM_DATA_FACTORY_BACKEND_V0_3.md](PHM_DATA_FACTORY_BACKEND_V0_3.md).

## Human-reviewed blockers

The following cannot be inferred safely from repository files alone:

- all staged v0.3 PRs have been reviewed and merged in dependency order;
- branch protection and required checks are configured for the final default branch;
- public Hugging Face and ModelScope artifacts are available at the pinned immutable revisions;
- the two providers return byte-identical required bundle files;
- the backend organization transfer preserves the reviewed source tree or has verified replacement-tree parity;
- the final backend adapter remains optional and does not modify protected Data Factory behavior;
- every legacy paper gitlink has destination-level source/config/result verification;
- the GitHub repository rename and redirect behavior have been verified;
- release notes accurately separate compatibility guarantees from experimental features;
- the final wheel and source distribution were built from the tagged commit;
- installation, CLI, module entrypoint, offline smoke, Pipeline 06, UXFD, Streamlit, dependency ownership, repository-layout, submodule-policy, and CWRU gates all pass on the final release commit.

## Release order

```text
1. merge the reviewed v0.3 PR stack in dependency order
2. retain the approved v0.2 release-candidate provenance record
3. transfer and integrate the organization-owned phm-data-factory backend
4. complete content-level migration of the remaining paper gitlinks
5. publish and pin the dual-source CWRU bundle
6. finalize repository branding and citation metadata
7. change versions from 0.3.0.dev0 to 0.3.0
8. validate RELEASE_NOTES_v0.3.0.md against the final tree
9. rename the GitHub repository to PHMbench/phmfactory
10. rerun all required checks on the final repository identity
11. build wheel and sdist from the final commit
12. create tag v0.3.0 and publish the release
```

The repository rename, final version change, tag, and release publication must not occur before all blockers are cleared.

## Rollback

Before tagging, revert the release-preparation commit or keep the repository at `0.3.0.dev0`. After tagging, use a corrective release rather than moving or recreating the published tag.
