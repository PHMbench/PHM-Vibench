# PHMFactory v0.3 Release Readiness

This page is the blocking checklist for the final PHMFactory v0.3 release. It is an
audit contract, not a claim that the release is already ready.

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

## Repository-resolved requirements

The staged release-metadata change resolves the following repository-controlled
requirements:

```text
README branding                         RESOLVED
Chinese README branding                 RESOLVED
CITATION title and final repository URL RESOLVED
CHANGELOG v0.3.0 section                RESOLVED
RELEASE_NOTES_v0.3.0.md                 RESOLVED
v0.2 provenance                         RESOLVED WITHOUT FABRICATING A FINAL TAG
```

The v0.2 provenance record states that the earlier changelog was a Release Candidate,
that no final `v0.2.0` tag existed, and that commit
`a331769d4005018bc833534ecf4efeb5e8a5a78d` is the immutable pre-v0.3 migration
baseline rather than a retroactive final release.

## Remaining machine-checked blockers

The release remains blocked by:

1. `pyproject.toml` and `phmfactory.__version__` must change together from
   `0.3.0.dev0` to exactly `0.3.0` on the final release commit;
2. Hugging Face and ModelScope CWRU revisions must be immutable rather than `main` or
   `master`;
3. required CWRU `metadata.xlsx` and `RM_001_CWRU.h5` SHA-256 values must be populated;
4. GitHub Actions must run under the final `PHMbench/phmfactory` repository identity;
5. a `v0.3.0` tag must not exist before the release gate passes.

The checker also continues to verify the resolved README, citation, changelog, release
notes, version consistency, and v0.2 provenance requirements so they cannot regress.

## Human-reviewed blockers

The following cannot be inferred safely from repository files alone:

- all staged v0.3 PRs have been reviewed and merged in dependency order;
- dedicated Agent-content cleanup PRs are either merged or explicitly deferred with a
  documented ownership decision;
- branch protection and required checks are configured for the final default branch;
- public Hugging Face and ModelScope artifacts are available at the pinned immutable
  revisions;
- the two providers return byte-identical required bundle files;
- the optional `phm-data-factory` backend decision has been finalized and its
  integration branch rebased without reintroducing legacy submodules;
- the eight paper/research gitlinks have verified destination coverage or an explicit
  release deferral;
- the GitHub repository rename and redirect behavior have been verified;
- the final wheel and source distribution were built from the reviewed release commit;
- installation, CLI, module entrypoint, offline smoke, Pipeline 06, UXFD, Streamlit,
  dependency ownership, repository-layout, and CWRU gates all pass on the final
  repository identity.

## Release order

```text
1. review and merge the staged v0.3 PR graph in dependency order
2. merge or explicitly defer the dedicated Agent and paper/submodule cleanup work
3. publish the identical CWRU bundle to Hugging Face and ModelScope
4. pin provider revisions and populate required SHA-256 values
5. finalize the optional phm-data-factory backend decision
6. rename the GitHub repository to PHMbench/phmfactory and verify redirects
7. change versions from 0.3.0.dev0 to 0.3.0 on the final release commit
8. rerun all required checks under the final repository identity
9. build wheel and sdist from that reviewed commit
10. create the immutable v0.3.0 tag and publish the release
```

The repository rename, final version change, tag, and release publication must not
occur before their preceding blockers are cleared.

## Rollback

Before tagging, revert the release-preparation commit or keep the repository at
`0.3.0.dev0`. After tagging, publish a corrective release rather than moving,
deleting, or recreating the published tag.
