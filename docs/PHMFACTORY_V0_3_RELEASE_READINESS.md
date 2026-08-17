# PHMFactory v0.3.0-rc1 Release Readiness

This document is the blocking contract for the first PHMFactory v0.3 release candidate.
The source identity has been promoted to `0.3.0rc1` on `dev`; this does not claim that an
RC1 tag, GitHub Release, wheel upload, source-distribution upload, or package-index
publication already exists.

## Current status

```text
status: RC1_SOURCE_PROMOTED
release target: v0.3.0-rc1
current repository: PHMbench/PHM-Vibench
package version: 0.3.0rc1
release-readiness blockers: 0
current baseline_valid references: 1
RC1 tag present: false
published artifacts: false
```

The machine-checked result for the promoted source identity is:

```text
PHMFactory v0.3.0-rc1 readiness PASS: 0 blockers
```

Audit commands:

```bash
python tools/repo/check_submodule_policy.py --mode release
python tools/repo/check_release_readiness.py --mode audit
python tools/repo/check_release_readiness.py --mode release
```

## First-principles readiness contract

A release candidate is scientifically ready only when its maintained user path executes a
defined experiment rather than merely importing successfully:

$$
C_{\mathrm{RC1}}
=
C_{\mathrm{config}}
\land
C_{\mathrm{runtime}}
\land
C_{\mathrm{baseline}}
\land
C_{\mathrm{package}}
\land
C_{\mathrm{docs}}
\land
C_{\mathrm{repository}}.
$$

The factors mean:

- `C_config`: one resolved configuration is used from preflight through Pipeline execution;
- `C_runtime`: failures propagate from their source and no alternate algorithm is selected;
- `C_baseline`: at least one exact real-data configuration has a closed data, split,
  checkpoint, evaluation, metric, and repeated-run estimator contract;
- `C_package`: wheel/source build and clean installed entrypoints work;
- `C_docs`: user-facing claims match the generated support authority;
- `C_repository`: repository and optional-submodule boundaries remain explicit.

A file hash, receipt, ledger, or artifact index is not one of these scientific conditions.

## Machine-checked scientific reference

The RC1 authority requires exactly one reviewed registry row:

```text
id: baseline_01_mfpt_global_average_linear
config: configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
pipeline: Pipeline_01_Fault_Diagnosis
execution status: sanity_ok
protocol status: baseline_valid
```

The reference uses the public MFPT provider split, file-grouped and label-stratified
training/validation groups, a held-out provider test population, the transparent
`GlobalAverageLinear` model, best-checkpoint restoration, and explicit seeds 17, 18, and
19. Its low accuracy is retained as the honest result of a deliberately weak transparent
model. `baseline_valid` denotes protocol closure, not model superiority.

The checker also requires the reviewed preparation command, strict MFPT reader, focused
contract test, and real-data workflow to remain present. The workflow itself performs the
real download and end-to-end scientific validation; the release checker does not replace
that experiment with metadata bookkeeping.

## CWRU compatibility boundary

CWRU remains a compatibility bundle and a later local acceptance target. It is not the
current `baseline_valid` reference and it does not block unrelated RC1 progress.

The CWRU bundle contract is:

$$
C_{\mathrm{CWRU}}
=
C_{\mathrm{provider}}
\land
C_{\mathrm{schema}}
\land
C_{\mathrm{ID}}
\land
C_{\mathrm{shape}}
\land
C_{\mathrm{metadata}}.
$$

The executable validator checks:

```text
explicit provider and revision declaration
required metadata.xlsx and RM_001_CWRU.h5 mappings
required Dataset_id / Label / Domain_id fields
non-empty selector and Id field
unique selected Id values
selected Id -> HDF5 signal coverage
signal shape (L, C)
metadata sample-length agreement
metadata channel-count agreement
optional corpus foreign-key validity
```

RC1 does **not** require:

```text
per-file SHA-256 pins
cross-provider byte identity
hash-chain or receipt construction
artifact-integrity attestation
```

Those mechanisms may remain available as optional diagnostics for users who need them,
but they do not establish reader semantics, label correctness, split validity, or
benchmark validity.

## Resolved release areas

The following areas are complete and must not return as RC1 blockers:

- public `phmfactory` package, CLI, configuration resolver, and canonical Pipeline names;
- one configuration authority shared by inspect, preflight, CLI, and maintained runtime;
- fail-fast data population, task, device, objective, metric, and checkpoint semantics on
  maintained paths;
- deterministic maintained evaluation boundaries;
- strict Dummy and MFPT readers;
- 2 x 2 Data Factory x Model Factory replacement acceptance;
- one real MFPT `baseline_valid` reference;
- optional compatibility run records that cannot override Pipeline success or failure;
- P01-P09 migration, zero legacy gitlinks, and deny-by-default submodule policy;
- formal deferral of optional `phm-data-factory` integration to v0.3.1;
- synchronized package metadata and public `__version__` at `0.3.0rc1`;
- wheel/sdist construction, wheel inspection, clean installation, public entrypoints, and
  offline Dummy smoke on the RC1 source identity.

The backend decision authority remains:

```text
docs/releases/v0.3.0-backend-deferral.yaml
```

A valid deferral means the backend is absent, optional, not imported by the RC1 runtime,
not claimed as supported, and not release-blocking.

## Repository identity

The actual RC1 repository is:

```text
PHMbench/PHM-Vibench
```

Documentation and citation metadata must use this real URL. A future rename to
`PHMbench/phmfactory` is a product-governance decision, not a scientific-validity gate and
not an RC1 blocker. A rename, when authorized, requires its own bounded migration PR.

## Version promotion

The two source-version authorities now agree:

```text
pyproject.toml:          0.3.0rc1
phmfactory.__version__:  0.3.0rc1
```

The version-promotion PR passed:

```text
release readiness: PASS, 0 blockers
public package build and clean installation: PASS
offline Dummy smoke: PASS
core quality gates: PASS
CWRU bundle contract: PASS
dependency ownership: PASS
repository layout and submodule policy: PASS
```

The user-facing status synchronization PR must additionally rerun the public MFPT
three-seed workflow on the merged RC1 source identity.

Version promotion does not itself create a tag, GitHub Release, or package-index
publication.

## RC1 promotion state

```text
1. scientific-readiness authority merged                         DONE
2. pre-promotion audit isolated VERSION_NOT_RC1                  DONE
3. version-only RC1 promotion PR created                         DONE
4. 0.3.0.dev0 -> 0.3.0rc1 in both authorities                   DONE
5. release/public-package/core/repository gates rerun            DONE
6. release-readiness PASS with zero blockers                     DONE
7. user-facing status + MFPT revalidation on RC1 source          IN PROGRESS
8. tag or publish only under separate explicit authorization     NOT AUTHORIZED
```

## Final v0.3.0 boundary

The final `v0.3.0` release remains a later decision. Before a final tag, review:

- whether RC1 user feedback requires code or documentation corrections;
- whether the current repository name should remain or be changed;
- whether wheels and source distributions are built from the exact approved commit;
- whether supported combinations and known limitations remain accurate;
- whether any additional real-data baseline is necessary for the final claim boundary.

CWRU local acceptance may contribute to that review, but it must validate scientific data
semantics rather than substitute byte identity for experiment correctness.

## Rollback

Before tagging, a reviewed revert may return both version authorities to `0.3.0.dev0` if
the candidate is invalidated. After an RC1 artifact is published, do not move or recreate
the tag; issue a corrected release candidate instead.
