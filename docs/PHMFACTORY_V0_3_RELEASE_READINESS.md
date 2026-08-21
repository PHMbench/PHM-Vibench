# PHMFactory v0.3.0-rc1 Release Readiness

This page is the current release-claim authority for the `0.3.0rc1` source tree. It does
not claim that an RC1 tag, GitHub Release, wheel upload, source-distribution upload, or
package-index publication exists.

## Current status

```text
source version: 0.3.0rc1
repository: PHMbench/PHM-Vibench
release state: BLOCKED
current baseline_valid references: 0
published artifacts: none
```

The current blocker is intentional:

```text
BASELINE_VALID_REFERENCE_INVALID
```

The MFPT transparent reference was validated on an earlier source state. Subsequent
changes modified metric lifecycle, checkpoint selection, and repeated-run aggregation.
Until the exact MFPT protocol is rerun on the current source, it remains `smoke_only` and
must not be presented as current `baseline_valid` evidence.

## Readiness contract

A release candidate is ready only when:

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
C_{\mathrm{docs}}.
$$

- `C_config`: inspect, preflight, and run resolve the same visible experiment.
- `C_runtime`: failures remain failures; no alternate data, model, task, device, loss,
  checkpoint, or estimator is selected silently.
- `C_baseline`: at least one exact real-data configuration has current-source evidence for
  its data population, split, objective, checkpoint selection, evaluation, declared
  metrics, and repeated-run estimator.
- `C_package`: the supported installation and offline first-run path work.
- `C_docs`: public claims match the current registry and generated support tables.

A hash, receipt, ledger, attestation, or artifact index is not a scientific readiness
condition.

## What is already established

The current source retains the following reviewed behavior:

- the public `phmfactory` command and configuration-first execution path;
- explicit Data, Model, Task, Trainer, and Pipeline responsibilities;
- fail-fast task, device, objective, checkpoint, and evaluation boundaries on maintained
  paths;
- deterministic maintained HSE validation/test behavior;
- strict local Dummy and MFPT reader contracts;
- one fully offline Dummy first-run path;
- current bounded smoke configurations listed in `SUPPORTED_COMBINATIONS.md`;
- optional `phm-data-factory` integration deferred outside the v0.3 core runtime.

These facts support software use and bounded smoke claims. They do not substitute for a
current real-data `baseline_valid` experiment.

## MFPT requalification gate

The candidate configuration remains:

```text
configs/baselines/01_mfpt/mfpt_global_average_linear.yaml
```

It may return to `protocol_status=baseline_valid` only after the unchanged protocol is run
on the current source and all of the following hold:

```text
provider revision and 20-file population unchanged
provider test files excluded from fit/validation/checkpoint selection
seeds exactly 17, 18, 19
one best checkpoint restored before each test
declared acc and f1 reported for every seed
non-empty finite metrics
count=3 for every repeated-run metric
finite mean and sample standard deviation
independent workflow-only recomputation agrees with framework accuracy and macro-F1
```

The requalification task must not change the data population, split, model, loss, metrics,
optimizer, epochs, or seeds to recover a preferred result. A negative result is evidence
and should leave the candidate unpromoted.

## CWRU and optional backends

CWRU is a later local acceptance target, not the current baseline claim. Its useful checks
are provider declaration, required metadata, unique IDs, signal coverage, shape, sample
length, channel count, labels, domains, and reader semantics. Per-file hashes or
cross-provider byte identity are optional diagnostics, not release gates.

`phm-data-factory` and IoTDB remain optional and deferred. The v0.3 core path must install,
preflight, and run its offline Dummy experiment without them. An unavailable optional
backend must fail when explicitly selected; it must not fall back to local data.

## Audit commands

```bash
python tools/repo/check_submodule_policy.py --mode release
python tools/repo/check_release_readiness.py --mode audit
python tools/repo/check_release_readiness.py --mode release
```

Expected current behavior:

```text
audit mode  -> reports the baseline-valid blocker
release mode -> exits non-zero
```

Release mode may pass only after a current-source `baseline_valid` registry row is restored
through reviewed execution evidence.

## Publication boundary

A future readiness pass still does not create a tag or publication automatically. Tagging,
GitHub Release creation, wheel/source upload, and package-index publication require
separate explicit authorization for the exact approved commit.

## Rollback

Each readiness change should be one bounded squash commit. Revert that commit if its
contract is wrong. Do not restore a `baseline_valid` claim merely to make the release gate
pass.
