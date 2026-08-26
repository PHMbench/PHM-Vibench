# User-experience review prompts

These prompts review the latest `dev` without modifying the repository. Use one prompt per
review group. Group 10 merges Groups 1–9 and decides whether another review cycle is
useful.

## Shared review contract

Prepend this contract to every group prompt:

```text
Review the latest dev branch of PHMbench/PHM-Vibench.

This is a review-only task. Do not modify code, open a PR, or invent execution results.
Every finding must cite a concrete file, function, command, test, or observed output.

Evidence states:
- VERIFIED: directly supported by current code or an executed counterexample.
- LIKELY_REQUIRES_TEST: plausible, but a small executable counterexample is still needed.
- UNRESOLVED: the available evidence does not decide the issue.

Priority:
scientific semantics > real user path > historical compatibility > engineering form.

Do not recommend:
- hash, checksum, digest, fingerprint, receipt, ledger, attestation, or evidence chains;
- silent fallback or automatic repair of scientific inputs;
- warning-and-continue behavior that drops samples, metrics, or checkpoints;
- FactoryManager, BackendManager, ResultManager, SchemaV2, a second Runtime, or a
  registry-of-registries;
- architecture for hypothetical future cases;
- Python wrappers whose only purpose is to hide a direct error or forward arguments.

Change order:
DELETE > INLINE > MERGE > SIMPLIFY > DOCUMENT > ADD.

For every issue report:
ID
Severity: P0 / P1 / P2 / P3
Evidence state
Evidence
User impact
Root cause
Smallest correction
Deletion opportunity
Forbidden overengineering
Acceptance command or counterexample

Distinguish a potential counterexample from a verified failure. Repeated wording of the
same root cause is not a new issue.
```

---

## Group 1 — Installation and first successful run

```text
Act as a first-time PHMFactory user and Python packaging reviewer.

Test or inspect the shortest CPU path:

wheel install
→ phmfactory --help
→ phmfactory doctor
→ phmfactory preflight --config smoke
→ phmfactory demo
→ locate result_dir, best_checkpoint, test_metrics, and run_summary

Review:
1. Whether normal dependency resolution is tested, rather than only `--no-deps`.
2. Whether the commands work outside the repository checkout.
3. Whether configs and Dummy data are packaged correctly.
4. Whether the default path imports optional tracking, UI, provider, or research packages.
5. Whether any command downloads data or models without an explicit request.
6. Whether `doctor` diagnoses only and leaves the environment unchanged.
7. Whether an unavailable requested CUDA device fails instead of switching to CPU.
8. Whether README commands can be copied exactly.
9. Whether success paths exist and primary metrics are finite.
10. How many PHMFactory-specific concepts a user must understand before first success.

Use a fresh CPU environment when execution is available. If not, mark installation
claims as UNRESOLVED rather than inferring success from source layout.

Output the shortest valid first-run path and at most three bounded repair PRs.
```

---

## Group 2 — CLI and configuration comprehension

```text
Act as a CLI and configuration-contract reviewer.

Review these public entrypoints:
- phmfactory
- phmfactory doctor
- phmfactory preflight
- phmfactory demo
- python -m phmfactory
- python main.py
- scripts.config_inspect

Check:
1. Every experiment-selection action requires one explicit config.
2. `--config` and the deprecated alias are mutually exclusive.
3. Base config, experiment YAML, explicit local config, and CLI override have one order.
4. Validate, inspect, preflight, and run use the same resolved mapping.
5. A maintained Runtime does not read YAML again.
6. Strings such as `"false"` and `"1"` are rejected rather than converted.
7. Data, device, epochs, checkpoint selection, cache, and evaluation behavior are visible.
8. Invalid configuration fails before Pipeline import, output creation, and Factory work.
9. Help text describes user actions rather than internal classes.
10. Public output does not expose identity digests that no decision consumes.

Search specifically for:
`getattr(..., default)`, `setdefault`, `bool(value)`, `int(value)`,
`str(value).lower()`, duplicate YAML loaders, implicit local files, and duplicate parsers.

Do not propose a ConfigManager or another config representation.
```

---

## Group 3 — Local data, metadata, cache, and split

```text
Act as a local-data user and Data Factory reviewer.

Use only tiny CSV files and arrays created in a temporary directory. Do not download a
public dataset.

Review:
1. The declared metadata path is the path actually read.
2. Missing metadata fails without provider calls.
3. File format is selected explicitly, not guessed after failure.
4. Reader exceptions retain their original type and cause.
5. None, empty, complex, non-finite, or wrong-rank reader outputs fail before cache publish.
6. Cache reuse is a visible strict boolean and never inferred from file presence.
7. Failed construction cannot publish a partial HDF5 cache.
8. No selected sample is dropped with warning-and-continue.
9. Label, domain, dataset, file, channel, and sample-rate metadata remain sample aligned.
10. Train, validation, and test are separated at the correct experimental-unit level.

Construct a minimal `stride < window_size` case and test raw-sample interval overlap.
Only call it leakage if the counterexample actually overlaps.

Prefer direct ID, group, interval, shape, and metadata checks. Do not recommend data or
split hashes.
```

---

## Group 4 — Replace-one-component contributor experience

```text
Act as a researcher adding one compatible component for the first time.

Simulate separately:
1. adding a reader;
2. adding a small model;
3. adding a Task;
4. adding a Trainer;
5. switching between two existing models by config only.

For each case record:
- files that must change;
- whether CLI or Pipeline changes are required;
- duplicate registry, CSV, decorator, symbol, or schema registrations;
- unrelated concepts the contributor must understand;
- the first failure location;
- the smallest offline smoke command;
- whether the contribution guide is sufficient and current.

Review the invariant:

Replace One Module
→ Change Only That Module and Config

Find wrappers that only forward arguments, Factories that mutate another Factory's
configuration, device movement outside Trainer ownership, silent duplicate registration,
and Factory construction that returns `None`.

Prefer deletion of duplicate registration paths. Do not propose PluginSpec or a universal
component manager.
```

---

## Group 5 — Training, checkpoint, and evaluation semantics

```text
Act as a machine-learning experiment-semantics reviewer.

Use small tensors, a Dummy loader, and one epoch where execution is available.

Check:
1. CE/NLL receive integer class indices with a valid ontology.
2. BCE target range and logit shape are unambiguous.
3. Regression losses preserve continuous targets.
4. Accuracy and F1 consume the intended representation.
5. AUROC receives scores or logits, not argmax labels.
6. Stateful metrics compute and reset over the complete evaluation population.
7. ModelCheckpoint and EarlyStopping consume the same explicit monitor and mode.
8. The selected best checkpoint is restored before test.
9. `training_only` and `evaluated` results cannot be confused.
10. Multiple test populations are represented explicitly or rejected.
11. Every metric declared in config is returned for every evaluated seed.
12. Optimizer and scheduler fields that change the trajectory are visible.

Do not change hyperparameters to make a broken contract pass. Do not introduce metric,
checkpoint, or scheduler managers.
```

---

## Group 6 — Result paths and repeated-run integrity

```text
Act as a user consuming results from one seed and from three repeated seeds.

Review:
1. One invocation owns one immutable result root.
2. `iter_0`, `iter_1`, and `iter_2` share the same parent.
3. Result directories are never silently reused.
4. A failed seed cannot publish aggregate success files.
5. Pipeline success requires a non-empty structured mapping.
6. Bool, string, list, tuple, empty mapping, and returned failure objects are rejected.
7. Direct output paths exist and do not require directory scanning.
8. `all_results.csv` and `run_summary.json` have one authority.
9. Every seed has the same reported key set.
10. Every declared metric has count equal to the configured number of iterations.
11. Means and uncertainty values are finite and statistically meaningful.
12. Result summaries contain scientific outputs, not unused identity digests.

Provide the expected directory tree and the smallest stable success mapping. Do not
recommend a ResultManager, artifact registry, or result manifest hierarchy.
```

---

## Group 7 — Error quality and fallback audit

```text
Act as a fail-fast adversarial reviewer.

Search maintained paths for:
- `except Exception`;
- warning followed by continue;
- `return None` on failure;
- `getattr(..., default)` for behavior fields;
- `setdefault`;
- `strict=False`;
- fallback, skip, ignore, backend switching, checkpoint globbing, or device switching.

Classify each catch:
A. necessary resource cleanup;
B. adds useful context while preserving the cause;
C. hides the original root cause;
D. changes the requested experiment;
E. converts failure into success.

C, D, and E require correction.

Test these cases:
- CUDA unavailable;
- reader failure;
- unknown Factory or component;
- missing metric;
- malformed sample;
- missing selected checkpoint;
- missing explicitly enabled optional package;
- conflicting config inputs.

Keep narrow cleanup in `finally` blocks. Do not propose an ErrorManager or a new exception
hierarchy unless a current caller genuinely needs a shared exception type.
```

---

## Group 8 — Documentation authority and stale information

```text
Act as a technical editor for a research software repository.

Review:
README.md
README_CN.md
CORE.md
docs/quickstart.md
docs/installation.md
configs/README.md
Factory README files
KNOWN_LIMITATIONS.md
CONTRIBUTING.md
TROUBLESHOOTING.md

Create one authority for each user question:
installation, first run, config editing, device selection, local data, replacing a
component, checkpoint, result directory, metrics, support status, and release status.

Find:
1. commands that no longer parse;
2. removed fields still shown in examples;
3. roadmaps written as current capability;
4. module docs that contradict CORE or current tests;
5. generated claims stronger than their source status;
6. empty marketing language, exaggerated completion claims, and decorative emoji;
7. stale TODO, review, Agent, and implementation-plan files in maintained directories;
8. pages that explain internal architecture before the user action.

Give every file one verdict:
KEEP / SHORTEN / MERGE / DELETE.

Git history is the archive. Do not create a new archive directory for obsolete plans.
Comments and prose should explain facts, invariants, boundaries, commands, and failures;
remove text that merely sounds comprehensive.
```

---

## Group 9 — Packaging, optional dependencies, and CI value

```text
Act as a packaging and test-economics reviewer.

Review:
pyproject.toml
requirements files
package data
top-level imports
doctor
wheel workflows
core quality gates
Dummy smoke
real-data workflows
UI workflows

Check:
1. A normal wheel installation resolves dependencies and passes `pip check`.
2. The installed command works outside the checkout.
3. Configs and Dummy files are in the wheel.
4. Tracking, UI, provider, database, and experimental model packages import only when
   explicitly selected.
5. Doctor covers the actual default smoke import graph without starting training.
6. CI executes a real fit → best-checkpoint restore → test path.
7. A mock Pipeline is not presented as proof of first-run success.
8. `--no-deps` is not presented as proof of normal installation.
9. The same expensive test is not repeated in unrelated workflows.
10. Real-baseline workflows trigger when their semantic dependency files change.
11. Formatting, generated docs, and directory audits do not dominate scientific tests.

Rank gates using:

probability of catching a real failure × impact / maintenance cost

Prefer deleting ritual or duplicate gates. Do not add a general CI manager.
```

---

## Group 10 — Meta-review and Occam convergence

```text
Input: reports from Groups 1–9.

Do not review the repository again.

Tasks:
1. Merge duplicate findings by root cause.
2. Remove findings whose evidence does not support the claim.
3. Separate verified failures, executable candidates, and unresolved evidence gaps.
4. Find the smallest set of shared root causes.
5. Assign one owner to every P0/P1.
6. Build a deletion-first implementation queue.
7. Propose no more than three sequential PRs.
8. Give each PR one invariant, one user-visible result, acceptance commands, out-of-scope
   items, and a one-commit rollback.
9. State which issues are deferred and why.
10. Decide whether another review cycle can produce new information.

Conflict rules:
scientific semantics > user path > historical compatibility
original error > wrapped error
one authority > synchronized copies
explicit failure > fallback
delete > new abstraction

Stop review when code-path, user-process, and scientific-semantic lenses produce no new,
independent, evidence-supported P0/P1 issue. Do not continue to satisfy a reviewer count.
```
