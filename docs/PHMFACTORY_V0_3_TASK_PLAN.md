# PHMFactory v0.3.0 Task and Pull-Request Plan

Status: accepted for staged implementation  
Execution rule: one bounded draft PR at a time

This plan decomposes the v0.3.0 transition into reviewable changes. A later PR
must not be started merely because an earlier branch exists; the earlier scope,
validation, and compatibility impact must first be reviewed.

## Global PR rules

Every PR in this program must include:

- exact base and head commits;
- scope and explicit non-goals;
- protected paths touched;
- changed-file grouping by concern;
- commands run and observed results;
- tests not run and the reason;
- compatibility impact;
- migration or provenance evidence when content is removed;
- risks and rollback;
- follow-up work that is intentionally deferred.

Do not combine repository cleanup, algorithm changes, broad formatting, and
release preparation into one PR.

## PR-01 — Freeze the v0.3.0 repository contract

Suggested title:

```text
docs(v0.3): freeze PHMFactory repository contract
```

Scope:

- record the final project/package/CLI names;
- record dependency direction and the single governed `phm_data_factory`
  submodule exception;
- freeze protected runtime and reader paths;
- record the direct Pipeline rename policy;
- record CWRU corpus-optional behavior;
- record the staged PR sequence.

Non-goals:

- no runtime changes;
- no deletion;
- no Pipeline rename;
- no `.gitmodules` edit;
- no repository rename.

Validation:

```bash
python -m scripts.validate_docs
git diff --check
```

If the documentation validator is unavailable in the execution environment,
report it as not run rather than claiming success.

## PR-02 — Baseline inventory and allowlists

Suggested title:

```text
chore(v0.3): record runtime, reader, and submodule baselines
```

Scope:

- record current main SHA and v0.2.x tag/release status;
- inventory every reader and current consumer;
- generate protected-path hashes and callable fingerprints;
- inventory all Git submodules;
- identify the exact `phm_data_factory` backend path, public URL, pinned SHA,
  license, owner, and consumers;
- introduce a submodule allowlist without deleting entries yet;
- inventory personal paths, Agent content, generated output, paper/results, and
  historical directories.

Non-goals:

- no reader modification;
- no submodule deletion;
- no paper/result deletion;
- no package rename.

Primary gate: the inventory must distinguish maintained, compatibility,
experimental, unverified, and placeholder readers.

## PR-03 — Remove generated and personal/Agent-only upstream content

Suggested title:

```text
chore(repo): remove generated and personal-only artifacts
```

Scope:

- delete tracked bytecode;
- classify `src/data_factory/reader/output/` before removal;
- preserve golden fixtures or paper evidence in the correct destination;
- migrate Agent files and personal development material to the personal fork;
- merge only accurate neutral knowledge into maintained documentation;
- clean approved personal paths in non-runtime development blocks;
- add generated-file and personal-information checks.

Protected-path gate: reader runtime callable fingerprints must remain unchanged.

## PR-04 — Add the public `phmfactory` package and CLI

Suggested title:

```text
feat(api): add phmfactory public package and CLI
```

Scope:

- add root-level `phmfactory/` public package;
- add `python -m phmfactory`;
- add `phmfactory` console entrypoint;
- convert root `main.py` into a thin dispatcher;
- package the required protected `src.*` compatibility runtime;
- verify clean wheel installation.

Non-goals:

- do not move the four existing factory trees;
- do not rename Pipelines in this PR;
- do not consolidate config internals.

Required parity:

```text
python main.py
python -m phmfactory
phmfactory
```

must share one parser and dispatch semantics.

## PR-05 — Rename Pipeline files directly

Suggested title:

```text
refactor(pipelines): adopt descriptive Pipeline names
```

Direct renames:

```text
Pipeline_01_default.py
  -> Pipeline_01_Fault_Diagnosis.py

Pipeline_02_pretrain_fewshot.py
  -> Pipeline_02_Pretraining_Few_Shot.py

Pipeline_03_multitask_pretrain_finetune.py
  -> Pipeline_03_Multitask_Pretraining_Finetuning.py

Pipeline_04_unified_metric.py
  -> Pipeline_04_Unified_Evaluation.py

Pipeline_05_default_w_explain.py
  -> Pipeline_05_Explainable_Fault_Diagnosis.py

Pipeline_06_generative.py
  -> Pipeline_06_Generative_Modeling.py
```

Scope:

- use `git mv` semantics;
- add an explicit registry;
- update module imports, maintained configs, docs, and registry values;
- retain legacy configuration identifiers through aliases and warnings.

Non-goals:

- no wrapper modules for old filenames;
- no Pipeline lifecycle merge;
- no algorithm, seed, split, metric, checkpoint, or factory-construction change.

Gate: Pipeline callable fingerprints must be unchanged except for approved
identifier/module-reference edits.

## PR-06 — Add one public configuration resolver

Suggested title:

```text
feat(config): add the public configuration resolver
```

Scope:

- add `phmfactory.config.resolve_config`;
- expose source files, overrides, selected Pipeline, warnings, and config
  fingerprint;
- route CLI and later Streamlit integration through the public resolver;
- add parity snapshots against current maintained configs.

Non-goals:

- no immediate deletion of `src/configs`, `src/config_schema`, or existing config
  utilities;
- no wholesale schema rewrite;
- no precedence change hidden inside cleanup.

## PR-07 — Add the versioned CWRU dual-source quickstart

Suggested title:

```text
feat(data): add the CWRU dual-source quickstart
```

Bundle requirements:

```text
manifest.yaml
metadata.xlsx
RM_001_CWRU.h5
SHA256SUMS
LICENSES.md
corpus.xlsx              # optional
```

Scope:

- add Hugging Face selective download;
- add an optional ModelScope provider with colocated requirements;
- pin provider revisions;
- verify required-file SHA256 parity;
- validate metadata/HDF5 IDs and signal shapes;
- add one non-interactive CPU quickstart;
- retain the offline Dummy smoke.

Non-goals:

- no new raw MAT fixture is required;
- no change to `RM_001_CWRU.read()`;
- no fabricated corpus;
- no full-dataset download.

Release notes must not claim raw CWRU conversion was revalidated by this PR.

## PR-08 — Reassign optional requirements

Suggested title:

```text
build(deps): colocate optional subsystem requirements
```

Scope:

- audit actual imports and consumers;
- retain root `requirements.txt` for core runtime and the default Hugging Face
  quickstart;
- place ModelScope, Streamlit, test, docs, plotting, and genuinely optional
  model/integration dependencies in their owning directories;
- add consistency checks;
- verify clean installs by profile.

Non-goal: do not move a dependency solely because its package name appears
optional while maintained core imports still require it.

## PR-09 — Consolidate Streamlit under `apps/streamlit/`

Suggested title:

```text
refactor(ui): retire the legacy app workspace
```

Scope:

- compare `app/`, `apps/streamlit/`, and the root launcher;
- migrate any unique maintained behavior;
- use the public CLI/config surfaces;
- retain only `apps/streamlit/` as the maintained UI;
- remove or deprecate the old launcher based on verified consumers.

Non-goals:

- no second training framework in the UI;
- no core requirement on Streamlit;
- no deletion before capability and import audits.

## PR-10 — Migrate paper, result, development, and disallowed submodule content

Suggested title:

```text
chore(research): migrate non-framework workspaces
```

Scope:

- move personal and Agent material to the personal fork;
- move paper code/configs/results/checkpoints to the owning paper repositories;
- migrate or classify reports, plots, archives, and development scratchpads;
- remove all disallowed submodules;
- retain only the governed `phm_data_factory` backend submodule;
- update `docs/publications.md` with non-runtime public references.

Deletion gate for every item:

```text
source path + source commit + hash where applicable
+ destination + destination verification
+ reviewer confirmation + safe-to-remove
```

## PR-11 — Resolve historical, case-collision, and navigation debt

Suggested title:

```text
docs(repo): consolidate historical and duplicate navigation
```

Scope:

- resolve case-colliding duplicate files after link audits;
- classify `configs/v0.0.9`, `docs/past`, `docs/v0.1.0`, and archive material;
- retain redirects or release evidence where needed;
- update documentation navigation;
- do not delete runtime-referenced legacy configs.

## PR-12 — Brand, changelog, CI, repository rename, and release

Suggested title:

```text
release: prepare PHMFactory v0.3.0
```

Scope:

- update README, citation, security/contribution links, release notes, and
  migration guide;
- update CHANGELOG from actual merged changes, not from the original plan;
- add final repository-layout, reader-freeze, personal-information,
  requirements, CLI-parity, data-provider, and cross-platform gates;
- rename the GitHub repository to `phmfactory` only after links and packaging are
  ready;
- publish `v0.3.0` after all required gates pass.

## Stop conditions

Stop and report instead of expanding scope when:

- a protected reader or Pipeline callable changes unexpectedly;
- a deletion lacks a verified destination or recovery path;
- the approved `phm_data_factory` submodule cannot be made public, immutable,
  licensed, and credential-independent;
- packaging cannot include the required compatibility runtime;
- a step requires force-pushing `main` or rewriting a published tag;
- an external-data license is unclear;
- a cleanup PR becomes an algorithm or architecture rewrite;
- validation is blocked and no honest narrower evidence is possible.
