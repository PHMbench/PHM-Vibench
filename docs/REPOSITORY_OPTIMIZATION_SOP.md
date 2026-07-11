# Repository Optimization SOP

## Purpose

This SOP defines how PHM-Vibench should improve incomplete models, compatibility, code clarity, redundancy, tests, and open-source readiness without turning `main` into an integration dump.

The maintained public contract remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

with the five-block configuration model:

```text
environment / data / model / task / trainer
```

## Operating model

Repository optimization uses two separate lanes.

### Cloud governance lane

The cloud lane:

- audits branches, PRs, contracts, and maturity claims;
- creates small documentation, schema, governance, and migration PRs;
- decomposes large historical branches into reviewable vertical slices;
- assigns runtime-dependent work to a local implementation agent;
- reviews local evidence and recommends merge, revision, split, or rejection;
- never claims local runtime success without command evidence.

### Local implementation lane

The local lane:

- performs model, task, sampler, trainer, pipeline, and compatibility work;
- runs actual smoke tests, focused tests, config inspection, and complete gates;
- records environment, commands, exit codes, outputs, and limitations;
- completes or honestly demotes incomplete implementations;
- removes redundancy only when callers, registries, imports, and tests support removal;
- creates or updates a PR but does not merge its own work.

Private goal packs, execution prompts, reviewer prompts, logs, checkpoints, datasets, and generated results should remain outside Git unless a separate repository policy explicitly requires them.

## Core principles

1. Correctness before feature count.
2. Maintained claims must match implementation and tests.
3. A file, registry row, or smoke output does not prove a complete method.
4. Classic baselines and frontier methods need explicit maturity labels.
5. One PR should represent one coherent capability.
6. Compatibility fixes should be narrow and regression-tested.
7. Simplification must remove more ambiguity than it adds abstraction.
8. Private infrastructure must not become a core onboarding dependency.
9. External code requires provenance and license review.
10. No implementation agent self-merges.

## Main admission gates

Every runtime/config PR should report applicable results for:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
python -m scripts.validate_docs
python -m pytest test/ -q
```

A documentation-only PR may use a narrower gate, but its PR body must explain why runtime/config tests do not apply.

Report test states separately:

```text
passed / failed / skipped-with-reason / collection-error / not-run / not-applicable
```

Collection errors are failures, not skips.

## Priority classification

### P0 — blockers

- syntax/import failure in maintained code;
- broken `main.py --config` or five-block configuration contract;
- missing maintained registry/config target;
- security, credential, provenance, or data-leakage issue;
- destructive or unrelated-history integration;
- external code without license review.

### P1 — maintained capability failures

- maintained demo cannot run;
- registered model cannot be constructed through its factory;
- incorrect shape, device, dtype, checkpoint, loss, sampler, or metric behavior;
- unsupported compatibility claim;
- misleading maintained, paper-ready, benchmark-valid, or SOTA status.

### P2 — engineering debt

- mixed responsibilities and unclear ownership;
- exact or behaviorally redundant implementations;
- broad exception handling or silent fallback;
- missing focused tests;
- optional dependencies loaded as mandatory;
- stale or ambiguous documentation.

### P3 — enhancements

- new models;
- UI or branding work;
- performance optimization;
- broad architectural refactoring.

P0 and P1 take precedence over P3.

## Model completeness contract

For each maintained or newly exposed model, record:

```text
model_id
family
maturity
registry_target
implementation_file
factory_import
config
input_shape
output_shape
condition_keys
device_contract
dtype_contract
optional_dependencies
checkpoint_contract
focused_tests
maintained_demo
external_reference
license_status
known_limitations
```

The traceability chain is:

```text
registry
-> implementation
-> factory
-> config
-> shapes and conditions
-> focused tests
-> maintained demo
-> maturity claim
```

When an implementation cannot be completed within a bounded PR, remove or downgrade its maintained exposure rather than leaving a convincing-looking stub.

## Compatibility procedure

Review and test the dimensions relevant to the changed slice:

- documented Python minimum versus actual syntax and dependencies;
- PyTorch and PyTorch Lightning APIs;
- CPU and CUDA behavior;
- tensor layout, dtype, and device;
- metadata-derived channels/classes/domains;
- config precedence and legacy aliases;
- checkpoint format, map location, and state-dict prefixes;
- optional imports and compiled dependencies;
- offline, private-data, network, and GPU assumptions.

Avoid silent recovery for contract violations. Error messages should name the violated contract and received value.

## Redundancy removal procedure

Before deleting or consolidating code:

1. identify imports and public aliases;
2. identify factory and registry references;
3. identify configs, tests, and documentation consumers;
4. compare behavior, not only names;
5. add or run regression tests;
6. provide a migration note when public behavior changes.

Do not delete:

- generated configuration SSOT such as `docs/CONFIG_ATLAS.md`;
- `.github/` collaboration, security, or CI policy;
- compatibility wrappers still used publicly;
- files solely because their names appear similar.

## Generative model requirements

Generative work should keep train, sample, and eval as explicit stages. Benchmark-valid evidence requires, as applicable:

- training-source provenance;
- train-only normalization evidence;
- checkpoint requirement;
- condition-distribution provenance;
- generated-data manifest and hashes;
- sampler identity, NFE, seed, and output shape;
- duplicate/leakage checks;
- explicit `not_computable` metric status;
- explicit override for test-reference evaluation;
- downstream utility such as TSTR/TRTS before benchmark-valid claims.

Recent methods should remain `research-only` or `exploratory` until method-specific numerical and benchmark evidence is complete.

## Pull request structure

A good optimization PR contains:

- one coherent vertical slice;
- explicit scope and non-goals;
- changed-file grouping by concern;
- exact validation commands and observed results;
- compatibility and maturity changes;
- expected outputs;
- risks and rollback;
- remaining blockers and follow-up PRs;
- independent reviewer verdict when runtime behavior changes.

Recommended commit pattern:

```text
test: expose <contract> failure
fix: complete <model> runtime contract
refactor: remove verified duplicate <component>
docs: align maturity and compatibility claims
```

Do not use `git add .`; inspect and stage files explicitly.

## Independent review

Runtime/model PRs should be reviewed by an agent or person who did not implement the change. The reviewer should work read-only and challenge:

- public architecture contracts;
- implementation completeness;
- loss/sampler/metric semantics;
- compatibility and hidden fallback;
- train/validation/test leakage;
- code complexity and redundancy evidence;
- documentation and maturity claims;
- command evidence and unexplained skips.

Review verdicts:

```text
MERGE / REVISE / SPLIT / REJECT
```

Use `MERGE` only when applicable gates pass and no P0/P1 findings remain.

## Star-readiness standard

A high-trust, widely adopted repository needs more than model count. Optimization should improve:

- clear differentiated value;
- reproducible offline quickstart;
- honest classic/frontier maturity map;
- stable config and extension contracts;
- benchmark tables tied to code and evidence;
- citation, changelog, release, and provenance;
- contribution, governance, security, support, and issue/PR guidance;
- scoped good-first tasks;
- responsive, evidence-driven review;
- diagrams and examples that reflect current code.

Popularity cannot be guaranteed. The engineering target is a repository that earns trust through reproducibility, clarity, honest claims, and maintainable contributions.

## Stop conditions

Stop implementation and leave a precise draft report when:

- user changes would be overwritten;
- source provenance or branch ancestry is ambiguous;
- a change requires force push, unrelated-history merge, remote-branch deletion, or secret access;
- external-code licensing cannot be established;
- the selected slice expands into multiple independent architectures;
- required data/GPU access prevents meaningful validation and no offline contract can be built;
- P0/P1 reviewer findings cannot be resolved within the bounded scope.
