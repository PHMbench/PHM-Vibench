# Pipeline 06 Generative Migration Contract

## Decision

`cleanup/repo-slim-2026-07-05` already contains a substantial PHM generative stack, including `src/Pipeline_06_generative.py`, generative models, tasks, samplers, metrics, manifests, smoke demos, and paper configs. However, that branch has no common ancestor with the current `main`; it must be treated as a source snapshot rather than merged directly.

The migration policy is:

```text
source snapshot -> file-level audit -> small PR -> validation -> promotion
```

Do not use `git merge --allow-unrelated-histories` for this migration.

## Architectural constraints

All migrated changes must preserve:

```text
python main.py --config <yaml> [--override key=value ...]
```

and the five-block contract:

```text
environment / data / model / task / trainer
```

Pipeline 06 remains factory-driven:

```text
YAML
-> main.py
-> Pipeline_06_generative
-> data_factory
-> model_factory/generative_model
-> task_factory/task/generative
-> trainer_factory
-> sampler
-> manifest
-> evaluation evidence
```

## Source-branch inventory and intended maturity

| Family | Intended maturity in `main` | Migration decision |
|---|---|---|
| Conditional Flow Matching | smoke-runtime, then benchmark-candidate | migrate first |
| Rectified Flow | exploratory-runtime | migrate after CFM |
| DDPM | exploratory-runtime | migrate after CFM |
| Score-SDE | research-only until sampler review | defer runtime promotion |
| UNet1D | supported backbone | migrate with backbone PR |
| DiT1D | supported experimental backbone | migrate with backbone PR |
| SSM/Mamba-style backbone | optional/research-only | keep optional and stateless |
| MeanFlow | research-only | experimental PR only |
| Drifting Flow | research-only | experimental PR only |
| Transition Flow Matching | research-only | experimental PR only |
| OT-NFM | research-only | experimental PR only |

`sanity_ok` means a smoke path completed; it does not mean benchmark-valid.

## Migration PR sequence

### G0 — contracts and audit documentation

Scope:

```text
docs/PIPELINE_06_GENERATIVE_MIGRATION.md
```

Gate:

```bash
python -m scripts.validate_docs
```

No runtime behavior changes.

### G1 — Pipeline 06 runtime shell

Scope:

```text
src/Pipeline_06_generative.py
minimum import dependencies only
focused import/preflight tests
```

Requirements:

- keep train, sample, and eval as separate invocations;
- do not advertise `paperpack` as a native pipeline mode unless dispatch supports it;
- do not modify the public `main.py --config` entrypoint;
- fail explicitly when a required checkpoint or artifact is missing.

Gates:

```bash
python -m py_compile src/Pipeline_06_generative.py
python -m pytest test/test_pipeline_06_generative_import.py -q
python -m scripts.validate_docs
```

### G2 — Conditional Flow Matching minimal vertical slice

Scope:

```text
CFM model
CFM task
velocity-matching loss
Euler ODE sampler
base model/task configs
repo-shipped dummy smoke demo
focused tests
```

Required flow:

```text
train -> checkpoint -> sample -> manifest -> eval -> evidence manifest
```

The first merge target remains exploratory; promotion requires complete evidence.

Gates:

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override data.num_workers=0
python -m pytest test/ -q
python -m scripts.validate_configs
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md
```

### G3 — manifests, leakage guards, and evaluation

Scope:

```text
normalization evidence
synthetic dataset manifest
stage ledger
nearest-neighbor leakage checks
generative evaluation metrics
evaluation evidence manifest
```

Validity rules:

- benchmark-valid samples must originate from the train split;
- validation/test data must not be used to estimate normalization parameters;
- test-reference evaluation requires an explicit override;
- missing metrics must be reported as `not_computable`, not silently dropped;
- generated samples without complete manifest and leakage evidence remain exploratory.

### G4 — DDPM and Rectified Flow baselines

Split this scope into separate reviewable commits or PRs if the diff becomes large.

Required baselines:

```text
DDPM epsilon prediction
DDPM reverse sampler/scheduler
Rectified Flow velocity target
reviewed ODE sampler reuse
```

Do not combine these baselines with frontier one-step methods.

### G5 — backbone expansion

Scope:

```text
UNet1D
DiT1D
optional stateless SSM/Mamba-style backbone
```

Requirements:

- every denoising/velocity call must be stateless;
- optional dependencies must fail explicitly when requested but unavailable;
- sampler time must not be confused with physical sequence time;
- backbone changes must not alter loss semantics.

### G6 — frontier one-step research methods

Scope:

```text
MeanFlow
Drifting Flow
Transition Flow Matching
OT-NFM
```

Required config status:

```yaml
task:
  generative:
    experimental: true
    validity_status: exploratory
    num_steps: 1
```

These methods must not be presented as benchmark-valid until method-specific losses, numerical guards, sampling semantics, leakage evidence, and downstream utility have been independently validated.

## Condition contract

V0 direct model conditions remain deliberately narrow:

```text
fault_label
domain_id
```

Operating variables are resolved through the domain map:

```text
domain_id -> load / rpm / system_id / sampling_rate
```

Do not claim direct RPM/load-controlled generation until these variables become explicit model inputs and receive dedicated tests.

## Promotion states

| State | Meaning |
|---|---|
| `docs-only` | design/reference material only |
| `research-only` | no supported benchmark runtime |
| `smoke-runtime` | local wiring and finite-output smoke path |
| `exploratory-runtime` | runnable but not benchmark-valid |
| `benchmark-candidate` | complete runtime awaiting evidence review |
| `benchmark-valid` | protocol, manifest, leakage, metric, and reproducibility gates pass |

Registry status, README maturity, manifests, and paper claims must use compatible terminology.

## Required evidence before benchmark-valid promotion

A model family cannot be promoted without:

1. resolved config and config hash;
2. dependency/environment lock hash;
3. dataset and domain-map provenance;
4. train-only normalization evidence;
5. checkpoint provenance;
6. explicit condition counts and sampling policy;
7. synthetic-data manifest;
8. leakage/duplicate checks;
9. signal, spectral, conditional, and downstream-utility metrics;
10. fixed seeds and repeatable train/sample/eval commands;
11. finite-value and shape tests for loss and sampler;
12. no silent use of test data during model selection or evidence construction.

## Non-goals

This migration does not:

- merge the unrelated cleanup history;
- rename the existing `configs/demo/06_pretrain_cddg` demo;
- make frontier models default baselines;
- vendor external implementations without license review;
- combine train, sample, and eval into a hidden monolithic command;
- claim paper readiness from smoke tests alone.

## Completion criterion

Pipeline 06 is considered integrated into `main` only when the CFM vertical slice can run through the public config-first entrypoint and produce an auditable chain:

```text
resolved config
-> train evidence and checkpoint
-> generated samples
-> synthetic manifest
-> evaluation metrics
-> evaluation evidence manifest
```

All later model families must reuse this contract rather than create parallel pipeline or artifact systems.
