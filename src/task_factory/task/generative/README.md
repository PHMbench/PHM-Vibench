# Generative Tasks

This package contains PHM generative benchmark tasks. Module-specific
generative task guidance lives here rather than under `docs/`.

The public entrypoint remains:

```bash
python main.py --config <yaml>
```

## Task Contract

V0 supports Conditional Flow Matching for 1D vibration windows with `[N, C, L]`
signal tensors and explicit conditions:

- `fault_label`
- `domain_id`

The training contract is intentionally separate from fault classification
tasks. Generative training optimizes velocity matching only; FFT and
distributional metrics are evaluation signals, not training losses.

## Repository Placement

```text
Future pipeline:
  src/Pipeline_06_generative.py

Future models:
  src/model_factory/generative_model/

Future tasks:
  src/task_factory/task/generative/

Future losses:
  src/task_factory/Components/generative/losses/

Future samplers:
  src/task_factory/Components/generative/samplers/

Future schedulers:
  src/task_factory/Components/generative/schedulers/

Future metrics:
  src/task_factory/Components/generative/metrics/

Future manifests:
  src/task_factory/Components/generative/manifests/
```

Do not create `src/phm_factory/`, `docs/phm_generative/`, `docs/generative/`,
`projects/phm_generative/`, or `packs/` for module-specific generative
guidance.

Future task modes should preserve this flow:

```text
YAML config
-> main.py
-> Pipeline_06_generative
-> data_factory
-> model_factory/generative_model
-> task_factory/task/generative
-> task_factory/Components/generative/losses
-> trainer_factory
-> sampler
-> synthetic_data_manifest
-> generative_eval
```

## Domain ID Contract

V0 direct model condition keys are only:

```text
fault_label
domain_id
```

`load`, `rpm`, `system_id`, and `sampling_rate` are not direct V0 model
condition keys. They are resolved through a domain map for audit, grouping,
reporting, and paper analysis:

```text
domain_id -> load/rpm/system_id/sampling_rate
```

Required domain map columns:

- `domain_id`
- `load`
- `rpm`
- `system_id`
- `sampling_rate`

Optional domain map columns:

- `description`
- `dataset_name`
- `notes`

Example:

```csv
domain_id,load,rpm,system_id,sampling_rate,description,dataset_name,notes
0,0,1797,dummy_system_a,12000,"0hp 1797rpm",dummy,"example"
1,1,1772,dummy_system_b,12000,"1hp 1772rpm",dummy,"example"
```

Synthetic manifests that rely on a domain map must record:

- `domain_map_path`
- `domain_map_hash`

## Validity Policy

Synthetic data is `exploratory` unless the manifest, protocol, config,
normalization, leakage, and metric evidence chain is complete. Benchmark-valid
claims require source split `train`; forbidden synthetic source splits include
`val`, `valid`, `validation`, `test`, and `target_test`.

Nearest-neighbor leakage checks and explicit missing-metric reasons are required
before generated data can support benchmark-valid paper claims.

## Validation Gates

Immediate documentation/materials gates:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_docs
```

Runtime and paper goals may add stricter gates, but they must preserve the
public entrypoint and the five config blocks:
`environment / data / model / task / trainer`.

## Goal Queue

Execute one reviewable goal per PR:

```math
$$
\mathrm{PHM\text{-}GenBench}
=
\sum_i \mathrm{small\ verified\ goal}_i
$$
```

Use the queue as a controlled workflow: docs/materials goals define contracts,
demo-only goals stay isolated from benchmark claims, runtime goals touch one
factory slice at a time, paperpack goals consume completed evidence, and
research-only goals stay exploratory until a promotion goal passes.

```text
GOAL-GEN-000  Create module README pack and strategy docs
GOAL-GEN-001  Create domain_id mapping contract
GOAL-GEN-002  Create TaskFactory Components generative loss spec
GOAL-GEN-003  Create Codex-to-Claude handoff materials
GOAL-GEN-004  Create frontier model reference map
GOAL-GEN-005  Add Pipeline_06_generative skeleton
GOAL-GEN-006  Add model_factory/generative_model skeleton
GOAL-GEN-007  Add condition encoder and FiLM/AdaLN interface
GOAL-GEN-008  Add CFM loss unit tests
GOAL-GEN-009  Add phm_cfm_mlp1d smoke model
GOAL-GEN-010  Add Euler ODE sampler smoke test
GOAL-GEN-011  Add CFM training_step only
GOAL-GEN-012  Add synthetic_data_manifest writer
GOAL-GEN-013  Add generative_sample mode
GOAL-GEN-014  Add generative_eval smoke metrics
GOAL-GEN-015  Add paper-grade spectral/temporal/distribution metrics
GOAL-GEN-016  Add TSTR / augmentation utility protocol
GOAL-GEN-017  Add Rectified Flow baseline after CFM gates pass
GOAL-GEN-018  Add DDPM baseline after CFM gates pass
GOAL-GEN-019  Add stateless SSM/Mamba backbone after CFM gates pass
GOAL-GEN-020  Add MeanFlow / Drifting research-only notes
```

Do not implement `GOAL-GEN-005+` until `GOAL-GEN-000` through `GOAL-GEN-004`
are reviewed.
