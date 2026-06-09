# GOAL-GEN-M1-REPO-NATIVE: Repository-Native PHM Generative Materials Pack

## Goal ID

GOAL-GEN-M1-REPO-NATIVE

## Objective

Create a repository-native PHM generative benchmark guide, goal pack,
module-README materials pack, future-model reference map, and Codex-to-Claude
handoff protocol under the existing PHM-Vibench repository structure.

This is a docs/spec/templates-only goal. It must not implement runtime code.

## Why

PHM-Vibench needs a controlled path toward a paper-grade generative benchmark
for PHM raw-signal generation. Conditional Flow Matching is the V0 baseline,
but the repository must first define contracts, references, validation gates,
Speckit workflow, goal queue, and review handoff before adding new runtime.

Module-specific guidance belongs in the README next to the owning module.
Development-process guidance belongs under the active Speckit feature. Do not
create a central PHM generative docs pile under `docs/`.

## Current Facts To Verify

Before editing, inspect:

```bash
ls
ls src/model_factory/generative_model
ls src/task_factory/task/generative
ls src/task_factory/Components/generative
ls configs/paper/phm_generative
cat .specify/feature.json
sed -n '1,180p' main.py
sed -n '1,220p' .specify/memory/constitution.md
```

Codex must verify:

- `main.py` exists.
- The maintained entrypoint remains `python main.py --config <yaml>`.
- The maintained config model is five blocks:
  `environment / data / model / task / trainer`.
- `src/data_factory/`, `src/model_factory/`, `src/task_factory/`, and
  `src/trainer_factory/` exist.
- `src/task_factory/Components/` is the intended place for reusable losses,
  samplers, metrics, manifests, and regularizers.
- The active Speckit feature directory exists.
- Generative runtime, if present, uses existing factories and does not
  introduce a parallel runtime tree.

If any fact is false, stop and report `FACT_MISMATCH`.

## Speckit Workflow Contract

Future implementation should follow this order:

1. `$speckit-constitution`
2. `$speckit-specify`
3. `$speckit-clarify`
4. `$speckit-plan`
5. `$speckit-checklist`
6. `$speckit-tasks`
7. `$speckit-analyze`
8. `$speckit-implement`

For this M1 goal, the existing constitution is authoritative unless a fact
mismatch proves it stale. Do not amend the constitution just to duplicate rules
already present.

## Scope

Allowed to add or update module READMEs:

- `src/task_factory/task/generative/README.md`
- `src/model_factory/generative_model/README.md`
- `src/task_factory/Components/generative/README.md`
- `src/task_factory/Components/generative/losses/README.md`
- `src/task_factory/Components/generative/metrics/README.md`
- `src/task_factory/Components/generative/manifests/README.md`
- `src/task_factory/Components/generative/samplers/README.md`
- `configs/paper/phm_generative/README.md`
- `scripts/README.md`

Allowed to add or update active Speckit feature artifacts:

- `specs/<active-feature>/reviews/claude-team/<run-id>/TASK_SPEC.md`
- `specs/<active-feature>/reviews/claude-team/<run-id>/report.md`
- `specs/<active-feature>/reviews/claude-team/<run-id>/risks.md`
- `specs/<active-feature>/reviews/claude-team/<run-id>/test-log.md`
- `specs/<active-feature>/handoffs/<date>-goal-gen-m1-repo-native.md`
- `specs/<active-feature>/paper/README.md`

Allowed to modify:

- `docs/README.md` only to point readers to the module READMEs.
- `AGENTS.md` only to add a short pointer to the generative module READMEs if
  missing.
- `CLAUDE.md` only to add a short pointer to feature-scoped handoff/review
  artifacts if missing.

`.codex/` and `.claude/` may be used only as tool scratch or mirrors of
feature-scoped artifacts.

## Out Of Scope

Do not:

- Do not modify `main.py`.
- Do not modify runtime code under `src/`.
- Do not create `src/phm_factory/`.
- Do not create `docs/phm_generative/`, `docs/generative/`, `projects/`,
  `projects/phm_generative/`, `packs/`, top-level `templates/`, or top-level
  `schemas/`.
- Do not add `src/Pipeline_06_generative.py` if it is absent.
- Do not implement Flow Matching, Rectified Flow, DDPM, Score SDE, Mamba,
  MeanFlow, or Drifting runtime.
- Do not add dependencies.
- Do not generate synthetic data.
- Do not create checkpoints.
- Do not modify benchmark results.
- Do not launch Claude Code Teams automatically; prepare review materials only.

## Required Behavior

### Repository-Native Placement

The module README pack must state:

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

It must explicitly forbid `src/phm_factory/`, `docs/phm_generative/`,
`docs/generative/`, `projects/phm_generative/`, and `packs/`.

### Pipeline Contract

The generative task README must describe future modes:

- `train`
- `sample`
- `eval`

It must preserve this flow:

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

### Domain ID Contract

The generative task README must define direct model conditions:

```text
fault_label
domain_id
```

It must state that `load`, `rpm`, `system_id`, and `sampling_rate` are not V0
direct model condition keys. They are resolved through a domain map:

```text
domain_id -> load/rpm/system_id/sampling_rate
```

The README must include:

```csv
domain_id,load,rpm,system_id,sampling_rate,description,dataset_name,notes
0,0,1797,dummy_system_a,12000,"0hp 1797rpm",dummy,"example"
1,1,1772,dummy_system_b,12000,"1hp 1772rpm",dummy,"example"
```

### V0 Conditional Flow Matching

The generative loss README must define V0 as Conditional Flow Matching for raw
PHM signals `[N, C, L]`, conditioned on `fault_label + domain_id`, with
velocity prediction:

```math
z \sim \mathcal{N}(0,I), \qquad t \sim \mathcal{U}(0,1)
```

```math
x_t = (1-t)z + tx_1
```

```math
u_t = x_1 - z
```

Shape contract:

```text
x1/z/xt/pred_velocity: [N, C, L]
t: [N] or [N, 1, 1]
fault_label: [N]
domain_id: [N]
loss: scalar
```

### Future Model Blueprints

The generative model README must include docs-only or research-only outlines
for:

- Rectified Flow
- DDPM
- Score SDE
- Mamba/SSM backbone
- MeanFlow
- Drifting Models

Each entry must include status, loss target, prediction type, compatible
sampler, future runtime location, required tests before runtime, and maturity
gate.

Maturity labels:

- `docs-only`
- `research-only`
- `smoke-runtime`
- `exploratory-runtime`
- `benchmark-candidate`
- `benchmark-valid`

### External Code Policy

The generative model README must state:

- External code is reference-only.
- Do not copy external repository code.
- Do not vendor external code unless license-reviewed.
- Codex may use equations and interface ideas, but must reimplement minimal
  project-specific code from first principles.
- If official code cannot be verified, use `code_uncertain`; do not invent
  official repositories.

### Normalization Contract

The generative manifest README must state:

- Allowed V0 methods: `standardization`, `robust_scaler`, `none`.
- Recommended: `robust_scaler` or `standardization`.
- MinMaxScaler is not recommended as V0 default for PHM vibration generation.
- Synthetic manifests must later record normalization method, scope, params
  artifact, params hash, and whether inverse transform is required for
  physical-scale evaluation.

### Condition Injection Contract

The generative model README must recommend:

- sinusoidal time embedding for `t`
- embedding table for `fault_label`
- embedding table for `domain_id`
- FiLM or AdaLN injection
- no raw `torch.cat([x, condition], dim=1)` as the default

### FFT And Spectral Isolation

The generative loss and metric READMEs must state:

- V0 training loss is CFM velocity MSE only.
- FFT, log FFT, Hilbert envelope, envelope peak, and band-energy metrics are
  eval-only in V0.
- Do not add FFT loss to generative training in V0.
- Future spectral guidance must be a separate research goal and should prefer
  multi-scale STFT spectral convergence over direct full FFT loss.

### Synthetic Data Policy

The generative manifest README must require:

- generative train source split must be `train`
- synthetic data cannot train from real test split
- forbidden source splits include `val`, `valid`, `validation`, `test`, and
  `target_test`
- benchmark-valid requires manifest, protocol, config, seed, environment,
  normalization, condition-count, leakage, and metric-status evidence
- nearest-neighbor leakage check is mandatory before benchmark-valid

### Validation Gates

Module README guidance must include immediate docs/materials gates:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_docs
```

Future runtime gates may be listed but must not be implemented by this goal.

### GOAL-GEN Queue

The module/spec materials must define this order:

```text
GOAL-GEN-000  Create module README pack and feature-scoped strategy notes
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

### Claude Code Teams Package

Prepare, but do not launch, a Claude Code Teams review task spec:

- mode: `review`
- teammates: 3
- edits allowed: no
- target paths: module READMEs, `.specify/goals/v2`, and active Speckit feature
- out of scope: runtime code, secrets, push/deploy/delete
- required output: `report.md`, `risks.md`, `test-log.md`
- canonical path:
  `specs/<active-feature>/reviews/claude-team/<run-id>/TASK_SPEC.md`
- `.codex/claude-team-runs/<run-id>/TASK_SPEC.md` may exist only as a tool
  scratch copy or mirror.
- Subagent/teammate acceleration is allowed only for bounded, non-blocking
  read-only planning or review scopes. Codex remains lead-of-record, verifies
  outputs locally, and must not delegate urgent blocking work or overlapping
  runtime edits in this docs/spec goal.

### Handoff

Create a session handoff under the active Speckit feature directory, with an
optional `.claude/handoffs/` mirror, recording:

- active goal
- phase
- files changed
- runtime behavior changed: no
- validation commands
- known risks
- next steps

## Deliverables

This goal must deliver:

1. v2 goal pack under `.specify/goals/v2/`.
2. Future docs/materials requirements targeting module READMEs and active
   Speckit feature artifacts.
3. Claude Code Teams review package spec.
4. Handoff document.
5. Optional short docs pointers if missing.

## Acceptance Criteria

- No runtime code changed.
- `main.py` is not modified.
- No `src/phm_factory/` directory is created.
- No `docs/phm_generative/`, `docs/generative/`, `projects/`,
  `projects/phm_generative/`, or `packs/` directory is created.
- No top-level `templates/` or `schemas/` directory is created.
- All new goal files live under `.specify/goals/v2/`.
- All future module docs paths use module READMEs.
- CFM is identified as V0 baseline.
- MeanFlow and Drifting are research-only/demo-only.
- Mamba/SSM is backbone-only and stateless during flow/diffusion sampling.
- `domain_id` maps to load/rpm/system/sampling metadata.
- `load` and `rpm` are not V0 direct model condition keys.
- FFT/STFT/envelope metrics are eval-only in V0.
- External code is reference-only unless license-reviewed.
- Claude handoff format includes machine-parseable review tags.
- Validation commands are copy-paste runnable.

## Validation Commands

Run:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
eval "$(conda shell.bash hook)" && conda activate LQ_signal && python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_docs
find .specify/goals/v2 -maxdepth 1 -type f | sort
test ! -e src/phm_factory
test ! -e projects/phm_generative
test ! -e packs
test ! -e docs/phm_generative
test ! -e docs/generative
rg -n "src/model_factory/generative_model|src/task_factory/Components/generative/losses|fault_label|domain_id" .specify/goals/v2 src/model_factory/generative_model src/task_factory
```

If `scripts.validate_docs` fails due to unrelated pre-existing docs issues,
report the exact failure and run:

```bash
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml
```

## Failure Handling

If required current facts are false:

```text
FACT_MISMATCH:
- expected:
- observed:
- affected section:
- proposed correction:
```

If a requested file would create a forbidden structure:

```text
STRUCTURE_VIOLATION:
- forbidden path:
- reason:
- replacement path under module README or active Speckit feature:
```

If adding a file would require runtime code:

```text
SCOPE_VIOLATION:
- requested file:
- why it is runtime:
- recommended follow-up goal:
```

If validation cannot complete:

```text
VALIDATION_UNAVAILABLE:
- missing command:
- attempted alternative:
- result:
```

## Review Checklist

- [ ] Does this goal preserve `python main.py --config <yaml>`?
- [ ] Does it avoid runtime implementation?
- [ ] Does it avoid `src/phm_factory/`?
- [ ] Does it avoid `docs/phm_generative/`, `projects/phm_generative/`, and `packs/`?
- [ ] Does it use module READMEs for generated module materials?
- [ ] Does it place future generative models under `src/model_factory/generative_model/`?
- [ ] Does it place future losses under `src/task_factory/Components/generative/losses/`?
- [ ] Does it preserve `fault_label + domain_id` as V0 condition keys?
- [ ] Does it keep `load/rpm` behind the domain map?
- [ ] Does it keep FFT/spectral calculations out of V0 training loss?
- [ ] Does it mark MeanFlow and Drifting research-only/demo-only?
- [ ] Does it mark external code reference-only?
- [ ] Does it include Codex-to-Claude handoff requirements?
- [ ] Does it include validation gates?
- [ ] Does it define `GOAL-GEN-000` through `GOAL-GEN-004`?
