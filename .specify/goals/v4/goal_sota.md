# PHM-GenBench V4 SOTA Implementation Goals

Date: 2026-06-18

## Purpose

Implement every method in the V4 SOTA roster as repo-native PHM generative
code. A method may be exploratory in the paper, but it may not be only a
literature entry.

## Shared Implementation Contract

Every method goal must produce:

- task/loss/sampler code or a documented reason why an existing component is
  the faithful implementation
- maintained config under `configs/demo/10_generative/`
- registry/config integration where required
- focused tests for loss shape, finite values, sampler shape, and metadata
- method matrix entry for the paper
- explicit unsupported claims

Do not copy external code. Use primary papers for formulas and implement the
minimal repo-native version needed for PHM time-series experiments.

## GOAL-V4-SOTA-000-ROSTER-LOCK

Objective:
Lock the exact methods that V4 promises to implement.

Scope:
- Update this file only unless a missing method requires a separate goal.
- Keep the roster finite.

Required behavior:
- The roster must include CFM, Rectified Flow, DDPM, Score SDE, MeanFlow,
  Drifting, Transition Flow Matching, and OT-NFM.
- No method may be listed as V4 SOTA without an implementation goal.
- Methods that will not receive repo-native implementation must not be placed
  in the V4 SOTA roster.

Deliverables:
- Updated roster table in `.specify/goals/v4/goal.md`.
- Implementation mapping in this file.

Validation commands:

```bash
python -m scripts.validate_docs
rg -n "GOAL-V4-SOTA-(101|102|103|104|105|106|107)" .specify/goals/v4/goal_sota.md
```

## GOAL-V4-SOTA-101-BASELINE-CONTRACT

Owner:
S1 Baseline Contract

Objective:
Keep CFM, Rectified Flow, and DDPM as stable repo-native baselines while
protecting the shared generative task/model/config contracts used by all SOTA
methods.

Scope:
- `src/task_factory/task/generative/conditional_flow_matching.py`
- `src/task_factory/task/generative/rectified_flow.py`
- `src/task_factory/task/generative/ddpm_epsilon.py`
- shared generative losses/samplers/config schema only when required

Required behavior:
- CFM and Rectified Flow remain velocity/objective baselines with Euler
  sampling.
- DDPM remains epsilon prediction with explicit scheduler metadata.
- Shared model contract remains `model(x, t, condition) -> [N, C, L]`.
- Conditions remain explicit `fault_label` and `domain_id` fields.
- New schema fields needed by SOTA methods are minimal and typed.

Validation commands:

```bash
python -m scripts.validate_configs
python -m pytest test/generative
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
```

## GOAL-V4-SOTA-102-SCORE-SDE

Owner:
S2 Score SDE

Objective:
Implement a faithful Score SDE path, starting with a clearly declared VE/DSM
variant if full VP/VE support is too large for the first patch.

Scope:
- `src/task_factory/task/generative/score_sde.py`
- `src/task_factory/Components/generative/losses/score_sde.py`
- `src/task_factory/Components/generative/samplers/score_sde.py`
- `configs/demo/10_generative/dummy_generative_score_sde.yaml`
- focused Score SDE tests

Required behavior:
- Sample noise scale/sigma explicitly.
- Construct perturbed samples and score targets from the declared SDE variant.
- Record sampler metadata, stochastic parameters, and method status.
- Do not describe annealed Langevin smoke as full predictor-corrector evidence
  unless implemented and tested.

Validation commands:

```bash
python -m pytest test/generative -k "score_sde or generative"
python main.py --config configs/demo/10_generative/dummy_generative_score_sde.yaml --preflight-only
```

## GOAL-V4-SOTA-103-MEANFLOW

Owner:
S3 MeanFlow

Objective:
Replace the MeanFlow RF alias with a MeanFlow-specific one-step implementation
based on the paper's average-velocity objective.

Scope:
- `src/task_factory/task/generative/meanflow.py`
- new `src/task_factory/Components/generative/losses/meanflow.py`
- new `src/task_factory/Components/generative/samplers/meanflow.py` if needed
- `configs/demo/10_generative/dummy_generative_meanflow.yaml`
- focused MeanFlow tests

Required behavior:
- Do not reuse `RectifiedFlowLoss` while claiming MeanFlow fidelity.
- Keep sampler `num_steps=1` by default.
- If JVP or gradient terms are implemented, add finite-value tests and memory
  guard tests.
- Record `method_id=meanflow` and `status=repo_native_implemented` only after
  loss, sampler, config, and tests pass.

Validation commands:

```bash
python -m pytest test/generative -k "meanflow or generative"
python main.py --config configs/demo/10_generative/dummy_generative_meanflow.yaml --preflight-only
```

## GOAL-V4-SOTA-104-DRIFTING

Owner:
S4 Drifting and TFM

Objective:
Replace the Drifting RF alias with a method-specific drift-field objective and
one-step sampler metadata.

Scope:
- `src/task_factory/task/generative/drifting_flow.py`
- new `src/task_factory/Components/generative/losses/drifting_flow.py`
- new `src/task_factory/Components/generative/samplers/drifting_flow.py` if needed
- `configs/demo/10_generative/dummy_generative_drifting_flow.yaml`
- focused Drifting tests

Required behavior:
- Implement an explicit drift target, not a renamed RF velocity target.
- Keep dtype/device behavior stable.
- Mark paper status as exploratory unless later evidence supports stronger
  claims.

Validation commands:

```bash
python -m pytest test/generative -k "drifting or generative"
python main.py --config configs/demo/10_generative/dummy_generative_drifting_flow.yaml --preflight-only
```

## GOAL-V4-SOTA-105-TRANSITION-FLOW-MATCHING

Owner:
S4 Drifting and TFM

Objective:
Replace the Transition Flow Matching RF alias with a transition-path objective
that is distinct from straight-line Rectified Flow.

Scope:
- `src/task_factory/task/generative/transition_flow_matching.py`
- new `src/task_factory/Components/generative/losses/transition_flow_matching.py`
- new sampler component if needed
- `configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml`
- focused TFM tests

Required behavior:
- Define transition schedule and target explicitly.
- Keep physical sequence time separate from generative transition time.
- Record method metadata sufficient for the paper method matrix.

Validation commands:

```bash
python -m pytest test/generative -k "transition_flow_matching or generative"
python main.py --config configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml --preflight-only
```

## GOAL-V4-SOTA-106-OT-NFM

Owner:
S5 OT-NFM

Objective:
Implement a torch-native OT-NFM path with minibatch OT coupling and neural flow
map behavior. Do not use random pairing while calling it OT.

Scope:
- `src/task_factory/task/generative/ot_nfm.py`
- new `src/task_factory/Components/generative/losses/ot_nfm.py`
- new `src/task_factory/Components/generative/samplers/ot_nfm.py` if needed
- `configs/demo/10_generative/dummy_generative_ot_nfm.yaml`
- focused OT-NFM tests

Required behavior:
- Implement deterministic or seeded minibatch cost/coupling in torch.
- Avoid new SciPy/OT dependencies unless a separate dependency goal is approved.
- Add collapse diagnostics in sampler or manifest metadata.
- Keep implementation split-safe: OT pairing may not use validation/test data as
  synthetic source.

Validation commands:

```bash
python -m pytest test/generative -k "ot_nfm or generative"
python main.py --config configs/demo/10_generative/dummy_generative_ot_nfm.yaml --preflight-only
```

## GOAL-V4-SOTA-107-SOTA-SMOKE-MATRIX

Owner:
S6 Paper Integration

Objective:
Run and record a smoke/preflight matrix for every V4 roster method so the paper
can truthfully state what is implemented and what evidence exists.

Scope:
- configs under `configs/demo/10_generative/`
- registry/atlas docs if configs are promoted to maintained demos
- paper method matrix artifacts under `specs/002-phm-genbench-frontier/paper/`

Required behavior:
- Every roster method has a config path.
- Every roster method has smoke/preflight evidence or a blocking reason.
- The paper matrix uses statuses:

  ```text
  implemented
  smoke_passed
  exploratory_evidence
  blocked
  ```

- The matrix must not give roster methods a literature-only escape hatch.

Validation commands:

```bash
python -m scripts.validate_configs
python -m scripts.validate_docs
rg -n "dummy_generative_(cfm|rectified_flow|ddpm|score_sde|meanflow|drifting_flow|transition_flow_matching|ot_nfm)" configs/demo/10_generative .specify/goals/v4/goal_sota.md
```
