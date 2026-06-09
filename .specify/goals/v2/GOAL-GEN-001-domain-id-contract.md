# GOAL-GEN-001: Domain ID Mapping Contract

## Goal ID

GOAL-GEN-001

## Objective

Create the PHM generative `domain_id` mapping contract in the corresponding
module README files, not under `docs/`.

## Why

For PHM generation, operating conditions such as load and rpm should be
represented through `domain_id` mapping, not passed as default direct model
conditions. The contract belongs next to the generative task and data-selection
modules that use it.

## Current Facts To Verify

Run:

```bash
sed -n '1,220p' src/task_factory/task/generative/README.md
sed -n '1,220p' src/data_factory/ID/README.md
sed -n '1,180p' main.py
```

Verify that `fault_label` and `domain_id` are the intended V0 condition keys.

## Scope

Allowed to add or update:

- `src/task_factory/task/generative/README.md`
- `src/data_factory/ID/README.md` only for cross-reference to domain metadata
  selection if needed
- `src/task_factory/Components/generative/manifests/README.md`

## Out Of Scope

- Do not implement data loader code.
- Do not modify runtime.
- Do not add `load` or `rpm` as direct model condition keys.
- Do not create separate docs-only schema/template files under `docs/`.

## Required Behavior

Define required domain map columns:

- `domain_id`
- `load`
- `rpm`
- `system_id`
- `sampling_rate`

Define optional columns:

- `description`
- `dataset_name`
- `notes`

Define direct model condition keys:

- `fault_label`
- `domain_id`

Define manifest evidence:

- `domain_map_path`
- `domain_map_hash`

Include this CSV example inline in the owning README:

```csv
domain_id,load,rpm,system_id,sampling_rate,description,dataset_name,notes
0,0,1797,dummy_system_a,12000,"0hp 1797rpm",dummy,"example"
1,1,1772,dummy_system_b,12000,"1hp 1772rpm",dummy,"example"
```

## Deliverables

- Domain ID contract section in the generative task README.
- Manifest evidence section in the generative manifest README.
- Optional data-factory cross-reference only if it clarifies domain metadata.

## Acceptance Criteria

- The contract states `fault_label + domain_id` are V0 conditions.
- The contract states `load` and `rpm` are not direct V0 model conditions.
- The contract states `domain_id` maps to load/rpm/system/sampling metadata.
- No central generative docs path is introduced under `docs/`.

## Validation Commands

```bash
python -m scripts.validate_docs
rg -n "fault_label|domain_id|domain_map_path|domain_map_hash" src/task_factory/task/generative src/task_factory/Components/generative/manifests
```

## Failure Handling

Report `SCOPE_VIOLATION` if implementation would require data loader changes.
Report `STRUCTURE_VIOLATION` if paths would create `docs/phm_generative/`.

## Review Checklist

- [ ] Does the contract define `fault_label + domain_id` as V0 conditions?
- [ ] Does the contract keep `load/rpm` behind `domain_id` mapping?
- [ ] Does the contract avoid runtime code requirements?
- [ ] Does the contract live in module README files?
