# PHMFactory v0.3 `phm-data-factory` Backend Contract

## Decision

PHMFactory v0.3 permits exactly one optional Git submodule exception:

```text
path: packages/phm-data-factory
repository target: https://github.com/PHMbench/phm-data-factory.git
license: Apache-2.0
status: blocked_pending_org_transfer
```

The backend is intended to remain available as an experimental, opt-in data access
adapter. It is not part of the default Dummy or CWRU software path.

The machine-readable authority is:

```text
.github/phmfactory-v0.3-submodules.allowlist.yml
```

## Current blocker

The reviewed backend source tree is currently available only under a personal-account
repository. No PHMbench- or AI4Engineering-L-owned repository was found during the
v0.3 review.

The reviewed source tree commit is:

```text
5580fafec2ea5615f6d3276d95e1e5a948cc0f13
```

The source package identifies itself as:

```text
distribution: phm-data-factory
version:      0.2.0
license:      Apache-2.0
```

Before integration, the repository must be transferred or content-verified into the
organization-owned target URL. PHMFactory does not add a personal-account URL to its
final `.gitmodules` file.

## Why PR #82 is not merged directly

The earlier experimental PR is useful implementation evidence, but it is not the final
v0.3 integration because it:

1. points `.gitmodules` at a personal-account repository;
2. is based on the old pre-cleanup repository topology;
3. would reintroduce historical `.gitmodules` context when rebased mechanically;
4. modifies protected `src/data_factory/data_factory.py` lifecycle behavior;
5. mutates `sys.path` to import an uninstalled submodule checkout;
6. combines ownership transfer, schema changes, runtime adapter code, documentation,
   and base-factory lifecycle changes in one PR.

The final integration must be rebuilt as a bounded PR on the current PHMFactory v0.3
stack.

## Required ownership migration

The backend repository must satisfy:

```text
public organization-owned HTTPS URL
Apache-2.0 license
immutable reviewed commit
no personal path or credential
no private SSH dependency
no paper or Agent runtime dependency
```

Preferred target:

```text
https://github.com/PHMbench/phm-data-factory.git
```

The reviewed source commit may remain reachable after a normal GitHub transfer. If the
backend is recreated instead, the replacement commit must have explicit tree-content
parity evidence.

## Final integration scope

A compliant backend integration PR may change only the minimum required surface:

```text
.gitmodules
packages/phm-data-factory                 gitlink
.github/phmfactory-v0.3-submodules.allowlist.yml
src/data_factory/phm_data_factory.py      isolated registered adapter
src/data_factory/standalone.py            optional import boundary
src/data_factory/__init__.py               explicit export/registration
src/config_schema/models.py               conditional backend fields
src/configs/config_utils.py                compatibility validation
scripts/validate_docs.py                   submodule traversal exclusion
test/test_phm_data_factory_backend.py
test/test_validate_docs_scope.py
docs/PHM_DATA_FACTORY_BACKEND_V0_3.md
KNOWN_LIMITATIONS.md
```

The exact changed-file list must be reviewed again after rebasing. Inclusion in this
list is permission to review, not permission to change behavior arbitrarily.

## Protected paths

The backend integration must not modify:

```text
src/data_factory/data_factory.py
src/data_factory/reader/**
src/data_factory/dataset_task/**
src/data_factory/samplers/**
src/model_factory/**
src/task_factory/**
src/trainer_factory/**
src/Pipeline_*.py
```

If a real defect requires changing a protected path, that change must be separated into
a dedicated compatibility PR with before/after runtime evidence.

## Import contract

The adapter may import the installed `phm_data_factory` package lazily when
`data.factory_name: phm_data` is selected.

It must not:

- prepend the submodule source directory to `sys.path`;
- import the optional backend during ordinary `phmfactory --help`;
- initialize the submodule automatically;
- silently fall back to the existing HDF5 backend;
- catch and hide an unavailable-backend error.

Expected failure when the backend is selected but unavailable:

```text
phm-data-factory is optional and is not installed.
Initialize packages/phm-data-factory and install the required backend extra.
```

## Optionality contract

All of the following must pass with the submodule uninitialized and the backend package
absent:

```text
import phmfactory
python main.py --help
python -m phmfactory --help
phmfactory --help
fully offline Dummy_Data smoke
Hugging Face CWRU bundle validation and quickstart
core wheel build and clean installation
```

Backend-focused CI may initialize and install the exact gitlink separately.

## Configuration contract

The experimental selection remains explicit:

```yaml
data:
  factory_name: phm_data
  phm_data_config: path/to/backend.yaml
  dataset_name: CWRU
```

Rules:

- `phm_data_config` is required when `factory_name == phm_data`;
- existing `data_dir` and `metadata_file` requirements remain unchanged for the
  default backend;
- there is no implicit provider or storage fallback;
- the backend does not redefine task filtering, splits, windowing, normalization,
  Dataset/DataLoader behavior, task logic, or trainer logic.

## Submodule policy

`tools/repo/check_submodule_policy.py` enforces:

- one allowlisted backend candidate;
- exact organization-owned target URL and path;
- no branch tracking;
- exact immutable gitlink after approval;
- no unknown new submodules;
- no return of already removed personal gitlinks;
- exact tracking of still-frozen legacy paper gitlinks;
- release blocking while legacy paper gitlinks remain or the backend is not approved
  and integrated.

Policy mode permits the known migration state while rejecting structural drift:

```bash
python tools/repo/check_submodule_policy.py --mode policy
```

Release mode is strict:

```bash
python tools/repo/check_submodule_policy.py --mode release
```

## Maturity and support

Initial status:

```text
experimental
synthetic contract evidence only
no maintained real-data performance claim
no live-IoTDB support claim
not part of the supported demo matrix
```

The backend can move to a supported status only through separate evidence and
`SUPPORTED_COMPONENTS.md` / `SUPPORTED_COMBINATIONS.md` review.

## Rollback

The final integration must be removable by reverting one bounded PR. Existing default
backends and maintained demos must remain unchanged when that PR is reverted.
