# PHMFactory v0.3 `phm-data-factory` Backend Decision

## Final v0.3.0 decision

`phm-data-factory` is **deferred to v0.3.1**. It is not included in, required by, or release-blocking for PHMFactory v0.3.0.

```text
v0.3.0 status: deferred_to_v0.3.1
included in v0.3.0: false
required for core runtime: false
backend gitlink allowed in v0.3.0: false
runtime import allowed in v0.3.0: false
```

The machine-readable authority is:

```text
docs/releases/v0.3.0-backend-deferral.yaml
```

The deny-by-default candidate record remains in:

```text
.github/phmfactory-v0.3-submodules.allowlist.yml
```

This preserves the future integration contract without advertising an unavailable backend as part of v0.3.0.

## v0.3.1 integration authority

The v0.3.0 deferral remains a historical release fact. The bounded v0.3.1
integration is now authorized against one public, immutable provider release:

```text
integration status: implementation_in_review
provider repository: https://github.com/PHMbench/phm-data-factory.git
provider release: v0.2.0
provider commit: 16180b5fd9ca31d79fe65efd29b11439c1e54186
provider package: phm-data-factory 0.2.0
provider API schema: 1.0.0
provider capability schema: 1.0.0
license: Apache-2.0
```

The provider repository is organization-owned and public, its `v0.2.0` tag
peels to the reviewed merge commit above, and its release publishes a wheel,
source archive, and checksums. This satisfies the ownership, licensing, and
immutable-source entry gates; the remaining gates belong to the bounded
PHMFactory adapter PR and its validation.

This authority does not make the backend part of the default installation and
does not authorize claims about real-dataset accuracy, throughput, live IoTDB
operation inside PHMFactory, or additional signal modalities.

## Why deferral is the correct v0.3 boundary

The reviewed backend source tree exists at:

```text
5580fafec2ea5615f6d3276d95e1e5a948cc0f13
```

but a final organization-owned, immutable target integration has not been accepted. Integrating it now would mix repository transfer, optional dependency packaging, adapter behavior, runtime boundaries, and release promotion in one late-stage change.

The core v0.3 paths do not require this component:

```text
import phmfactory
python main.py --help
python -m phmfactory --help
phmfactory --help
repo-shipped Dummy_Data smoke
provider-neutral CWRU bundle interface
core wheel build and clean installation
```

Deferral therefore reduces release risk without removing a validated future path.

## v0.3.0 repository contract

For v0.3.0 all of the following are required:

- `.gitmodules` is absent unless another separately approved submodule exists;
- `packages/phm-data-factory` is not a gitlink;
- no placeholder commit or branch-tracking entry is permitted;
- no personal-account URL or private SSH dependency is permitted;
- core code does not import the backend;
- selecting an unavailable backend must fail explicitly rather than silently falling back;
- no statement may claim backend integration, support, live IoTDB support, or performance evidence.

A valid deferral is non-blocking in both:

```bash
python tools/repo/check_submodule_policy.py --mode release
python tools/repo/check_release_readiness.py --mode release
```

The second command remains blocked by unrelated release conditions until they are resolved.

## v0.3.1 entry gate

A future integration must satisfy all of the following before the allowlist can move to `approved`:

```text
organization-owned public HTTPS repository
compatible explicit Apache-2.0 license
immutable reviewed commit
bounded adapter PR
no protected runtime rewrite
explicit missing-backend error
core CLI, wheel, Dummy and CWRU paths pass without backend initialization
```

Preferred target:

```text
https://github.com/PHMbench/phm-data-factory.git
```

If the repository is recreated instead of transferred, the replacement commit requires explicit tree-content parity evidence.

## Allowed future integration surface

A bounded v0.3.1 integration may review changes to:

```text
.gitmodules
packages/phm-data-factory
.github/phmfactory-v0.3-submodules.allowlist.yml
src/data_factory/phm_data_factory.py
src/data_factory/standalone.py
src/data_factory/__init__.py
src/config_schema/models.py
src/configs/config_utils.py
scripts/validate_docs.py
test/test_phm_data_factory_backend.py
test/test_validate_docs_scope.py
docs/PHM_DATA_FACTORY_BACKEND_V0_3.md
KNOWN_LIMITATIONS.md
```

The following remain protected and require a separate compatibility PR if a real defect is found:

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

## Import and failure contract for v0.3.1

The future adapter may lazily import an installed `phm_data_factory` package only when explicitly selected. It must not mutate `sys.path`, initialize a submodule automatically, or silently fall back to the existing backend.

Expected unavailable-backend behavior:

```text
phm-data-factory is optional and is not installed.
Install the approved backend and its required extra before selecting data.factory_name: phm_data.
```

## Claim boundary

For v0.3.0:

```text
backend_integrated: false
backend_supported: false
live_iotdb_supported: false
performance_claim_authorized: false
```

These boundaries are release facts, not temporary documentation language. Changing them requires new implementation and validation evidence in v0.3.1 or later.
