# PHMFactory v0.3.0 Repository Contract

Status: accepted for staged implementation  
Implementation base: `a331769d4005018bc833534ecf4efeb5e8a5a78d`  
Release target: `v0.3.0`

This document is the governing contract for the staged transition from
PHM-Vibench v0.2.x to PHMFactory v0.3.0. It supplements the existing
[Repository Optimization SOP](REPOSITORY_OPTIMIZATION_SOP.md). When an older
planning document conflicts with this contract, this contract controls the
v0.3.0 work.

## 1. Frozen names

| Surface | v0.3.0 name |
| --- | --- |
| Project display name | `PHMFactory` |
| GitHub repository name after the final rename PR | `phmfactory` |
| Python distribution | `phmfactory` |
| Python import namespace | `phmfactory` |
| CLI command | `phmfactory` |
| Root entrypoint | `python main.py` |
| Module entrypoint | `python -m phmfactory` |

Do not introduce `phm_factory`, `phm_vibench`, or another compatibility
namespace in v0.3.0.

## 2. Repository ownership and dependency direction

The allowed dependency direction is:

```text
paper repositories -----------+
personal forks ---------------+--> PHMFactory
third-party projects ----------+

PHMFactory --X--> personal forks
PHMFactory --X--> paper repositories
PHMFactory --X--> Agent-specific tooling
```

A dependency includes runtime imports, installation requirements, Git
submodules, required CI checkouts, required data/results, local paths, private
credentials, and release-time access.

Public, read-only publication links are allowed in maintained documentation as
long as deleting the link would not break installation, tests, demos, builds,
or releases.

### Approved submodule exception

PHMFactory may retain exactly one governed backend submodule for the
`phm_data_factory` backend. Its exact repository URL, path, pinned commit,
license, owner, and consumers must be recorded by the inventory PR before any
`.gitmodules` cleanup occurs.

The retained backend submodule must:

- use a public HTTPS URL rather than a personal SSH URL;
- be pinned to an immutable full commit SHA;
- have a verified compatible license;
- be listed in an explicit allowlist;
- not pull paper, Agent, personal, or generated-result content into the core;
- not make the default wheel, offline Dummy smoke, or maintained CWRU
  quickstart depend on personal credentials.

Every other current submodule is a migration candidate for the personal fork
or the appropriate paper repository. This contract does not authorize deleting
any submodule before source SHA and destination verification are complete.

## 3. v0.3.0 scope

v0.3.0 is a compatibility-first repository-boundary and public-interface
release. It may:

- add the public `phmfactory` package and CLI;
- retain root `main.py` as a thin dispatcher;
- add a single public configuration resolver;
- rename the six numbered Pipeline files directly to descriptive names;
- add an explicit Pipeline registry and legacy identifier aliases;
- consolidate the maintained Streamlit workspace under `apps/streamlit/`;
- provide a minimal CWRU quickstart through Hugging Face and ModelScope;
- move optional requirements into the subsystem that owns them;
- migrate personal, Agent, paper, and generated-result workspaces out of the
  upstream framework repository;
- add repository, provenance, and preservation gates.

## 4. v0.3.0 non-goals

v0.3.0 must not:

- rewrite dataset reader algorithms;
- move or rename `src/data_factory/reader/`;
- convert function-based readers into classes;
- merge readers because their filenames look similar;
- physically move the four existing factory trees merely for aesthetics;
- perform a whole-repository Black, isort, or rename pass;
- rewrite model, task, trainer, sampler, metric, checkpoint, or data-split
  semantics as part of cleanup;
- rename `test/` solely for convention;
- delete historical/config paths before runtime-reference and recovery checks;
- rewrite normal commit-author history.

Physical convergence of the protected runtime core requires a separate later
RFC and parity evidence.

## 5. Public package and protected runtime core

The transitional v0.3.0 architecture is:

```text
phmfactory/                    # new public package and CLI
    |
    +--> explicit adapters into the protected runtime core

src/                           # existing compatibility runtime core
├── data_factory/
├── model_factory/
├── task_factory/
├── trainer_factory/
└── Pipeline_*.py
```

The distribution must package both `phmfactory` and the required `src.*`
compatibility modules. New integrations should import `phmfactory`; existing
runtime internals remain available while v0.3.0 is supported.

The three entrypoints must share one dispatch implementation:

```bash
python main.py --config <yaml>
python -m phmfactory --config <yaml>
phmfactory --config <yaml>
```

## 6. Protected reader policy

`src/data_factory/reader/` is a protected compatibility surface. Reader parsing,
input keys, channel order, returned shape, dtype handling, numerical transforms,
and error behavior must not change in repository-cleanup PRs.

Allowed reader-adjacent cleanup is limited to separately reviewed non-runtime
material, such as generated bytecode, generated output, or personal paths in a
standalone `__main__` block. The reader callable AST or equivalent fingerprint
must remain unchanged unless a dedicated bug-fix PR is explicitly approved.

See [PHMFactory v0.3.0 Reader Preservation Contract](PHMFACTORY_V0_3_READER_PRESERVATION.md).

## 7. Pipeline migration policy

The existing Pipeline files will be renamed directly; v0.3.0 will not add six
wrapper modules. The intended canonical names are:

```text
Pipeline_01_Fault_Diagnosis
Pipeline_02_Pretraining_Few_Shot
Pipeline_03_Multitask_Pretraining_Finetuning
Pipeline_04_Unified_Evaluation
Pipeline_05_Explainable_Fault_Diagnosis
Pipeline_06_Generative_Modeling
```

The rename PR may change filenames, module references, configuration values,
documentation, and registry entries. It must not change the Pipeline callable's
training algorithm, seed policy, split policy, metrics, checkpoint behavior, or
factory construction. Legacy configuration identifiers may resolve through a
registry alias and emit a deprecation warning.

Because no old filename wrapper is retained, direct third-party imports of old
Pipeline module paths are a documented breaking change.

## 8. Configuration policy

v0.3.0 adds one public entrypoint such as:

```python
from phmfactory.config import resolve_config
```

The public resolver may delegate to existing internal configuration code while
parity tests lock current precedence and composition behavior. Physical deletion
or wholesale consolidation of the legacy configuration implementation is
outside the initial v0.3.0 cleanup scope.

The maintained public configuration model remains:

```text
environment / data / model / task / trainer
```

## 9. CWRU quickstart data contract

The maintained CWRU quickstart should use a small versioned prebuilt bundle:

```text
manifest.yaml
metadata.xlsx
RM_001_CWRU.h5
SHA256SUMS
LICENSES.md
corpus.xlsx              # optional in v0.3.0
```

`corpus.xlsx` must not be fabricated. Fault-diagnosis quickstart paths must work
without it; a feature that requires corpus data must fail with a clear missing-
capability error.

Hugging Face and ModelScope releases must pin revisions and provide identical
required bundle files by SHA256. v0.3.0 does not require a new raw-MAT reader
fixture; release claims must therefore distinguish verified prebuilt-HDF5
execution from raw-reader conversion coverage.

The fully offline `Dummy_Data` smoke remains mandatory and must not be replaced
by a network-dependent quickstart.

## 10. Requirements ownership

The root `requirements.txt` remains the core installation contract. Optional
incremental requirements belong to the subsystem that owns them, for example:

```text
apps/streamlit/requirements.txt
phmfactory/data_sources/modelscope/requirements.txt
test/requirements.txt
docs/requirements.txt
tools/plotting/requirements.txt
```

Dependency movement must be preceded by import and usage auditing. A dependency
must not be removed from the root merely because it appears optional by name if
maintained core imports still require it.

## 11. Migration and deletion controls

Paper, result, development, Agent, archive, and submodule content may be removed
from upstream only after the migration record includes:

```text
source path
source commit
source file hash when applicable
destination repository and path
destination verification
reviewer confirmation
safe-to-remove status
```

Normal commit-author history is preserved. History rewriting is allowed only
for verified secrets or credentials, with explicit credential rotation and a
separate approved procedure.

## 12. Pull-request execution model

The work must proceed through bounded draft PRs in the order recorded in
[PHMFactory v0.3.0 Task and PR Plan](PHMFACTORY_V0_3_TASK_PLAN.md).

Each PR must state scope, non-goals, protected paths, exact validation, observed
results, risks, and rollback. Runtime changes and repository cleanup must not be
mixed unless the task contract explicitly requires both.

No PR in this sequence is automatically merged. Completion of one PR does not
authorize the next destructive or runtime-affecting step without review.
