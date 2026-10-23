# Local J1 execution commands

Base: PHMFactory `99725e6841ecb99e6c31aa9ca68e775a43dfe8a1`, remote dev verified on 2026-09-19.
Branch: `feat/tii-local-j1-20260919`. Commands below run from this child repository.

Environment correction: the author requested conda `LQ_signal` during implementation.
All subsequent Python/test/experiment commands must use
`conda run -n LQ_signal --no-capture-output ...`. Earlier plain `python` commands
below used `/home/user/anaconda3/bin/python` (base), not LQ_signal.

```bash
python -c 'import phmfactory; print(phmfactory.__file__)'
python -m phmfactory --help
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s test -p 'test_tii_target_head.py' -v
```

Import PASS: resolves to this `external/phmfactory/phmfactory/__init__.py`.
CLI help PASS. Head tests PASS: 11 tests, 0.297 seconds reported by unittest.
Fixtures are generated arrays, not industrial measurements. No GPU used.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s test -p 'test_tii_native_model.py' -v
```

PASS: six tensor-fixture model checks, 4.217 seconds reported by unittest.
Model construction uses `analyze_config('smoke', override_values=...)` and
`build_model`; this does not claim the default DG data adapter emits projected
common/incremental inputs or that a joint runtime YAML is already complete.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest test/test_per_sample_metadata.py -q -k 'two_by_two_data_model_factory_matrix_backpropagates and isfm'
```

PASS: two existing native ISFM/HSE Data Factory → Model Factory backward checks,
4.68 seconds; 14 unrelated tests deselected. No full-suite result claimed.

Base-environment shared Task tests: `python -m unittest discover -s test -p
'test_tii_joint.py' -v` with OMP/MKL threads=1 passed four tests (final run 0.542s).
Following identifier-helper extraction, head tests passed all 11 checks (0.030s).
These are generated-tensor checks, not the qualified real two-source micro-run.

Author-selected environment verification:

```bash
conda run -n LQ_signal --no-capture-output python -c 'import sys, torch, numpy, scipy, pandas, h5py, pytorch_lightning, phmfactory; print(sys.executable); print(phmfactory.__file__); print({"torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__, "h5py": h5py.__version__, "lightning": pytorch_lightning.__version__})'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n LQ_signal --no-capture-output python -m unittest discover -s test -p 'test_tii_*.py' -v
```

PASS: 21 tests, 3.161 seconds. Interpreter:
`/home/user/anaconda3/envs/LQ_signal/bin/python`; phmfactory resolves to this child.
Versions: torch 2.6.0+cu124, numpy 1.23.5, scipy 1.10.1, pandas 1.5.3,
h5py 3.13.0, pytorch_lightning 2.3.3. All tests used CPU tensor fixtures.

Setup failure: initial HTTPS submodule clone failed with curl 56 / GnuTLS (-9),
early EOF and index-pack failure. Its automatic full clone retry was terminated.
A local `git clone --no-checkout --no-hardlinks --separate-git-dir` copied only
committed Git objects from the existing checkout. Origin was restored to
`https://github.com/PHMbench/PHM-Vibench.git`; `git fetch origin dev` succeeded.
No dirty files from the old checkout were copied.

The failed clone removed its partial child Git directory. A subsequent relative
Git command consequently created the parent research branch at the unchanged
paper HEAD `ee64196`; parent file contents were unchanged. Child commands now
run with an explicitly verified child working directory.

## Native execution and final checks (LQ_signal)

All commands below ran from the child root; CPU tests/runs used OMP_NUM_THREADS=1
and MKL_NUM_THREADS=1. No packages were installed, and physical GPU 2 was never used.

```bash
conda run -n LQ_signal --no-capture-output python -m scripts.tii_make_fixture --output reports/tii_local_j1/native_fixture_v1
conda run -n LQ_signal --no-capture-output python -m phmfactory preflight --config reports/tii_local_j1/native_fixture_v1/native.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n LQ_signal --no-capture-output python -m phmfactory --config reports/tii_local_j1/native_fixture_v1/native.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n LQ_signal --no-capture-output python -m phmfactory --config reports/tii_local_j1/native_fixture_v1/native.yaml --override model.token_organization=support
/usr/bin/time -f 'wall_seconds=%e' conda run -n LQ_signal --no-capture-output python -m scripts.tii_verify_native_fixture --fixture reports/tii_local_j1/native_fixture_v1 --output reports/tii_local_j1
```

Both native fixture runs completed 20 updates. Trainer progress reported about
0.8 s/arm; end-to-end run time and training peak memory were not instrumented.
Recovery took 6.06 s wall, 0.716 s inside verifier, peak RSS 703020 KiB. Saved
source-validation scores precede the final float64 monitor correction; see
native_recovery.md. These are not real J1-C runs or target-query predictions.

Additional focused commands (each prefixed with `conda run -n LQ_signal
--no-capture-output python`):

| Command suffix | Result |
| --- | --- |
| `-m unittest test.test_tii_physical -v` | 4 PASS, 0.532 s |
| `-m pytest test/test_tii_evaluation.py -q` | 17 PASS, 35 subtests, 7.21 s |
| `-m pytest test/test_tii_target_head.py test/test_tii_joint.py -q` | 15 PASS, 21 subtests, 4.98 s after FixedEpisodeError/float64 changes |
| `-m pytest test/test_tii_checkpoint_tolerance.py -q` | 7 PASS, 5.44 s after float64 fix |
| `-m pytest test/test_tii_data.py -q` | 3 PASS, 2.93 s |
| `-m pytest test/test_classification_runtime.py::test_tii_iterations_preserve_source_rms_request_and_use_each_fitted_scale -q` | 1 PASS, 4.98 s |
| `-m pytest test/test_classification_runtime.py test/test_data_cache_contract.py test/test_trainer_lifecycle_contract.py test/test_trainer_lifecycle_schema.py -q` | 84 PASS, 5.45 s before later RMS deepcopy fix; its new regression passed separately |
| `-m pytest test/ -q` | FAIL collection, missing streamlit; 9.18 s; no complete-suite result |
| `-m scripts.gen_support_matrix` | PASS, generated authorities unchanged |
| `-m scripts.validate_docs` | PASS, 90 files |
| `-m scripts.validate_configs` | PASS, 12 configs |
| `-m phmfactory preflight --config configs/experiments/tii/m0_tensor_fixture.yaml` | PASS, CPU/1 device |
| `-m phmfactory demo` | PASS, existing Dummy train/selected-checkpoint/test path |

The earlier failed query test assertion is recorded in query_validation.md; it was not a successful
scientific run. All actual logs are retained. Source `git diff --check` passed;
the staged raw Lightning CSV/logs retain their original CRLF/trailing whitespace.

Data qualification command and resources are in DATA_QUALIFICATION_NOTES.md.
No paid API compute was used. Monetary local CPU cost was not measured. Raw
industrial signals remained immutable and local. J1-C and J2/N/H/J3/J3-R/J4,
real query statistics, episode sensitivity and result plots are NOT RUN.
