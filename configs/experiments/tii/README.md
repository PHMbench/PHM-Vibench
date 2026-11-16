# TII native fixture

This config exercises the public four-Factory path on constructed sine signals,
not industrial data. From the child repository root:

```bash
conda run -n LQ_signal --no-capture-output python -m scripts.tii_make_fixture --output /tmp/tii_new_fixture
conda run -n LQ_signal --no-capture-output python -m phmfactory preflight --config /tmp/tii_new_fixture/native.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n LQ_signal --no-capture-output python -m phmfactory --config /tmp/tii_new_fixture/native.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n LQ_signal --no-capture-output python -m phmfactory --config /tmp/tii_new_fixture/native.yaml --override model.token_organization=support
```

Use a new output directory; the generator refuses to overwrite existing runs.
The J1 qualification and gate records, including `native_fixture_v1`, were
relocated to the paper repository on 2026-11-16; do not recreate them under
`reports/` in this repository. `test/tii_m0_tensor_fixture.yaml` is the portable
form of the retained native_fixture_v1 input inventory, kept as a test asset.
It writes new runs under results/. It uses smaller K/P/D and
frequent validation for M0; it is not the frozen industrial primary budget.
Natural-acquisition mode additionally requires eligible source rows in an explicit
qualification CSV and record-level physical support evidence. The local data
qualification currently yields J1 No-Go; do not run industrial comparisons.
