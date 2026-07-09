# Generative Test Gates

Use the smallest gate that proves the change.

## Fast Gates

Run these for pipeline wiring, manifests, and config/debug changes:

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m pytest test/generative/test_stage_ledger.py test/generative/test_manifest_validity.py test/smoke/test_preflight.py
```

Focused unit gates that do not require a full training run:

```bash
python -m pytest \
  test/generative/test_condition_sampling.py \
  test/generative/test_pipeline_06_contract.py \
  test/generative/test_sampling_manifest_metrics.py
```

## Paper And Evidence Gates

Run these when changing benchmark-effect reports, paperpack outputs, six-dataset
submission logic, metric evidence, or promotion readiness:

```bash
python -m pytest \
  test/generative/test_benchmark_effect.py \
  test/generative/test_paperpack_generative.py \
  test/generative/test_six_dataset_submission.py
```

Full generative tests may require optional ML dependencies from the project
environment. If they are unavailable, record the missing import explicitly
instead of weakening the gate.
