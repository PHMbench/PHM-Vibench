# Dummy Generative CFM Demo

Runs the V0 PHM generative benchmark path on repo-shipped dummy data.

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml
```

This demo trains a minimal Conditional Flow Matching velocity model. The
conditions are explicit and traceable through metadata:

- `fault_label` from `Label`
- `domain_id` from `Domain_id`

Synthetic outputs are only benchmark-valid when produced through sample mode
with a manifest and leakage checks. The default train smoke is for runtime
validation.

