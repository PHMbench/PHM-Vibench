# Dummy Generative Demos

Runs the V0 PHM generative benchmark path on repo-shipped dummy data.

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml
python main.py --config configs/demo/10_generative/dummy_generative_rectified_flow.yaml
python main.py --config configs/demo/10_generative/dummy_generative_ddpm.yaml
python main.py --config configs/demo/10_generative/dummy_generative_score_sde.yaml
python main.py --config configs/demo/10_generative/dummy_generative_meanflow.yaml
python main.py --config configs/demo/10_generative/dummy_generative_drifting_flow.yaml
python main.py --config configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml
python main.py --config configs/demo/10_generative/dummy_generative_ot_nfm.yaml
```

These demos train minimal velocity models:

- Conditional Flow Matching
- Rectified Flow
- DDPM epsilon prediction
- Score-SDE denoising score matching
- MeanFlow/iMF experimental one-step flow
- Drifting Flow experimental one-step flow
- Transition Flow Matching experimental one-step flow
- OT-NFM experimental one-step flow

The conditions are explicit and traceable through metadata:

- `fault_label` from `Label`
- `domain_id` from `Domain_id`

Synthetic outputs are only benchmark-valid when produced through sample mode
with a manifest and leakage checks. The default train smoke is for runtime
validation.
