# Generative Model Card

## Model ID

## Module Path

## Prediction Target Compatibility

- [ ] velocity
- [ ] epsilon
- [ ] score
- [ ] average velocity
- [ ] drift

## Shape Contract

```text
input x: [N,C,L]
t: [N]
condition: fault_label [N], domain_id [N]
output: [N,C,L]
```

## Paper Role

- [ ] smoke only
- [ ] core baseline
- [ ] ablation
- [ ] exploratory appendix
- [ ] research-only

## Parameter Count

## Smoke Commands

```bash
python -m pytest test/generative/test_generative_model_forward.py
```

## Known Limitations
