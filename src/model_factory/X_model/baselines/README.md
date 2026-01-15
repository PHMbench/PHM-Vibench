# X_model Baselines

Baseline models used for UXFD paper comparisons.

Rules:
- Keep only **model definition + forward + vibench input/output adaptation**.
- Do not include dataset code, trainers, or paper-specific scripts here.
- Avoid extra dependencies beyond the main repo `requirements.txt`.

Typically used via:
```yaml
model:
  type: "X_model"
  name: "BASE_ExplainableCNN"
```

