# ISFM_Prompt Models

`ISFM_Prompt` provides prompt-style variants of ISFM for specific experiments (e.g. HSE-Prompt baselines).

## Config pattern (`model.type = "ISFM_Prompt"`)

Example:

```yaml
model:
  type: "ISFM_Prompt"
  name: "M_02_ISFM_Prompt"
  # prompt-specific configuration fields...
```

Unlike the standard ISFM family, current prompt implementations may handle components internally and may not always expose `model.embedding` / `model.backbone` / `model.task_head` as separate IDs. When this is the case:

- in the CSV registry, these fields should be set to `not_applicable`;
- prompt-specific hyperparameters should be documented here (to be added based on the implementation in `M_02_ISFM_Prompt.py`).

For now, please refer directly to `M_02_ISFM_Prompt.py` and `README_Simplified.md` for detailed arguments, and migrate the stable parts of that documentation into this README over time.

