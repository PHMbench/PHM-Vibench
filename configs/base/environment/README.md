# `configs/base/environment/`

## 1) What This Block Controls

`environment` controls experiment identity and run-level settings:
- `project`, `seed`, `iterations`
- `output_dir` (base directory for artifacts)
- Uppercase keys are exported as process env vars by pipelines (e.g. `PROJECT_HOME`).

## 2) Minimal Example (YAML Snippet)

```yaml
environment:
  PROJECT_HOME: "."
  project: "demo_project"
  seed: 42
  output_dir: "results/demo"
  iterations: 1
```

## 3) Key Fields

| Field | Type | Notes |
|---|---:|---|
| `environment.project` | str | Used for logging/experiment naming. |
| `environment.seed` | int | Global seed. |
| `environment.output_dir` | str | Where `path_name()` places results (prefer relative). |
| `environment.iterations` | int | Repeat experiments with different seeds. |
| `environment.PROJECT_HOME` | str | Optional; exported as env var if uppercase. |

## 4) Typical Overrides

```bash
python main.py --config <yaml> --override environment.seed=0
python main.py --config <yaml> --override environment.iterations=1
python main.py --config <yaml> --override environment.output_dir=results/my_run
```

## 5) Coupling Notes

- `environment.output_dir` is consumed by `src/configs/config_utils.py:path_name`.
- Uppercase fields are exported by `src/Pipeline_01_default.py` (and other pipelines).

## 6) How to Extend

- Add new `environment.*` keys in YAML (allowed; schema is permissive).
- If you want the value exported to the process environment, use an uppercase key.

## 7) Common Pitfalls

1) Setting `output_dir` to an absolute `/home/...` path in shared configs (use local override).
2) Forgetting to set `project`, leading to confusing output folders.
3) `iterations` > 1 creates multiple `iter_*` folders; ensure you expect that.
4) Assuming `PROJECT_HOME` is consumed everywhere (it is only exported; usage is code-dependent).
5) Mixing local machine paths into base configs instead of `configs/local/local.yaml`.

