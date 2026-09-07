# P01 runtime components

Paper-specific scripts, configurations, protocol tests and result summaries are maintained in `AI4Engineering-L/P01-UXFD-Multimodal-Alignment`. That project mounts this repository as `phmfactory/`; PHMFactory does not import the paper project.

Runtime files retained here:

- `src/model_factory/X_model/P01OperatorBias.py`
- `src/model_factory/X_model/P01Reference.py`
- `src/task_factory/Components/p01_bias_losses.py`
- `src/task_factory/task/DG/p01_operator_bias.py`
- `src/data_factory/dataset_task/DG/p01_operator_bias_dataset.py`
- `test/test_p01_operator_bias.py`

Test from this checkout:

```bash
python -m pytest -q test/test_p01_operator_bias.py
```

Run paper experiments from the P01 checkout with `bash scripts/p01/run.sh demo results/p01_demo_01`, not from this repository. The project configuration is for its explicit paper-level driver, not the native five-block Pipeline schema.

The native task/dataset seam and original TSPN reference still require a complete local installation and real-data integration checks. Synthetic execution does not establish native Pipeline compatibility or diagnostic performance. PR #230 remains Draft until those checks pass.

The research branch includes the inspected `dev` base. A paper gitlink can pin this research revision without promoting it to a stable platform release. Push runtime changes before updating the parent project gitlink.
