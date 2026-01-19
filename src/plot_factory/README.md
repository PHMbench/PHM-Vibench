# plot_factory（离线绘图 TODO）

目标：把绘图逻辑从训练/模型内部剥离出来，优先基于稳定产物：

- `<run_dir>/artifacts/manifest.json`
- `<run_dir>/logs/**/metrics.csv`
- `<run_dir>/test_result_*.csv`
- `<run_dir>/artifacts/predictions.*`（如 `predictions.npz`）

推荐入口脚本：`python -m scripts.uxfd_postrun --config <yaml>`

