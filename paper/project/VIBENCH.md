# VIBENCH 映射与一键复现（LLM_Explainable_FD_Toolkit）

Submission-readiness note: this file is the minimal PHM-Vibench mapping. The
canonical manuscript package, LLM evidence protocol, baseline/ablation gates,
and current blockers are defined in:

- `SUBMISSION_READINESS.md`
- `submission_prep/baseline_ablation_matrix.yaml`
- `submission_prep/ieee_trans_readiness.md`

## 1) 基本信息

- `paper_id`: `LLM_Explainable_FD_Toolkit`
- 主仓库版本（commit）：`b245d6d`
- submodule 版本（commit）：`08eb944`

## 2) 主仓库一键命令（唯一推荐入口）

配置文件（保存在本 paper submodule 内）：
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml`

最小可跑（建议先 1 epoch）。Accepted submission evidence must bind local RTX
4090 GPU `0` or `1` and must write artifacts inside this submodule:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.device=cuda --override model.device=cuda --override trainer.gpus=1 --override trainer.num_epochs=1 --override environment.seed=0 --override environment.output_dir=paper/UXFD_paper/LLM_Explainable_FD_Toolkit/results/vibench_min/gpu0_seed0
```

## 3) 说明（WP0 占位）

该 paper 的“LLM 增强解释”默认不启用网络；当前先落地 **LLM-free distillation**（TODO-only 结构化产物落盘）：

- `trainer.extensions.agent.enable: true`

产物（如启用）：

- `artifacts/distilled/summary.json`

For submission-readiness, the smoke output is not accepted until the run
directory also contains `run_meta.yaml`, `metrics.json`, logs, device metadata,
and the baseline/ablation evidence required by `SUBMISSION_READINESS.md`.

## 4) Current command-bound comparison surface

`submission_prep/baseline_ablation_matrix.yaml` records:

- PHM-Vibench proposed/no-agent smokes and six diagnostic baseline smokes;
- standalone template LLM pipeline and single-case demos;
- package-level template LLM smoke gates with non-accepted `run_meta.yaml` and
  `metrics.json` output when `--save` is used;
- non-accepted hallucination-checker, retrieval/context, and latency smoke
  runners in `experiments/scripts/run_llm_evidence_smoke.py`;
- missing accepted TOP, GPU, main-protocol evidence, and final evidence-bearing
  manuscript gates.

`manuscript/ieee_tii/main.tex` exists as a conservative IEEE compile
checkpoint. It is not submission-ready text until accepted `results/llm_evidence`
artifacts can support the manuscript claims.

The standalone and package-based demos are useful wiring checks only. They are
not accepted LLM evidence because they do not emit an accepted main-protocol
`results/llm_evidence/**/{run_meta.yaml,metrics.json}` package with GPU
metadata and same-prompt baseline/ablation coverage.

Smoke hallucination/context/latency probe:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_llm_evidence_smoke.py --condition all --output results/llm_evidence/demo_smoke --seed 0
```

This command is useful for checking artifact shape only. Its outputs are marked
`accepted_evidence=false`.
