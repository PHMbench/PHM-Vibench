# CONFIG_ATLAS

> This file is generated from `configs/config_registry.csv`.

Re-generate:

```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
```

## Index
- [BASE](#base)
- [Pipeline_01_Fault_Diagnosis](#pipeline-01-fault-diagnosis)
- [Pipeline_02_Pretraining_Few_Shot](#pipeline-02-pretraining-few-shot)
- [Pipeline_06_Generative_Modeling](#pipeline-06-generative-modeling)

## BASE

### base_data

#### `base_data_classification`
- Path: `configs/base/data/base_classification.yaml`
- Description: 单数据集分类 / DG data base
- Owner code: `src/data_factory/__init__.py:build_data`
- Keyspace: `data.*`
- Minimal run: `python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- Common overrides: `data.num_workers=0`, `data.batch_size=16`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/data/README.md`
- Status: `/`

#### `base_data_cross_domain`
- Path: `configs/base/data/base_cross_domain.yaml`
- Description: 单数据集 cross-domain DG data base
- Owner code: `src/data_factory/__init__.py:build_data`
- Keyspace: `data.*`
- Minimal run: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`
- Common overrides: `data.num_workers=0`, `data.batch_size=16`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/data/README.md`
- Status: `/`

#### `base_data_cross_system`
- Path: `configs/base/data/base_cross_system.yaml`
- Description: 多系统 CDDG data base
- Owner code: `src/data_factory/__init__.py:build_data`
- Keyspace: `data.*`
- Minimal run: `python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml`
- Common overrides: `data.num_workers=0`, `data.batch_size=16`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/data/README.md`
- Status: `/`

#### `base_data_cross_system_fewshot`
- Path: `configs/base/data/base_cross_system_fewshot.yaml`
- Description: 跨系统 few-shot data base
- Owner code: `src/data_factory/__init__.py:build_data`
- Keyspace: `data.*`
- Minimal run: `python main.py --config configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml`
- Common overrides: `data.num_workers=0`, `trainer.num_epochs=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/data/README.md`
- Status: `/`

#### `base_data_fewshot`
- Path: `configs/base/data/base_fewshot.yaml`
- Description: 单系统 few-shot data base
- Owner code: `src/data_factory/__init__.py:build_data`
- Keyspace: `data.*`
- Minimal run: `python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml`
- Common overrides: `data.num_workers=0`, `trainer.num_epochs=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/data/README.md`
- Status: `/`

### base_environment

#### `base_env_default`
- Path: `configs/base/environment/base.yaml`
- Description: 通用 environment base（PROJECT_HOME + iterations）
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`
- Minimal run: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`, `environment.seed=0`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/environment/README.md`
- Status: `/`

### base_model

#### `base_model_global_average_linear`
- Path: `configs/base/model/global_average_linear.yaml`
- Description: Transparent temporal-mean linear classification baseline
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`
- Minimal run: `python main.py --config configs/demo/00_smoke/dummy_global_average_linear.yaml`
- Common overrides: `trainer.num_epochs=1`, `model.input_dim=2`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/base/model/README.md`, `src/model_factory/README.md`
- Status: `/`

#### `base_model_isfm_hse`
- Path: `configs/base/model/backbone_dlinear.yaml`
- Description: M_01_ISFM + E_01_HSE + B_04_Dlinear + H_01_Linear_cla
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`
- Minimal run: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `model.embedding=E_01_HSE`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/model/README.md`, `src/model_factory/README.md`
- Status: `/`

#### `base_model_tspn_uxfd`
- Path: `configs/base/model/tspn_uxfd.yaml`
- Description: TSPN_UXFD #61 core base; optional modules disabled by default
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`, `model.uxfd.*`
- Minimal run: `python main.py --config configs/demo/uxfd/20_smoke_tspn_uxfd_full_cpu.yaml`
- Common overrides: `trainer.device=cpu`, `trainer.devices=1`, `data.num_workers=0`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `src/model_factory/X_model/UXFD/FACT_TABLE.md`, `src/model_factory/X_model/UXFD/OPERATOR_CATALOG.md`
- Status: `/`

#### `base_model_xoan_operator_path`
- Path: `configs/base/model/xoan_operator_path.yaml`
- Description: P07 standalone typed executable operator-path software base
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`, `model.operator_path.*`
- Minimal run: `python main.py --config configs/experiments/p07_xoan_operator_attention/g030_executable_operator_path_smoke.yaml`
- Common overrides: `trainer.device=cpu`, `trainer.devices=1`, `data.num_workers=0`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/experiments/p07_xoan_operator_attention/README.md`
- Status: `/`

### base_task

#### `base_task_cddg`
- Path: `configs/base/task/cddg.yaml`
- Description: 多系统 CDDG 任务 base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`
- Minimal run: `python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml`
- Common overrides: `trainer.num_epochs=1`, `task.target_domain_num=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/CDDG/README.md`
- Status: `/`

#### `base_task_cddg_fewshot`
- Path: `configs/base/task/cddg_fewshot.yaml`
- Description: 跨系统 few-shot 任务 base（GFS 类型）
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`
- Minimal run: `python main.py --config configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml`
- Common overrides: `trainer.num_epochs=1`, `task.target_domain_num=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/GFS/README.md`
- Status: `/`

#### `base_task_classification`
- Path: `configs/base/task/classification.yaml`
- Description: 单数据集分类 / 简单 DG 任务 base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`
- Minimal run: `python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml`
- Common overrides: `trainer.num_epochs=1`, `task.lr=0.001`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/README.md`
- Status: `/`

#### `base_task_dg`
- Path: `configs/base/task/dg.yaml`
- Description: cross-domain DG 任务 base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`
- Minimal run: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `task.target_domain_id=[3]`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/DG/README.md`
- Status: `/`

#### `base_task_fewshot`
- Path: `configs/base/task/fewshot.yaml`
- Description: 单系统 few-shot 任务 base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`
- Minimal run: `python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml`
- Common overrides: `trainer.num_epochs=1`, `task.n_way=5`, `task.k_shot=5`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/FS/README.md`
- Status: `/`

#### `base_task_pretrain`
- Path: `configs/base/task/pretrain.yaml`
- Description: HSE / ISFM 预训练任务 base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`
- Minimal run: `python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- Common overrides: `trainer.num_epochs=1`, `task.lr=0.0005`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/pretrain/README.md`
- Status: `/`

### base_trainer

#### `base_trainer_default_single_gpu`
- Path: `configs/base/trainer/default_single_gpu.yaml`
- Description: 单 GPU 默认 Trainer base
- Owner code: `src/trainer_factory/__init__.py:build_trainer`
- Keyspace: `trainer.*`
- Minimal run: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `trainer.devices=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/trainer/README.md`, `src/trainer_factory/README.md`
- Status: `/`

### protocol

#### `p07_g040_protocol_preflight`
- Path: `configs/experiments/p07_xoan_operator_attention/g040_protocol.yaml`
- Description: P07-G040 standalone check-only protocol preflight; approval false and never claim evidence
- Owner code: `scripts/p07_protocol_preflight.py:main`
- Keyspace: `protocol.*`, `runtime.*`, `manifests.*`, `cwru.*`, `seeds.*`, `budgets.*`, `thresholds.*`
- Minimal run: `python scripts/p07_protocol_preflight.py --config configs/experiments/p07_xoan_operator_attention/g040_protocol.yaml --protocol-sha256 <sha256> --metadata-path <metadata> --raw-dir <raw> --reader-source-path <reader> --preprocessing-source-path <preprocessing>`
- Common overrides: `--device=cpu (default)`, `CUDA requires one of physical GPU 0 or 1`, `optional --emit-dir=<new-derived-dir>`
- Outputs: `stdout canonical JSON; explicit --emit-dir derived manifests only`
- Related docs: `configs/experiments/p07_xoan_operator_attention/README.md`
- Status: `/`


## Pipeline_01_Fault_Diagnosis

### baseline

#### `baseline_01_mfpt_global_average_linear`
- Path: `configs/baselines/01_mfpt/mfpt_global_average_linear.yaml`
- Description: MFPT official train/test, file-grouped validation, three-seed transparent baseline
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/global_average_linear.yaml`
  - task: `configs/base/task/dg.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/baselines/01_mfpt/mfpt_global_average_linear.yaml`
- Common overrides: `data.data_dir=/absolute/path/to/mfpt`, `environment.output_dir=/absolute/path/to/results`, `data.split.manifest_path=/absolute/path/to/results/split_manifest.json`
- Outputs: `results/baselines/mfpt_global_average_linear_v1/{experiment_name}/iter_{i}/`
- Related docs: `configs/baselines/01_mfpt/README.md`, `configs/base/model/README.md`
- Status: `sanity_ok`

### demo

#### `demo_00_smoke_dummy_dg`
- Path: `configs/demo/00_smoke/dummy_dg.yaml`
- Description: Smoke demo（repo 内置 dummy 数据，开箱即用）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/dg.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/00_smoke/dummy_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `trainer.devices=1`, `data.num_workers=0`
- Outputs: `results/demo/dummy_dg_smoke/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/00_smoke/README.md`
- Status: `sanity_ok`

#### `demo_01_cross_domain`
- Path: `configs/demo/01_cross_domain/cwru_dg.yaml`
- Description: Cross-domain DG demo（单数据集 DG 示例）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/dg.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/cwru_dg/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/01_cross_domain/README.md`
- Status: `sanity_ok`

#### `demo_02_cross_system`
- Path: `configs/demo/02_cross_system/multi_system_cddg.yaml`
- Description: Cross-system CDDG demo（多系统 CDDG 示例）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_system.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/cddg.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/multi_system_cddg/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/02_cross_system/README.md`
- Status: `sanity_ok`

#### `demo_03_fewshot`
- Path: `configs/demo/03_fewshot/cwru_protonet.yaml`
- Description: Few-shot demo（单系统 few-shot 示例）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_fewshot.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/fewshot.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/03_fewshot/README.md`
- Status: `sanity_ok`

#### `demo_04_cross_system_fewshot`
- Path: `configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml`
- Description: Cross-system generalized few-shot demo（GFS + DLinear/HSE）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_system_fewshot.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/cddg_fewshot.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/04_cross_system_fewshot/gfs_dlinear.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/cross_system_fewshot_dlinear/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/04_cross_system_fewshot/README.md`
- Status: `sanity_ok`

#### `demo_06_pretrain_cddg`
- Path: `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml`
- Description: Pretrain HSE for CDDG demo（单阶段 HSE 对比预训练视角）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_system.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/pretrain.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/pretrain_hse_cddg/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/06_pretrain_cddg/README.md`
- Status: `sanity_ok`

#### `demo_uxfd_full_cpu`
- Path: `configs/demo/uxfd/20_smoke_tspn_uxfd_full_cpu.yaml`
- Description: UXFD #61 full-module CPU smoke; no benchmark or claim status
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/tspn_uxfd.yaml`
  - task: `configs/base/task/dg.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `model.uxfd.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/uxfd/20_smoke_tspn_uxfd_full_cpu.yaml`
- Common overrides: `trainer.device=cpu`, `trainer.devices=1`, `data.num_workers=0`
- Outputs: `results/demo/uxfd/tspn_uxfd_full_cpu/{experiment_name}/iter_{i}/`
- Related docs: `src/model_factory/X_model/UXFD/FACT_TABLE.md`, `src/model_factory/X_model/UXFD/OPERATOR_CATALOG.md`
- Status: `needs_smoke`

### experiment

#### `p07_g030_xoan_operator_path_smoke`
- Path: `configs/experiments/p07_xoan_operator_attention/g030_executable_operator_path_smoke.yaml`
- Description: P07-G030 CPU software smoke; explicitly not C6-C9 evidence
- Owner code: `src/Pipeline_01_Fault_Diagnosis.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `model.operator_path.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/experiments/p07_xoan_operator_attention/g030_executable_operator_path_smoke.yaml`
- Common overrides: `trainer.device=cpu`, `trainer.devices=1`, `data.num_workers=0`
- Outputs: `results/experiments/p07/g030_executable_operator_path_smoke/{experiment_name}/iter_{i}/`
- Related docs: `configs/experiments/p07_xoan_operator_attention/README.md`
- Status: `sanity_ok`


## Pipeline_02_Pretraining_Few_Shot

### demo

#### `demo_05_pretrain_fewshot`
- Path: `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- Description: Pretrain + few-shot two-stage demo（当前为单阶段 HSE 对比预训练视角）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_classification.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/pretrain.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_02_Pretraining_Few_Shot.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/pretrain_hse_then_fewshot/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/05_pretrain_fewshot/README.md`
- Status: `sanity_ok`


## Pipeline_06_Generative_Modeling

### demo

#### `demo_10_generative_cfm`
- Path: `configs/demo/10_generative/dummy_generative_cfm.yaml`
- Description: Conditional Flow Matching CPU candidate smoke; promotion requires a fresh locked-dev E-chain
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_cfm.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_Generative_Modeling.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `trainer.devices=1`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_cfm/stage_ledger.json`
- Related docs: `configs/demo/10_generative/README.md`, `docs/PIPELINE_06_GENERATIVE_MIGRATION.md`
- Status: `needs_smoke`
