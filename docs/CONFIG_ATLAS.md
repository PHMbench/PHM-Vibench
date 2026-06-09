# CONFIG_ATLAS

> This file is generated from `configs/config_registry.csv`.

Re-generate:

```bash
python -m scripts.gen_config_atlas --registry configs/config_registry.csv
```

## Index
- [BASE](#base)
- [Pipeline_01_default](#pipeline-01-default)
- [Pipeline_02_pretrain_fewshot](#pipeline-02-pretrain-fewshot)
- [Pipeline_06_generative](#pipeline-06-generative)

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
- Minimal run: `python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`
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
- Owner code: `src/Pipeline_01_default.py:pipeline`
- Keyspace: `environment.*`
- Minimal run: `python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`, `environment.seed=0`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/environment/README.md`
- Status: `/`

### base_model

#### `base_model_generative_cfm`
- Path: `configs/base/model/generative_cfm.yaml`
- Description: Generative PHM CFM velocity model base
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml`
- Common overrides: `model.hidden_dim=32`, `model.condition_dim=16`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/model/README.md`, `src/model_factory/generative_model/README.md`
- Status: `/`

#### `base_model_generative_dit1d`
- Path: `configs/base/model/generative_dit1d.yaml`
- Description: Generative PHM DiT1D backbone base
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --override base_configs.model=configs/base/model/generative_dit1d.yaml`
- Common overrides: `model.hidden_dim=32`, `model.patch_size=8`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/model/README.md`, `src/model_factory/generative_model/README.md`
- Status: `/`

#### `base_model_generative_ssm1d`
- Path: `configs/base/model/generative_ssm1d.yaml`
- Description: Generative PHM stateless SSM/Mamba-placeholder backbone base
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --override base_configs.model=configs/base/model/generative_ssm1d.yaml`
- Common overrides: `model.hidden_dim=32`, `model.use_true_mamba=false`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/model/README.md`, `src/model_factory/generative_model/README.md`
- Status: `/`

#### `base_model_generative_unet1d`
- Path: `configs/base/model/generative_unet1d.yaml`
- Description: Generative PHM UNet1D backbone base
- Owner code: `src/model_factory/__init__.py:build_model`
- Keyspace: `model.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --override base_configs.model=configs/base/model/generative_unet1d.yaml`
- Common overrides: `model.hidden_dim=32`, `model.condition_dim=16`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/model/README.md`, `src/model_factory/generative_model/README.md`
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
- Description: 跨系统 few-shot 任务 base（仍用 FS 类型）
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`
- Minimal run: `python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`
- Common overrides: `trainer.num_epochs=1`, `task.target_domain_num=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/FS/README.md`
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

#### `base_task_generative_cfm`
- Path: `configs/base/task/generative_cfm.yaml`
- Description: Generative PHM CFM task base with task.generative.*
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml`
- Common overrides: `task.generative.mode=train`, `task.lr=0.0001`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`
- Status: `/`

#### `base_task_generative_ddpm`
- Path: `configs/base/task/generative_ddpm.yaml`
- Description: Generative PHM DDPM epsilon task base with task.generative.*
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_ddpm.yaml`
- Common overrides: `task.generative.mode=train`, `task.num_train_timesteps=100`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`
- Status: `/`

#### `base_task_generative_drifting_flow`
- Path: `configs/base/task/generative_drifting_flow.yaml`
- Description: Experimental PHM Drifting Flow one-step task base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_drifting_flow.yaml`
- Common overrides: `task.generative.experimental=true`, `task.generative.num_steps=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
- Status: `/`

#### `base_task_generative_meanflow`
- Path: `configs/base/task/generative_meanflow.yaml`
- Description: Experimental PHM MeanFlow/iMF one-step task base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_meanflow.yaml`
- Common overrides: `task.generative.experimental=true`, `task.generative.num_steps=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
- Status: `/`

#### `base_task_generative_ot_nfm`
- Path: `configs/base/task/generative_ot_nfm.yaml`
- Description: Experimental PHM OT-NFM one-step task base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_ot_nfm.yaml`
- Common overrides: `task.generative.experimental=true`, `task.generative.num_steps=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
- Status: `/`

#### `base_task_generative_rectified_flow`
- Path: `configs/base/task/generative_rectified_flow.yaml`
- Description: Generative PHM Rectified Flow task base with task.generative.*
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_rectified_flow.yaml`
- Common overrides: `task.generative.mode=train`, `task.lr=0.0001`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`
- Status: `/`

#### `base_task_generative_score_sde`
- Path: `configs/base/task/generative_score_sde.yaml`
- Description: Generative PHM Score-SDE task base with stochastic task.generative.*
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_score_sde.yaml`
- Common overrides: `task.generative.mode=train`, `task.generative.stochastic_sampler=annealed_langevin`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`
- Status: `/`

#### `base_task_generative_transition_flow_matching`
- Path: `configs/base/task/generative_transition_flow_matching.yaml`
- Description: Experimental PHM Transition Flow Matching one-step task base
- Owner code: `src/task_factory/__init__.py:build_task`
- Keyspace: `task.*`, `task.generative.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml`
- Common overrides: `task.generative.experimental=true`, `task.generative.num_steps=1`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/task/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
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
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`
- Outputs: `{environment.output_dir}/{experiment_name}/iter_{i}/`
- Related docs: `configs/README.md`, `configs/base/trainer/README.md`, `src/trainer_factory/README.md`
- Status: `/`


## Pipeline_01_default

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
- Owner code: `src/Pipeline_01_default.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/00_smoke/dummy_dg.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
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
- Owner code: `src/Pipeline_01_default.py:pipeline`
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
- Owner code: `src/Pipeline_01_default.py:pipeline`
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
- Owner code: `src/Pipeline_01_default.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/03_fewshot/README.md`
- Status: `sanity_ok`

#### `demo_04_cross_system_fewshot`
- Path: `configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`
- Description: Cross-system few-shot demo（跨系统 few-shot 示例）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_system_fewshot.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/cddg_fewshot.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_01_default.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/cross_system_fewshot_tspn/{experiment_name}/iter_{i}/`
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
- Owner code: `src/Pipeline_01_default.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/pretrain_hse_cddg/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/06_pretrain_cddg/README.md`
- Status: `sanity_ok`


## Pipeline_02_pretrain_fewshot

### demo

#### `demo_05_pretrain_fewshot`
- Path: `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- Description: Pretrain + few-shot two-stage demo（当前为单阶段 HSE 对比预训练示例）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_classification.yaml`
  - model: `configs/base/model/backbone_dlinear.yaml`
  - task: `configs/base/task/pretrain.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_02_pretrain_fewshot.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml`
- Common overrides: `trainer.num_epochs=1`, `data.num_workers=0`
- Outputs: `results/demo/pretrain_hse_then_fewshot/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/05_pretrain_fewshot/README.md`
- Status: `sanity_ok`


## Pipeline_06_generative

### demo

#### `demo_10_generative_cfm`
- Path: `configs/demo/10_generative/dummy_generative_cfm.yaml`
- Description: Generative PHM CFM smoke demo（repo 内置 dummy 数据）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_cfm.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_cfm/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`
- Status: `sanity_ok`

#### `demo_10_generative_ddpm`
- Path: `configs/demo/10_generative/dummy_generative_ddpm.yaml`
- Description: Generative PHM DDPM epsilon smoke demo（repo 内置 dummy 数据）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_ddpm.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_ddpm.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_ddpm/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`
- Status: `sanity_ok`

#### `demo_10_generative_drifting_flow`
- Path: `configs/demo/10_generative/dummy_generative_drifting_flow.yaml`
- Description: Experimental PHM Drifting Flow one-step smoke demo
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_drifting_flow.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_drifting_flow.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_drifting_flow/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
- Status: `sanity_ok`

#### `demo_10_generative_meanflow`
- Path: `configs/demo/10_generative/dummy_generative_meanflow.yaml`
- Description: Experimental PHM MeanFlow/iMF one-step smoke demo
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_meanflow.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_meanflow.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_meanflow/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
- Status: `sanity_ok`

#### `demo_10_generative_ot_nfm`
- Path: `configs/demo/10_generative/dummy_generative_ot_nfm.yaml`
- Description: Experimental PHM OT-NFM one-step smoke demo
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_ot_nfm.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_ot_nfm.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_ot_nfm/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
- Status: `sanity_ok`

#### `demo_10_generative_rectified_flow`
- Path: `configs/demo/10_generative/dummy_generative_rectified_flow.yaml`
- Description: Generative PHM Rectified Flow smoke demo（repo 内置 dummy 数据）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_rectified_flow.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_rectified_flow.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_rectified_flow/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`
- Status: `sanity_ok`

#### `demo_10_generative_score_sde`
- Path: `configs/demo/10_generative/dummy_generative_score_sde.yaml`
- Description: Generative PHM Score-SDE smoke demo（repo 内置 dummy 数据）
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_score_sde.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_score_sde.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_score_sde/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`
- Status: `sanity_ok`

#### `demo_10_generative_transition_flow_matching`
- Path: `configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml`
- Description: Experimental PHM Transition Flow Matching one-step smoke demo
- Base configs:
  - environment: `configs/base/environment/base.yaml`
  - data: `configs/base/data/base_cross_domain.yaml`
  - model: `configs/base/model/generative_cfm.yaml`
  - task: `configs/base/task/generative_transition_flow_matching.yaml`
  - trainer: `configs/base/trainer/default_single_gpu.yaml`
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml`
- Common overrides: `trainer.num_epochs=1`, `trainer.device=cpu`, `data.num_workers=0`
- Outputs: `results/demo/dummy_generative_transition_flow_matching/{experiment_name}/iter_{i}/`
- Related docs: `configs/demo/README.md`, `configs/demo/10_generative/README.md`, `src/task_factory/task/generative/README.md`, `src/model_factory/generative_model/README.md`
- Status: `sanity_ok`

### paper

#### `paper_phm_gen_cfm_ablation_first_metadata`
- Path: `configs/paper/phm_generative/cfm_ablation_first_metadata_seed0.yaml`
- Description: Paper PHM generative CFM condition ablation first-metadata seed 0
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/paper/phm_generative/cfm_ablation_first_metadata_seed0.yaml`
- Common overrides: `task.generative.condition_sampling_policy=first_metadata_repeated`
- Outputs: `results/paper/phm_generative/cfm_ablation_first_metadata_seed0/{experiment_name}/iter_{i}/`
- Related docs: `configs/paper/phm_generative/README.md`, `scripts/README.md`, `src/task_factory/Components/generative/metrics/README.md`
- Status: `paper_ready`

#### `paper_phm_gen_cfm_eval_seed0`
- Path: `configs/paper/phm_generative/cfm_eval_train_reference_seed0.yaml`
- Description: Paper PHM generative CFM eval train-reference seed 0
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/paper/phm_generative/cfm_eval_train_reference_seed0.yaml`
- Common overrides: `task.generative.mode=eval`, `task.generative.eval_split=train`
- Outputs: `results/paper/phm_generative/cfm_eval_train_reference_seed0/{experiment_name}/iter_{i}/`
- Related docs: `configs/paper/phm_generative/README.md`, `scripts/README.md`, `src/task_factory/Components/generative/metrics/README.md`
- Status: `paper_ready`

#### `paper_phm_gen_cfm_sample_grid_seed0`
- Path: `configs/paper/phm_generative/cfm_sample_grid_seed0.yaml`
- Description: Paper PHM generative CFM sample grid seed 0
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/paper/phm_generative/cfm_sample_grid_seed0.yaml`
- Common overrides: `task.generative.mode=sample`, `task.generative.num_steps=32`
- Outputs: `results/paper/phm_generative/cfm_sample_grid_seed0/{experiment_name}/iter_{i}/`
- Related docs: `configs/paper/phm_generative/README.md`, `scripts/README.md`, `src/task_factory/Components/generative/metrics/README.md`
- Status: `paper_ready`

#### `paper_phm_gen_cfm_train_grid_seed0`
- Path: `configs/paper/phm_generative/cfm_train_grid_seed0.yaml`
- Description: Paper PHM generative CFM train grid seed 0
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/paper/phm_generative/cfm_train_grid_seed0.yaml`
- Common overrides: `trainer.num_epochs=50`, `task.generative.condition_sampling_policy=grid`
- Outputs: `results/paper/phm_generative/cfm_grid_seed0/{experiment_name}/iter_{i}/`
- Related docs: `configs/paper/phm_generative/README.md`, `scripts/README.md`, `src/task_factory/Components/generative/metrics/README.md`
- Status: `paper_ready`

#### `paper_phm_gen_cfm_train_grid_seed1`
- Path: `configs/paper/phm_generative/cfm_train_grid_seed1.yaml`
- Description: Paper PHM generative CFM train grid seed 1
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/paper/phm_generative/cfm_train_grid_seed1.yaml`
- Common overrides: `trainer.num_epochs=50`, `environment.seed=1`
- Outputs: `results/paper/phm_generative/cfm_grid_seed1/{experiment_name}/iter_{i}/`
- Related docs: `configs/paper/phm_generative/README.md`, `scripts/README.md`, `src/task_factory/Components/generative/metrics/README.md`
- Status: `paper_ready`

#### `paper_phm_gen_ddpm_train_seed0`
- Path: `configs/paper/phm_generative/ddpm_train_distribution_seed0.yaml`
- Description: Paper PHM generative DDPM train distribution seed 0
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/paper/phm_generative/ddpm_train_distribution_seed0.yaml`
- Common overrides: `trainer.num_epochs=50`, `task.generative.condition_sampling_policy=train_distribution`
- Outputs: `results/paper/phm_generative/ddpm_distribution_seed0/{experiment_name}/iter_{i}/`
- Related docs: `configs/paper/phm_generative/README.md`, `scripts/README.md`, `src/task_factory/Components/generative/metrics/README.md`
- Status: `paper_ready`

#### `paper_phm_gen_rectified_flow_train_seed0`
- Path: `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`
- Description: Paper PHM generative Rectified Flow train grid seed 0
- Owner code: `src/Pipeline_06_generative.py:pipeline`
- Keyspace: `environment.*`, `data.*`, `model.*`, `task.generative.*`, `trainer.*`
- Minimal run: `python main.py --config configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`
- Common overrides: `trainer.num_epochs=50`, `model.name=phm_dit1d`
- Outputs: `results/paper/phm_generative/rectified_flow_seed0/{experiment_name}/iter_{i}/`
- Related docs: `configs/paper/phm_generative/README.md`, `scripts/README.md`, `src/task_factory/Components/generative/metrics/README.md`
- Status: `paper_ready`
