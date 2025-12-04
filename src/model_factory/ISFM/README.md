# Industrial Signal Foundation Models (ISFM)

ISFM is the foundation model family for industrial signals. ISFM models are composed from three modular parts:

- `embedding` (`E_xx_*`): signal embedding / patching layers
- `backbone` (`B_xx_*`): core sequence model
- `task_head` (`H_xx_*`): task-specific output head

## 1. Config pattern (`model.type = "ISFM"`)

Minimal example:

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"

  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"

  # shared hyperparameters
  d_model: 256
  n_layers: 2
  dropout: 0.1

  # embedding-specific args (for E_01_HSE)
  patch_size_L: 256
  patch_size_C: 1
  num_patches: 128
  output_dim: 1024
```

## 2. Supported ISFM models (`model.name`)

| model.name                 | Description                                  |
|---------------------------|----------------------------------------------|
| `M_01_ISFM`               | Standard ISFM (single-system batch)          |
| `M_02_ISFM`               | Enhanced ISFM with system-aware HSE          |
| `M_02_ISFM_heterogeneous_batch` | ISFM for heterogeneous batches (mixed systems) |
| `M_03_ISFM`               | Lightweight / research-oriented ISFM         |

All of them expect `model.embedding`, `model.backbone`, and `model.task_head` to be set to valid component IDs.

## 3. Embedding components (`model.embedding`)

Current embedding modules (under `ISFM/embedding/`):

| ID             | File                     | Summary                       |
|----------------|--------------------------|-------------------------------|
| `E_01_HSE`     | `E_01_HSE.py`            | Hierarchical Signal Embedding |
| `E_01_HTFE`    | `E_01_HTFE.py`           | Time-frequency style embedding|
| `E_02_HSE_rec` | `E_02_HSE_rec.py`        | HSE variant with reconstruction|
| `E_03_Patch`   | `E_03_Patch.py`          | Basic patch embedding         |
| `E_com_00_PE`  | `E_com_00_PE.py`         | Common positional embedding   |

> 提示：未来如果增加新的 `E_xx_*` 文件，请在此表中补一行，保持 ID 与文件名一致。

### 3.1 `E_01_HSE` 参数表（示例）

`E_01_HSE` 的核心构造参数来自 `args`，主要包括：

| Field          | Type  | Required | Description                                      |
|----------------|-------|----------|--------------------------------------------------|
| `patch_size_L` | int   | Yes      | Patch size along length dimension `L`           |
| `patch_size_C` | int   | Yes      | Patch size along channel dimension `C`          |
| `num_patches`  | int   | Yes      | Number of patches to sample per sample          |
| `output_dim`   | int   | Yes      | Output feature dimension after mixing           |

> 实际 forward 会额外接收 `fs`（采样频率），用于归一化时间轴；该信息通常来自 `metadata` / DataFactory，而不是 `model` 配置中的字段。

其它消融版本（如 `E_01_HSE_abalation`）会引入额外参数（`sampling_mode`、`apply_mixing`、`linear_config` 等），视你是否在实验中暴露为 config 字段，再逐步扩展本 README。

## 4. Backbone components (`model.backbone`)

当前 backbone 模块（在 `ISFM/backbone/`）包括：

| ID                   | File                    | Type          | Notes                     |
|----------------------|-------------------------|---------------|---------------------------|
| `B_01_basic_transformer` | `B_01_basic_transformer.py` | Transformer   | basic transformer encoder |
| `B_02_basic_other`       | `B_02_basic_other.py`       | misc          | generic backbone          |
| `B_03_FITS`              | `B_03_FITS.py`              | forecasting   | FITS-style backbone       |
| `B_04_Dlinear`           | `B_04_Dlinear.py`           | linear / decomp | DLinear backbone        |
| `B_05_Mamba`             | `B_05_Mamba.py`             | sequence model | Mamba-based backbone     |
| `B_06_TimesNet`          | `B_06_TimesNet.py`          | TimesNet      | periodicity-aware         |
| `B_07_TSMixer`           | `B_07_TSMixer.py`           | mixer         | time-series mixer         |
| `B_08_PatchTST`          | `B_08_PatchTST.py`          | transformer   | patch-based transformer   |
| `B_09_FNO`               | `B_09_FNO.py`               | neural operator | FNO backbone           |
| `B_10_VIBT`              | `B_10_VIBT.py`              | transformer   | vibration transformer     |
| `B_11_MomentumEncoder`   | `B_11_MomentumEncoder.py`   | momentum encoder | MoCo-style backbone   |

### 4.1 `B_04_Dlinear` 参数要点

`B_04_Dlinear` 构造函数签名大致为：

```python
class B_04_Dlinear(nn.Module):
    def __init__(self, configs, individual=False):
        ...
        self.patch_size_L = configs.num_patches
        self.channels = configs.output_dim
```

关键配置字段（来自 `args_model` / `configs`）：

| Field         | Type   | Required | Description                                     |
|---------------|--------|----------|-------------------------------------------------|
| `num_patches` | int    | Yes      | used as `patch_size_L` (length of each series) |
| `output_dim`  | int    | Yes      | number of channels `C`                          |
| `individual`  | bool   | Optional | per-channel linear layers (default `False`)    |

其中 `individual` 通常不直接在 YAML 中暴露，为高级使用者保留；一般通过 ISFM 默认设置即可。

## 5. Task heads (`model.task_head`)

当前 task head（在 `ISFM/task_head/`）包括：

| ID                               | File                             | Task type         |
|----------------------------------|----------------------------------|-------------------|
| `H_01_Linear_cla`                | `H_01_Linear_cla.py`            | classification    |
| `H_02_distance_cla`              | `H_02_distance_cla.py`          | distance-based cla|
| `H_02_Linear_cla_heterogeneous_batch` | `H_02_Linear_cla_heterogeneous_batch.py` | classification (heterogeneous batch) |
| `H_03_Linear_pred`               | `H_03_Linear_pred.py`           | prediction / regression |
| `H_04_VIB_pred`                  | `H_04_VIB_pred.py`              | vibration-specific prediction |
| `H_09_multiple_task`             | `H_09_multiple_task.py`         | multi-task        |
| `H_10_ProjectionHead`            | `H_10_ProjectionHead.py`        | projection head (contrastive) |
| `multi_task_head`                | `multi_task_head.py`            | composite heads   |

### 5.1 `H_01_Linear_cla` 参数要点

构造函数依赖：

```python
class H_01_Linear_cla(nn.Module):
    def __init__(self, args):
        num_classes = args.num_classes
        for data_name, n_class in num_classes.items():
            self.mutiple_fc[str(data_name)] = nn.Linear(args.output_dim, n_class)
```

关键配置字段：

| Field          | Type              | Required | Description                                          |
|----------------|-------------------|----------|------------------------------------------------------|
| `num_classes`  | dict[str, int]    | Yes      | map from dataset/system id → number of classes      |
| `output_dim`   | int               | Yes      | feature dimension from backbone/embedding           |

> 实际 forward 中还依赖 `system_id`（通常来自 DataFactory / metadata），以支持多系统分类头。

## 6. 组件 CSV（isfm_components.csv）

For a machine-readable list of all ISFM components, see:

- `src/model_factory/ISFM/isfm_components.csv`

Columns:
- `component_type`: one of `embedding` / `backbone` / `task_head`.
- `component_id`: the ID to use in config (e.g. `E_01_HSE`, `B_04_Dlinear`, `H_01_Linear_cla`).
- `module_path`: Python file path of the implementation.
- `key_args`: a short list of key configuration fields (when documented).
- `args`: example-style argument combinations, useful as a starting point in YAML.
- `notes`: short description of the component.
- `test_status`: testing status marker (e.g. `/` = not recorded, `pass`, `fail`).

This CSV is the canonical registry for ISFM subcomponents and should be updated whenever new embeddings/backbones/heads are added.

## 7. 示例配置组合

### 6.1 标准 CDDG 分类（推荐 demo）

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"

  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"

  # E_01_HSE
  patch_size_L: 256
  patch_size_C: 1
  num_patches: 128
  output_dim: 1024

  # shared / backbone
  d_model: 256
  n_layers: 2
  dropout: 0.1

  # head
  num_classes:
    0: 10   # dataset/system id -> n_class
```

### 6.2 多任务场景（分类 + 预测）

```yaml
model:
  type: "ISFM"
  name: "M_02_ISFM"

  embedding: "E_02_HSE_rec"
  backbone: "B_08_PatchTST"
  task_head: "H_09_multiple_task"

  # embedding-related
  patch_size_L: 256
  patch_size_C: 1
  num_patches: 128
  output_dim: 1024

  # backbone / shared
  d_model: 256
  n_layers: 4
  dropout: 0.1

  # head-related (example)
  num_classes:
    0: 10
  pred_output_dim: 1
```

## 8. Notes

- For ISFM, `model.embedding`, `model.backbone`, and `model.task_head` **must** be valid IDs; the factory will use them to build submodules from `embedding/`, `backbone/`, and `task_head/`.
- In the central model registry (`src/model_factory/model_registry.csv`), ISFM rows record only `model.type`/`model.name`/`module_path`; ISFM component details live in `isfm_components.csv`.
- For non-ISFM model types, component-style fields are not used; their testing status can be tracked via the `test_status` column in `model_registry.csv`.
