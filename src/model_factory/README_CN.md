# PHM-Vibench 模型工厂（Model Factory）

PHM-Vibench 的模型工厂提供了一套统一的接口，用于构建 PHM 场景下的各种深度学习模型（ISFM、Transformer、CNN、RNN、MLP、Neural Operator 等）。  

本文件重点说明：
- 如何通过配置文件的 `model` 段选择模型；
- `model.type` / `embedding` / `backbone` / `task_head` 的含义和关系；
- 去哪里查完整的可选项列表。

> 英文版本见 `src/model_factory/README.md`。  
> 对 Agent/维护者的内部说明见 `src/model_factory/CLAUDE.md`。

## 1. 目录结构概览

主要文件与子目录：

| 路径                   | 说明                                                                 |
| :--------------------- | :------------------------------------------------------------------- |
| `model_factory.py`     | 主入口函数 `model_factory(args_model, metadata)`，返回 `nn.Module`。 |
| `MLP/`                 | 各类 MLP 模型。                                                      |
| `CNN/`                 | 一维卷积模型（如 `ResNet1D`）。                                      |
| `RNN/`                 | 循环网络模型。                                                      |
| `NO/`                  | Neural Operator 模型（如 `FNO`）。                                   |
| `Transformer/`         | Transformer 系列模型（如 `PatchTST`）。                              |
| `ISFM/`                | 工业信号基础模型（含 `embedding` / `backbone` / `task_head` 子模块）。 |
| `ISFM_Prompt/`         | Prompt 风格的 ISFM 变体。                                            |
| `X_model/`             | XAI / 辅助模型。                                                     |

一般每个模型文件会暴露一个 `Model` 类，由工厂动态导入并实例化。

## 2. 配置接口（YAML）

所有模型的选择都通过配置文件中的 `model` 段完成，例如：

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"

  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"

  d_model: 256
  n_layers: 2
  dropout: 0.1
```

关键字段说明：

- `model.type`：模型类型 / 目录名，例如 `ISFM`、`Transformer`、`CNN` 等；
- `model.name`：该类型下具体模型文件/类名，例如 `M_01_ISFM`、`PatchTST` 等；
- `model.embedding`：嵌入模块 ID，主要用于 ISFM（如 `E_01_HSE`）；
- `model.backbone`：骨干网络 ID（如 `B_04_Dlinear`、`B_08_PatchTST`）；
- `model.task_head`：任务头 ID（如 `H_01_Linear_cla`、`H_03_Linear_pred`、`H_09_multiple_task`）；
- 其余字段：作为超参数透传给模型构造函数（例如 `patch_size_L`、`num_layers`、`dropout` 等）。

### 2.1 工厂内部解析流程

模型工厂的主要步骤如下：

1. 从 YAML 读取 `config.model`，构造 `args_model`。
2. 使用 `model.type` 和 `model.name` 构造导入路径：
   - `src.model_factory.{type}.{name}`，例如 `src.model_factory.ISFM.M_01_ISFM`。
3. 导入模块并实例化：

   ```python
   model_module = importlib.import_module(
       f".{args_model.type}.{args_model.name}", package="src.model_factory"
   )
   model = model_module.Model(args_model, metadata)
   ```

4. 若配置了 `weights_path`，则加载对应 checkpoint 权重。
5. 返回初始化好的 `torch.nn.Module` 实例，供任务/训练器使用。

## 3. v0.1.0 推荐 demo 配置（ISFM）

在 v0.1.0 中，我们推荐所有 demo 统一使用以下 ISFM 组合（对应 `configs/base/model/backbone_dlinear.yaml`）：

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"
  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"
```

该组合与 `configs/reference/experiment_1_cddg_hse.yaml` 主干一致，可以在以下场景中复用：

- 单数据集 / 多数据集 CDDG 故障诊断；
- DG / FS / 预训练 等场景；

具体任务行为由 `task.type` 和 `task.name` 决定（例如分类 vs 对比学习），模型本身配置可以保持不变。

## 4. 模型注册表（CSV）

为了管理不同模型及其组件的对应关系，我们维护了一个 CSV 文件：

- `src/model_factory/model_registry.csv`

该文件每一行代表一个“可用组合”，包含列：

- `model.type`：模型类型（例如 `ISFM`、`Transformer`、`CNN`）；
- `model.name`：模型文件/类名（例如 `M_01_ISFM`、`PatchTST`）；
- `module_path`：Python 源码路径（例如 `src/model_factory/ISFM/M_01_ISFM.py`）；
- `notes`：该组合的推荐用途或备注；
- `test_status`：测试状态标记（例如 `/` 表示暂未记录，可在跑完单测/集成测试后更新为 `pass` 或 `fail` 等）。

**使用建议：**

- 想要使用某个模型，先在 CSV 中找到对应行（确认 `model.type` 和 `model.name` 以及源码路径）；
- 然后根据对应 `model.type` 目录下的 README，补充该类型需要的字段（如 ISFM 的 `embedding/backbone/task_head`，或 Transformer/CNN 的模型超参数）；
- 最终在 YAML 的 `model` 段中写全这些字段。

## 5. 各 `model.type` 的详细参数说明

不同类型有各自的专有配置参数，建议在对应目录添加/查看 README：

- `src/model_factory/ISFM/README.md`：
  - 解释 ISFM 的整体结构（`embedding` / `backbone` / `task_head`）；
  - 列出所有可选的 `embedding`（如 `E_01_HSE`、`E_02_HSE_v2`、`E_03_Patch` 等），并**说明每个 embedding 需要的专有参数**，例如：
    - `E_01_HSE` 需要 `patch_size_L`、`patch_size_C`、`num_patches`、`output_dim` 等；
  - 列出所有可选 `backbone`（如 `B_04_Dlinear`、`B_08_PatchTST` 等）及关键超参；
  - 列出所有可选 `task_head`（如 `H_01_Linear_cla`、`H_03_Linear_pred`、`H_09_multiple_task` 等）及其配置。

- `src/model_factory/Transformer/README.md`（如存在）：
  - 介绍 `PatchTST`、`Autoformer`、`Informer` 等模型；
  - 说明它们在 `model` 段中需要哪些字段（如 `d_model`、`n_heads`、`e_layers` 等）。

- 类似地，`CNN/`、`RNN/`、`MLP/`、`NO/` 等目录也可以逐步补充对应 README，列出：
  - 可选模型名（`model.name`）；
  - 关键超参数及默认值；
  - 推荐的使用场景。

如果某个目录暂时没有 README，可以直接阅读代码并在修改时顺便补充文档。

## 6. 工厂调用流程小结

整体流程可以概括为：

1. 配置文件中定义好 `model` 段（type/name/embedding/backbone/task_head 等）；
2. 训练入口通过 `load_config` 等方式读取 YAML，构造 `args_model`；
3. 调用 `model_factory(args_model, metadata)`，根据 `type` 和 `name` 动态导入并实例化模型；
4. （可选）从 `weights_path` 加载预训练权重；
5. 返回 `nn.Module`，供 task/trainer 使用。

## 7. 对贡献者的建议

如果你要新增或修改模型，请尽量遵循：

- 把新模型放在正确的 `model.type` 目录下，并提供 `Model(args_model, metadata)` 构造函数；
- 在 `docs/v0.1.0/codex/model_registry.csv` 中增加一行，记录新的组合（type/name/embedding/backbone/task_head 等）；
- 在对应类型目录的 README 中：
  - 添加最小可运行的 YAML 片段；
  - 把新增的 embedding/backbone/task_head 及其专有参数完整列出；
- 保持 YAML 中 key 全部小写、使用下划线，和仓库现有风格一致。

这样可以保证主配置入口（`model` 段）清晰易懂，同时让新模型/新组件能够被 pipeline 正确发现和复用。 
