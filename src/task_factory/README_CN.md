-----

# Task Factory 模块说明（简体中文）

> 本文件与 `src/task_factory/readme.md` 对应，是 Task Factory 的中文简要版，重点说明：  
> 1）目录结构；2）`task.type` / `task.name` 配置方式；3）常用组合一览（CSV）。

-----

## 📂 目录结构（task_factory）

Task Factory 主要负责把「模型 + 配置」组装成一个 PyTorch Lightning 的 `LightningModule`。  
相关代码集中在：

| 文件 / 目录           | 说明 |
| :-------------------- | :--- |
| `task_factory.py`     | 工厂入口，暴露 `task_factory(...)` / `build_task(...)` 接口，根据 `task.type` 和 `task.name` 动态导入任务模块。 |
| `Default_task.py`     | 默认的单任务 Lightning 封装，实现了标准分类训练逻辑，也是很多自定义任务的基类。 |
| `task/`               | 具体任务实现：`DG/`、`CDDG/`、`pretrain/`、`FS/`、`GFS/`、`ID/`（如 `ID_task`）、`MT/`（多任务 Lightning 模块）。 |
| `Components/`         | 任务通用组件：loss、metrics、正则化、flow 等，可被多个任务复用。 |
| `utils/`              | Task 相关的小工具（例如数据预处理、窗口切分等）。 |

-----

## ⚙️ 配置方式：`task.type` + `task.name`

在 YAML 配置中，Task 工厂只关心两列：

```yaml
task:
  type: "DG"             # 对应 src/task_factory/task/ 下的子目录名
  name: "classification" # 对应该子目录中的 Python 文件名（去掉 .py）
  # 其余字段交给具体任务自己解析
```

- 导入规则（简化版）：
  - 模块路径 = `src.task_factory.task.{task.type}.{task.name}`
  - 例如：`type: "DG"`, `name: "classification"` → `src/task_factory/task/DG/classification.py`
- `Default_task` 和 `ID_task` 走的是同一套工厂体系，只是文件分别在 `Default_task.py` 和 `task/ID/ID_task.py`。

-----

## 🔖 常用 `task.type` / `task.name` 组合（CSV）

下面的 CSV 列出了当前版本中已经实现 / 推荐使用的任务组合。一行对应一种可选任务：

```csv
id,task.type,task.name,module_path,notes
1,Default_task,Default_task,src/task_factory/Default_task.py,"基础单任务 Lightning 封装"
2,Default_task,ID_task,src/task_factory/task/ID/ID_task.py,"基于 ID_dataset 的按需窗口化任务"
3,DG,classification,src/task_factory/task/DG/classification.py,"单数据集领域泛化分类（DG）"
4,CDDG,classification,src/task_factory/task/CDDG/classification.py,"跨数据集领域泛化分类（CDDG）"
5,pretrain,classification,src/task_factory/task/pretrain/classification.py,"监督式分类预训练（通常配合 ID_task / ID_dataset）"
6,pretrain,hse_contrastive,src/task_factory/task/pretrain/hse_contrastive.py,"HSE 提示引导对比预训练"
7,pretrain,masked_reconstruction,src/task_factory/task/pretrain/masked_reconstruction.py,"掩码重建预训练（自监督）"
8,pretrain,prediction,src/task_factory/task/pretrain/prediction.py,"序列预测预训练"
9,pretrain,classification_prediction,src/task_factory/task/pretrain/classification_prediction.py,"分类 + 预测联合预训练"
10,FS,prototypical_network,src/task_factory/task/FS/prototypical_network.py,"Few-shot 原型网络分类"
11,FS,matching_network,src/task_factory/task/FS/matching_network.py,"Few-shot Matching Networks 分类"
12,FS,knn_feature,src/task_factory/task/FS/knn_feature.py,"Few-shot 特征 + kNN 评估"
13,FS,finetuning,src/task_factory/task/FS/finetuning.py,"Few-shot 微调式适配"
14,GFS,classification,src/task_factory/task/GFS/classification.py,"广义 few-shot 分类（base + novel 类）"
15,GFS,matching,src/task_factory/task/GFS/matching.py,"广义 few-shot Matching 风格任务"
```

使用建议：

- 选择任务时先从上表中挑一行，根据需求设置：
  - `task.type` = 对应行的 `task.type`
  - `task.name` = 对应行的 `task.name`
- 任务内部需要的其他字段（如 `loss`, `metrics`, `mask_ratio`, few-shot 的 `num_support` 等）：
  - 请参考各自子目录下的 README，例如：
    - 域泛化：`src/task_factory/task/DG/README.md`
    - 预训练：`src/task_factory/task/pretrain/README.md`
    - Few-shot：`src/task_factory/task/FS/README.md`
    - GFS：`src/task_factory/task/GFS/README.md`

-----

## 🔁 与主 Pipeline 的关系（简要）

- 主入口 `main.py` / 各 Pipeline 会先通过配置系统构造：
  - `args_data`, `args_model`, `args_task`, `args_trainer`, `args_environment`
- 然后调用：

```python
from src.task_factory import build_task

task = build_task(
    args_task=args_task,
    network=model,
    args_data=args_data,
    args_model=args_model,
    args_trainer=args_trainer,
    args_environment=args_environment,
    metadata=data_factory.get_metadata(),
)
```

- `build_task(...)` 内部会根据 `args_task.type` 与 `args_task.name` 使用上面的映射规则去导入并实例化对应任务。

如需查看更底层的实现细节（包括注册装饰器 `@register_task`、多任务 Lightning、ID_task 的特殊逻辑等），请参考：

- `src/task_factory/task_factory.py`
- `src/task_factory/CLAUDE.md`

-----

