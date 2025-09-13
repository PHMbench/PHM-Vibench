# Data Directory Override 功能使用说明

## 概述

现在可以通过 `--data_dir` 命令行参数来覆盖配置文件中的 `data_dir` 设置，而无需修改配置文件。

## 使用方法

### 基本用法

```bash
# 使用配置文件中的默认data_dir
python main.py --config_path configs/demo/Single_DG/CWRU.yaml

# 通过命令行参数覆盖data_dir
python main.py --config_path configs/demo/Single_DG/CWRU.yaml --data_dir /your/custom/data/path
```

### 多种Pipeline支持

所有Pipeline都支持data_dir覆盖：

```bash
# 默认Pipeline
python main.py --pipeline Pipeline_01_default --config_path configs/demo/Single_DG/CWRU.yaml --data_dir /new/path

# 预训练+少样本Pipeline
python main.py --pipeline Pipeline_02_pretrain_fewshot --config_path configs/demo/Pretraining/Pretraining_demo.yaml --fs_config_path configs/demo/GFS/GFS_demo.yaml --data_dir /new/path

# 多任务Pipeline
python main.py --pipeline Pipeline_03_multitask_pretrain_finetune --config_path configs/multitask_pretrain_finetune_config.yaml --data_dir /new/path

# ID Pipeline
python main.py --pipeline Pipeline_ID --config_path configs/demo/ID/id_demo.yaml --data_dir /new/path
```

## 技术实现

### 配置系统集成

- 使用 `load_config()` 函数的覆盖机制
- 支持点符号语法：`'data.data_dir': '/new/path'`
- 不影响配置文件的其他设置

### Pipeline适配

- **Pipeline_01_default**: 直接在配置加载时应用覆盖
- **Pipeline_02_pretrain_fewshot**: 修改 `run_stage()` 函数支持args参数传递
- **Pipeline_03_multitask_pretrain_finetune**: 通过全局变量和构造函数覆盖
- **Pipeline_ID**: 继承默认Pipeline的覆盖功能

## 验证方法

可以在Pipeline输出中看到覆盖确认信息：

```
[INFO] 通过命令行参数覆盖data_dir: /your/custom/data/path
[INFO] 加载配置文件: configs/demo/Single_DG/CWRU.yaml
```

## 优势

1. **灵活性**: 无需修改配置文件即可更改数据路径
2. **便于部署**: 同一配置文件可在不同环境使用不同数据路径
3. **向后兼容**: 不指定`--data_dir`时使用配置文件中的默认值
4. **统一接口**: 所有Pipeline使用相同的参数格式