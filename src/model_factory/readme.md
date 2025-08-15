# Model Factory 模块

`model_factory` 模块负责管理项目中所有可用的诊断模型。它允许根据配置文件动态地创建和初始化模型实例。

## 核心功能

- **模型注册**: 提供一个机制，让新定义的模型能够被工厂“发现”和注册。
- **模型构建**: 包含一个核心的 `get_model()` 函数，它接收模型名称和相关参数（如输入维度、类别数等），并返回一个初始化的模型对象（通常是 `torch.nn.Module` 的子类）。

## 模块结构

- **`get_model.py`** (建议): 包含核心的 `get_model()` 函数和模型注册表。
- **`backbones/`** (建议): 存放基础的特征提取网络，如 CNN、ResNet、Transformer 等。
- **`heads/`** (建议): 存放分类头或其他任务头，可以与 `backbones` 灵活组合。

## 如何添加新模型

1.  在 `model_factory` 目录下创建一个新的 Python 文件（例如 `my_model.py`）。
2.  在该文件中定义你的模型类（继承自 `torch.nn.Module`）。
3.  使用工厂提供的装饰器或注册函数，将你的新模型注册到一个全局的模型字典中。
4.  现在，你就可以在配置文件中通过指定模型名称来使用它了。

## 使用示例 (in config.yaml)

```yaml
model:
  name: DANN  # 模型名称，工厂会据此查找并构建
  params:
    num_classes: 10
    pretrained: false
    # ... DANN 模型所需的其他参数
```