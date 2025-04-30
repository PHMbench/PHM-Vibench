# Vbench 模型贡献指南

本指南旨在帮助贡献者向 Vbench 项目添加新的模型和算法实现。通过遵循这些步骤，您可以确保您的贡献能够顺利集成到项目中。

## Git 协作流程（可以Vscode图形化处理）

1. **Fork 项目仓库**：在 GitHub 上 fork 本项目到您自己的账户
2. **克隆您的 fork**：
   ```bash
   git clone https://github.com/YOUR-USERNAME/Vbench.git
   cd Vbench
   ```
3. **创建新分支**：
   ```bash
   git checkout -b add-model-NAME
   ```
   将 NAME 替换为您要添加的模型名称

4. **提交更改**：
   ```bash
   git add .
   git commit -m "Add model: NAME"
   git push origin add-model-NAME
   ```

5. **创建 Pull Request**：在 GitHub 上创建 PR，详细描述您添加的模型

## 添加新模型的步骤

### 1. 模型结构与存放

模型应按以下结构存放：

```
model_factory/
├── Model_type/ # 例如: CNN, LSTM
│   ├── YOUR_MODEL_NAME.py # Dlinear.py
│   │   └── ...
```

### 2. 模型元数据准备

1. **在飞书中准备元数据**：首先在飞书中准备您的模型元数据，详见飞书

1. **包含必要的模型信息**：
   - 模型名称与版本
   - 适用任务类型
   - 输入输出规格
   - 模型参数说明
   - 训练与推理要求

### 3. 实现模型接口

在 `model_factory/Model_type/` 目录下创建或修改相关文件以支持您的模型：

1. 如果是全新类型的模型，创建新的模型文件


## 提交检查清单

提交 PR 前，请确保：

- [ ] 模型结构符合要求
- [ ] 实现了必要的模型接口
- [ ] 所有测试通过
- [ ] 代码质量符合项目标准
- [ ] 添加了必要的文档和注释

## 问题与帮助

如有任何问题或需要帮助，请在 GitHub 上创建 issue 或联系项目维护者。

感谢您对 Vbench 项目的贡献！
