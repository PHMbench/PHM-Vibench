# Vbench 数据集贡献指南

本指南旨在帮助贡献者向 Vbench 项目添加新的数据集和数据处理方法。通过遵循这些步骤，您可以确保您的贡献能够顺利集成到项目中。

## Git 协作流程（可以Vscode图形化处理）

1. **Fork 项目仓库**：在 GitHub 上 fork 本项目到您自己的账户
2. **克隆您的 fork**：
   ```bash
   git clone https://github.com/YOUR-USERNAME/Vbench.git
   cd Vbench
   ```
3. **创建新分支**：
   ```bash
   git checkout -b add-dataset-NAME
   ```
   将 NAME 替换为您要添加的数据集名称

4. **提交更改**：
   ```bash
   git add .
   git commit -m "Add dataset: NAME"
   git push origin add-dataset-NAME
   ```

5. **创建 Pull Request**：在 GitHub 上创建 PR，详细描述您添加的数据集

## 添加新数据集的步骤

### 1. 数据集结构与存放

数据集应按以下结构存放：

```
Your Data Path/
├── raw
│   ├── YOUR_DATASET_NAME # Example: CWRU
│   │   ├── file1.csv # 原始数据文件，原始形式,可以嵌套目录，保证和元文件一致
│   │   ├── file2.csv
│   │   └── ...


### 2. 元数据准备

1. **在飞书中准备元数据**：首先在飞书中准备您的数据集元数据，详见飞书


2. **根据onedrive(网盘的目录形式)**

### 3. 实现数据读取功能

在 `data_factory/reader` 目录下创建或修改相关文件以支持您的数据集：

1. 如果是全新类型的数据集，创建新的读取器文件
2. 如果是现有类型的数据集，可能只需修改现有读取器


## 测试您的贡献

添加新数据集后，请进行以下测试：

.Vbench/test/test_data_factory.ipynb

## 提交检查清单

提交 PR 前，请确保：

- [ ] 数据集结构符合要求
- [ ] 元数据格式正确且已放入 `meta_data` 目录
- [ ] 实现了必要的数据读取功能
- [ ] 所有测试通过

## 问题与帮助

如有任何问题或需要帮助，请在 GitHub 上创建 issue 或联系项目维护者。

感谢您对 Vbench 项目的贡献！
