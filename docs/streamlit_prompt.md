# Streamlit App Prompt

The following instructions provide a structured prompt for an AI to design a user-friendly Streamlit application for PHMbench/vibench. The application should allow users to configure experiments via a graphical interface, choose data loading methods, adjust parameters, run the pipeline and visualize results.

```
### **为 PHMbench/vibench 设计 Streamlit 应用的AI提示词**

#### **# 角色与目标 (Role and Goal)**

你是一位资深的AI工程师，精通 `Streamlit`、`PyTorch` 和 `MLOps`。你的任务是为现有的 `PHMbench`（或称 `vibench`）框架设计并构建一个交互式的 Streamlit Web 应用。

这个应用的核心目标是：**将基于命令行和 `.yaml` 文件的实验流程，转化为一个直观、可交互的图形化界面**，从而极大地提升研究和调试效率。

#### **# 核心用户体验 (Core User Experience)**

最终的应用应该遵循以下用户流程：

1. **选择数据模式**: 用户在两种数据加载模式中选择一种。
2. **加载数据**:
   * 模式一：用户直接上传训练/验证/测试的信号文件（如 `.npy`, `.csv`）。
   * 模式二：用户上传一个元数据文件 (`metadata.csv/xlsx`)，并指定 HDF5 数据目录，然后通过ID勾选来定义训练/验证/测试集。
3. **加载基础配置**: 用户从现有的 `configs/` 目录中选择一个基础的 `.yaml` 配置文件。
4. **交互式调参**: 应用界面上会展示出基础配置中的关键参数（如学习率、批大小、模型类型等），用户可以通过滑块、下拉菜单、输入框等UI组件进行修改。
5. **执行与监控**: 用户点击“开始实验”按钮后，应用会在后端执行 `vibench` 的训练和测试流程，并实时展示进度。
6. **结果可视化**: 实验结束后，应用会清晰地展示关键的性能指标、损失曲线、预测对比图等结果。

#### **# 关键组件与实现细节 (Key Components & Implementation Details)**

##### **## 1. 配置侧边栏 (`st.sidebar`)**

侧边栏将作为主要的配置区域。

* **数据加载模块 (Data Loading Module)**:
    * 使用 `st.radio` 提供两种数据加载选项：**"传统文件上传模式 (Traditional File Upload)"** 和 **"PHMbench元数据模式 (PHMbench Metadata Mode)"**。
    * **根据选项动态显示UI**:
        * 如果选择“传统模式”，则显示三个 `st.file_uploader` 组件，分别用于上传 `train`, `val`, `test` 文件（需要支持多文件上传 `accept_multiple_files=True`）。后端逻辑应调用您提供的 `build_env_traditional` 函数。
        * 如果选择“元数据模式”，则显示：
            * 一个 `st.file_uploader` 用于上传 `metadata.csv` 或 `.xlsx` 文件。
            * 一个 `st.text_input` 用于输入存放 `.h5` 文件的数据目录路径。
            * 在成功加载元数据后，使用 `st.multiselect` 或带复选框的 `st.dataframe` 让用户从ID列表中选择 `train_ids`, `val_ids`, 和 `test_id`。后端逻辑应调用您提供的 `build_env_phmbench` 函数。

* **实验配置模块 (Experiment Configuration Module)**:
    * 使用 `st.selectbox` 列出 `configs/` 目录下的所有 `.yaml` 文件，供用户选择一个作为**基础模板**。
    * 在加载模板后，使用 `st.expander` 将可调参数进行分组显示，例如：
        * **`with st.expander("模型参数 | Model Parameters"):`**:
            * `st.selectbox("模型名称 (model.name)", ...)`
            * `st.number_input("学习率 (model.lr)", ...)`
        * **`with st.expander("训练器参数 | Trainer Parameters"):`**:
            * `st.number_input("训练轮数 (trainer.max_epochs)", ..., step=1)`
            * `st.number_input("批大小 (data.batch_size)", ..., step=1)`
        * **`with st.expander("任务参数 | Task Parameters"):`**:
            * 根据任务类型（如 `FewShot`）显示特定参数，例如 `st.slider("N-way", ...)`。

##### **## 2. 主页面 (Main Page)**

主页面用于触发实验和展示结果。

* **执行控制**:
    * 一个醒目的 `st.button("🚀 开始实验 | Run Experiment")`。
    * 在按钮下方，使用 `st.spinner` 或 `st.progress` 来显示实验进行中的状态。

* **结果展示区**:
    * 使用 `st.tabs` 将不同结果分开展示，例如 "指标总结", "训练曲线", "预测详情", "原始日志"。
    * **"指标总结"**: 使用 `st.metric` 展示最终的测试准确率、F1分数、RMSE等关键指标。使用 `st.table` 或 `st.dataframe` 展示详细的指标表格。
    * **"训练曲线"**: 使用 `st.pyplot` 或 `st.plotly_chart` 绘制训练和验证的损失/准确率曲线。
    * **"预测详情"**: 对于回归任务，绘制预测值与真实值的对比图（就像您之前遇到的问题图一样）。
    * **"原始日志"**: 在一个 `st.expander` 中，使用 `st.code` 来显示完整的训练过程日志。

#### **# 后端集成逻辑 (Backend Integration)**

* Streamlit 应用的核心是收集用户在UI上配置的所有参数，并将它们整合成一个与 `Hydra` 兼容的配置对象（例如一个字典或 OmegaConf 对象）。
* 当用户点击“开始实验”时，后端调用 `vibench` 的主 `Pipeline` 函数，并将这个动态生成的配置对象作为输入。
* `Pipeline` 的执行结果（如最终指标、保存的图像路径、日志等）需要被捕获并返回给前端，以便在UI上进行展示。

#### **# 代码参考 (Code Context)**

请将用户提供的 `build_env_traditional` 和 `build_env_phmbench` 函数作为数据加载模块的实现基础。这些函数定义了将UI输入转化为 `vibench` 管道所需环境的核心逻辑。
```
