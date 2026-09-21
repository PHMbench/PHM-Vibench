# TII：至多五个工业来源，只训练一个共享模型

从 PHMFactory 根目录，在已有 `LQ_signal` 环境运行：

```bash
conda run -n LQ_signal --no-capture-output python -m scripts.tii_one_model
```

命令使用 `configs/experiments/tii/one_model.local.yaml`。该机器本地配置**不能从历史正弦 fixture 直接改名得到**。它尚不存在时，命令只读取已保存资格表中的 CWRU、MFPT、DIRG、PU、HUST24 五个候选，输出具体缺口并以 2 退出；不扫描18份H5、不生成信号、不训练。当前保留的资格表没有合格来源，因此默认命令不能产生真实工业模型。

原生配置已准备好时，一条命令完成检查和训练：

```bash
conda run -n LQ_signal --no-capture-output python -m scripts.tii_one_model \
  --config configs/experiments/tii/one_model.local.yaml \
  --output results/tii_one_model
```

加 `--check-only` 只检查公开配置、来源资格与文件存在，不读取波形或创建模型。

## 单次执行的含义

- 一个 `M_01_ISFM`：`SupportConditionedTokenizer` 的 **support** 组织 + `B_04_Dlinear` + `H_01_Linear_cla`。
- 2–5个经明确资格化的工业来源，共同更新同一 tokenizer/backbone；局部标签 heads 是同一个模型的组成部分。相同H5的多个任务身份不冒充独立来源。
- seed 0，`environment.iterations=1`，一个固定-round epoch，一个GPU。每源每round32窗，数据/优化参数由原生配置声明并继续受原生Task检查。
- 只调用一次 `python -m phmfactory --config ...`。不循环dataset分别建模，不启动U/N/H、LODO、inner encoder、head sweep或多seed。
- 明确请求物理GPU0；已有不同CUDA映射时报错，CUDA不可用由原生训练器失败，不改CPU/DDP。
- 仅source train/validation。没有目标适配/独立query评价，不能估计S-U、pooling interaction或宣称迁移优势。

这是作者缩小资源后的**单模型实施阶段**，不是偷偷降低完整J1/J2资格，也不替代历史冻结的确认性比较。两来源单次训练至多建立该池上的执行证据；三个来源、多个目标及强参照的旧门槛没有因此通过。

## 本地配置准备：一次填实，不猜

先复用已有 `reports/tii_local_j1/` 的检查。新证据确已解决问题后，为固定的2–5个来源写原生runtime YAML，并给出：

1. `data.data_dir` 指向原数据根，`metadata_file`和`record_inventory`严格对应同一固定record集合。字段与检查见现有 `src/data_factory/tii_data.py`；角色仅source_train/source_val，先物理组划分再切窗。
2. `qualification_file`用既有资格CSV字段，保留真实来源与排除原因。所有selected source行必须有证据支持eligible；不要只手改布尔状态。记录级输入仍须通过原生时间、单位、支持与group检查。
3. `evidence_kind: natural_acquisition`，`use_cache: true`，`normalization: source_rms`；有效采样率、通道、单位、物理时长、common/increment频带、grid/K/P/D按原始材料确定。无依据就停止，不由Nyquist替代硬件响应，也不生成假原record/group。
4. `task.source_system_ids`与原生metadata选择字段`target_system_id`为相同有序列表；后者在该Task中不是留出目标。源数上限5，原生Task为`DG.tii_joint`。损失CE、AdamW、零共同/私有惩罚保持不变。
5. 固定整数round budget（至多10000，且在验证边界结束）、num_epochs=1、devices=1、device=cuda、test_after_fit=false、early_stopping=false。源验证决定checkpoint；源validation不是独立test。

参数不全、来源未合格或cache/metadata冲突仍在时，只提交缺口，不运行旧全矩阵。当前 `07_PU_OBSERVATION_PROVENANCE` 的未完成原始依据不能由本启动器补造。不要重新执行已经完成的全库检查。

## 输出与失败

新目录包含 `runtime.yaml`（同一公开配置编译器解析后的配置，仅输出路径显式转入本次目录）、`run.log`、`run.json` 和原生 `native/` 产物。checkpoint使用原生命令返回的路径，不按时间找“最新模型”。不另造指标体系，不在画图时推理。

目录已存在就拒绝，避免重复训练或覆盖失败。运行失败保留退出码/日志，不自动重试、缩减数据、换模型或硬件。检查阶段错误在stderr保留原始诊断；由本地Agent把命令和失败输出同步。成功状态叫 `completed_training_only`，不能当成baseline-valid或J1 Go。

测试（不训练工业模型）：

```bash
python -m unittest discover -s test -p test_tii_one_model.py -v
```

执行代码/正式配置/允许公开的必要原生产物先通过子仓dev同步；论文仓随后更新gitlink与证据指向。原H5/Excel不提交。保持历史No-Go和负结果，下一轮再决定证据是否支持论文主张。
