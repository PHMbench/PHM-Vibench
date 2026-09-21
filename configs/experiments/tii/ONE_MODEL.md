# TII：至多五个工业来源，一个共享模型

正式实现仍是 `scripts/tii_one_model.py`、原生 `DG.tii_joint` 和 `tii_evaluation`。没有第二套 loader、trainer 或同义 runner。

## 当前本地动作：一次20轮验收

在 PHMFactory 根目录，使用已有 `LQ_signal` 环境：

```bash
bash scripts/run_tii_experiments.sh --check-only
bash scripts/run_tii_experiments.sh \
  --config configs/experiments/tii/one_model.local.yaml \
  --output results/tii_acceptance_001
```

相对配置和输出路径均以 PHMFactory 为根。第一条只读取公开配置、资格表和文件路径，不读波形、不创建模型。没有已审定的本地YAML时，读取保留资格报告并以2退出；不扫描18个H5、不生成fixture。当前保留的资格表仍没有合格来源，成功安装不等于可以真实训练。

第二条只执行一个S模型、seed0、物理GPU0、一次20-round fit，每源每round32窗；第10和20轮源验证选checkpoint。不会继续1000轮、启动U/N/H、LODO、HPO或目标head拟合。2–5个真实来源共同更新同一个 `M_01_ISFM`：SupportConditionedTokenizer + B_04_Dlinear + H_01_Linear_cla。局部heads属于同一个模型，相同H5内的多个任务身份不能充当独立来源。

执行链是：

```text
准入检查 → 一次20轮联合训练 → 选定checkpoint
→ 完整source_val预测 → 重载核对 → 冻结增量遮蔽
→ 离线复算 → CSV生成SVG/PDF/PNG
```

## 配置：一次填实，不猜

复用 `reports/tii_local_j1/`，只补最终选定来源缺失的实际依据。不得直接从正弦fixture改名。

- `data_dir`、`metadata_file`、`record_inventory`对应同一批原记录；先物理组划分再切窗，角色仅source_train/source_val。
- `qualification_file`使用现有CSV字段，selected行的eligible必须有新证据，不能手改布尔值绕过No-Go。
- `natural_acquisition`、`use_cache: true`、`normalization: source_rms`；原生/有效采样、通道、单位、时长、共同/增量频带、grid/K/P/D依据原材料确定。Nyquist不是完整硬件响应。
- `task.source_system_ids`与原生metadata选择字段`target_system_id`为相同有序列表；后者在此Task中不代表留出目标。
- CE/AdamW，lr0.001、weight_decay0.0001，两个额外惩罚为0；`data.rounds: 20`、`num_epochs: 1`、`val_check_interval: 10`、`num_sanity_val_steps: 0`、`devices: 1`、`device: cuda`、`test_after_fit: false`、`early_stopping: false`。原生Task继续核对其余冻结条件。
- 验收实际检查Trainer使用32-true精度、梯度累积1；不能通过添加未被消费的YAML字段假装控制了这些量。

原始数据根和历史失败文件只读。参数或依据缺失就停止，不能换CPU、换来源、修标签或自动改频带。

## 产物与解释

新目录保留runtime.yaml、run.log、run.json、原生checkpoint和日志；拟合前固定expected_source_windows.csv及local_class_map.json。

Task写入audit/source_gradients.csv和audit/joint_updates.csv：每源loss对共享参数和自己head的梯度，以及完整joint step的参数变化。额外VJP只观察已有loss图，不额外forward或step，其时间/显存计入成本。**joint delta不能被说成单个source的因果更新**；完整joint step包含全部head，不能要求其他已选head的参数不变。

source_validation_predictions.csv与source_validation_masked_predictions.csv保留完整logits、概率、record/group/channel/window、checkpoint。原生monitor与离线NLL复算相符后才置为`completed_source_acceptance_not_transfer`。恢复核对比较同一选定状态重载前后的logits和RMS，不验证恢复后新增训练步。

源验证参与checkpoint选择，不是独立test。冻结遮蔽只测模型依赖，不替代common-only重新训练，不产生S-U、迁移或pooling结论。

## 恢复而不重训

```bash
bash scripts/run_tii_experiments.sh --reuse --output results/tii_acceptance_001
```

仅当run.json记录一次fit已完成才恢复。预测已导出时只重做离线检查，不打开H5、checkpoint或GPU；预测尚未完整导出时，从固定checkpoint补后处理，不调用fit。现有成功状态词不能代替完整CSV检查；缺窗口、原生score不符或恢复未通过都停止。图文件不完整时从CSV再生成，不调用模型。

失败日志保留，输出目录不能被另一次训练覆盖。不自动重试失败训练或增加预算。训练-only历史命令`python -m scripts.tii_one_model --config ...`仍保留，但本轮入口只授权20轮验收。

## 验证和同步

```bash
python -m pytest test/test_tii_one_model.py test/test_tii_one_model_acceptance.py -q
```

第一组保护原有launcher；第二组检查新增源预测/恢复，并在临时明确正弦数据上执行真实原生CPU Model/Data/Task/Trainer。该回归没有工业数据资格含义。CI不访问用户H5、不产生工业论文结果。

实际代码/配置/许可允许的产物先经正常PR合入子仓dev；论文仓随后更新准确gitlink和Goal。保留旧No-Go、不修改主比较或Results；完成本地验收后等待下一轮证据解释。
