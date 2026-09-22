# 一个共享模型的逐源95%拟合诊断

此模式扩展 `scripts.tii_one_model`，不创建第二个trainer。一个S模型、2–5个合格工业来源、seed0、物理GPU0；一次fit完成固定10,000次联合更新。原来的 `--acceptance` 20轮模式保持不变，但本轮不先跑20轮再重新初始化。

## 新的观测，不是强制正结果

同一次训练在0、20、100、500、1000、2500、5000、10000次更新后，读取**完整且固定**的 source_train/source_val 窗口，各自计算逐源NLL、组等权准确率、窗口准确率、macro-F1、balanced accuracy及多数类参考。不是随机抽样batch准确率。诊断前后恢复训练模式和PyTorch RNG；没有额外优化步。

验证仍每100步计算一次，唯一checkpoint选择依据是源验证组等权NLL，平局沿用原规则。源训练集的95%要求不替换选模指标。最终在**同一选定checkpoint**上要求所有源的窗口准确率和组等权准确率均≥0.95；不取各源不同的最好checkpoint，也不拿全源均值替代最差源。终态是否达到、首次观测到所有源同时达到的步数分别报告。

多数类模型也可能有高Accuracy，因此macro-F1、balanced accuracy和多数类参考不能省略。达到95%只表示预定训练拟合目标达到，不证明少数类充分学习、独立泛化、迁移或Foundation Model能力。

## 真实配置

使用已有公开分析入口接受的 `configs/experiments/tii/one_model.local.yaml`。以下只是需要固定的字段，**不是完整可训练YAML**：

```yaml
data:
  rounds: 10000
  source_batch_size: 32
model:
  backbone: B_04_Dlinear
  token_organization: support
trainer:
  num_epochs: 1
  val_check_interval: 100
  num_sanity_val_steps: 0
  early_stopping: false
  monitor: val_group_nll
  monitor_mode: min
  checkpoint_min_delta: 1.0e-8
```

其他已有约束不变，包括source RMS、seed0、CE/AdamW(lr=.001, weight_decay=.0001)、1个设备、自然采集资格。K/P/D、实际频带、时间网格、记录角色/物理组和单位必须来自已审定配置；不能套用K4/D8构造夹具，也不在脚本里猜数值或调整来源以达到95%。

## 命令（PHMFactory根目录）

```bash
bash scripts/run_tii_fit95.sh --check-only configs/experiments/tii/one_model.local.yaml
bash scripts/run_tii_fit95.sh configs/experiments/tii/one_model.local.yaml results/tii_fit95_001
# 一次fit已完成后，恢复导出/分析，不再训练
bash scripts/run_tii_fit95.sh --reuse results/tii_fit95_001
```

环境沿用LQ_signal。首轮资格检查不读全库波形；无合格输入就停止。脚本最终退出码0=执行与95%目标均完成，3=已完成的拟合结果未达目标，其他错误保留原失败与日志。`python -m scripts.tii_one_model --fit95` 本身只负责训练与导出；质量裁决由shell最后调用的 `--analyze-only` 给出。

`--reuse`不重新fit；完整导出存在时只验证原始CSV和统计。缺失训练轨迹或中断fit不能靠重复训练补造。没有“训练直到95%”的无界循环，10,000步不够时保留曲线与未达标结论，等待下一轮只改变一个因素的诊断。

## 产物

- `runtime.yaml`、`run.json`、原生日志、validation-selected checkpoint。
- `expected_source_windows.csv`：训练前保存的train/val完整窗口定义。
- `fit95/step_*.ckpt` 及对应的完整train/val预测：同一次优化轨迹的观测点，不是额外fit。
- `source_fit_curve.csv`：每个观测点、每源、每role一行。
- `source_training_predictions.csv` / `source_validation_predictions.csv`：同一验证选定checkpoint。
- `source_validation_masked_predictions.csv`：冻结增量遮蔽，非重训消融。
- `fit95_summary.json`、`analysis.json`、各源/组统计与前20次真实梯度和joint update记录。
- `fit_figures/`：从CSV绘制的逐源拟合曲线，SVG/PDF/PNG；0.95虚线明确是目标线。

训练时间包括诊断推理和快照开销，不能当成无诊断的算法吞吐量。10,000更新×32窗口=每源320,000次采样曝光，不是320,000个独立记录。
