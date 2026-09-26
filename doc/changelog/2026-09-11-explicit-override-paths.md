# 2026-09-11：错误 override 不再被当作有效设置

公共配置解析器拒绝空路径段和段首尾空格，例如 `trainer..num_epochs=2`、`.trainer.num_epochs=2`。错误保留原路径，不自动折叠点号或修改字段名。

现有 Trainer Schema 明确列出 Default_trainer 实际消费的可选参数。选择 Default_trainer 时，未知字段（如 `num_epoch`）在公共配置分析阶段失败，并给出支持字段；不再添加一个被训练器忽略的新键后继续运行。正确字段仍是 `trainer.num_epochs`。

合法可选字段无需预先出现在 base YAML 中。Schema 仍只做验证，不把其默认值写入运行配置；自定义 Trainer 和现有 Model/Task 研究扩展字段保持原来的显式配置能力。本次不采用“只能覆盖已存在叶子”的全局禁令。

回归测试进入已有 `test/test_trainer_lifecycle_schema.py`，覆盖两个本地复现输入、路径边界、公开 analyze/inspect/preflight/run 一致拒绝、错误 YAML、合法可选字段、研究扩展、最后覆盖优先及严格类型。被拒绝请求不会进入 Pipeline import 或创建最终结果目录。

本次不修改运行期默认值、训练器、checkpoint/early-stopping 数学行为、数据或实验配置，不处理 L03 成功状态合同，也不下载或训练 THU。验证结果以本次 PR 的最终检查为准。
