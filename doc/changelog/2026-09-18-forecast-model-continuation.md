# 2026-09-18：继续原生预测模型集成

在现有模型集成分支增加 SOFTS、FreTS 和 Time-Series-Library 的 TSMixer 实现，复用 `DG.point_forecasting`，不新增 Pipeline、Trainer 或第三方训练框架。三项均附完整 Dummy 配置，公共验证要求真实 fit、selected checkpoint 恢复和 MSE/MAE。

- SOFTS 保留训练随机池化、评估加权池化及历史前缀归一化；没有将评估改为随机抽样。
- FreTS 保留官方公开代码的对角频域运算和 softshrink，不把 `einsum('bijd,dd->bijd')` 改成稠密矩阵。`channel_mixing=true` 明确对应上游字符串开关为 `'1'` 的分支。配置中的较小 embedding/hidden 宽度属于 smoke 设置，不是论文结果。
- TSMixer 标明 TSL 变体，不宣称复现 Google 原始训练体系。

本轮先在 Python 3.13 的源码切片中运行 51 项真实组件测试，全部通过；不把该局部测试等同于完整仓库安装验收。公共 CLI 和现有回归由该提交的 GitHub Actions 在 Python 3.10 上执行，结果以实际日志为准。

来源：

- [SOFTS](https://github.com/Secilia-Cxy/SOFTS/blob/main/models/SOFTS.py)，MIT，保留完整许可。
- [FreTS](https://github.com/aikunyi/FreTS/blob/main/models/FreTS.py)，Apache-2.0。
- [TSMixer TSL](https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/models/TSMixer.py)，MIT。

187 项总表仍需区分研究索引、既有模型和真正完成的执行路径。未取得权重、许可证或实际结果的条目不记通过；不以本批冒充全目录完成。最终集成 PR 继续保留 Draft，尚未写入 dev。
