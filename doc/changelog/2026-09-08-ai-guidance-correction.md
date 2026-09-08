# 2026-09-08：纠正 AI 指导中的旧接口

本次先处理配置、Trainer、Utils 和 Task Components 四个高风险指导入口。

- 删除上述目录的旧 `CLAUDE.md`，不再把旧 ConfigWrapper 主路线、旧 Trainer 配方和自动修复工具示例作为 AI 操作指令。
- 配置兼容边界和 Trainer 接口继续使用已有同级 README；不重复搬运已经覆盖的内容。
- 缩短 Utils README，改用真实的公共入口与当前 Registry 接口；旧 API_REFERENCE 只保留到 README 的导航。
- 修正 Components README：`get_metrics` 必须接收 metadata，维护 Task 显式传入 loss_name；删除 TypeError 后更换 loss 调用的示例。
- 保留实际组件入口与研究适用范围，不用源码存在证明 benchmark 支持。历史审计仍作为历史记录保留。

本次不修改运行源码、配置、loss、metric、split、checkpoint、依赖或发布状态。
不下载或重跑 THU。后续统一根级 AI 入口时须同步现有边界检查，不能直接绕过检查。
