# AGENTS_CN.md（运行手册 + double-check）

本文件用于“快速执行与复核”。修改策略/边界/顺序请看 `CLAUDE.md`；项目概览与上手路径请优先看
`README_CN.md`（配置体系细节见 `configs/README.md`）。

## 1 分钟理解仓库
- PHM-Vibench 是配置优先（config-first）的工业振动信号基准：实验由 YAML 配置定义（environment/data/model/task/trainer）。
- 通过 `src/*_factory/` 工厂与注册表实现可插拔扩展（dataset/model/task/trainer）。
- 维护入口：`python main.py --config <yaml> [--override key=value ...]`（pipeline 由 YAML 顶层 `pipeline:` 选择）。

## 快速命令（可复制）
```bash
# 离线冒烟（仓库内置 Dummy_Data；无需下载数据）
python main.py --config configs/demo/00_smoke/dummy_dg.yaml

# 校验 demo 配置（loader 驱动 + pydantic schema）
python -m scripts.validate_configs

# Inspect：最终配置 + 字段来源 + 实例化落点 + sanity
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1

# Registry → Atlas（docs/CONFIG_ATLAS.md 必须保持同步）
python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md

# 校验文档链接 / @README 约定
python -m scripts.validate_docs

# 维护中的测试
python -m pytest test/
```

## 目录导航（改哪里）
- `configs/demo/`：维护中的模板（优先从这里复制）。
- `configs/experiments/<task_dataset_variant>/`：本地研究配置（不要污染 demo）。
- `configs/reference/`：历史遗留（计划迁移/删除；不要当模板）。
- `src/data_factory/` / `src/model_factory/` / `src/task_factory/` / `src/trainer_factory/`：核心扩展点（按工厂注册）。
- `docs/`：维护文档；`docs/LQ_fix/`：内部计划与记录。

## 扩展流程（最常见）
- 新实验配置：
  1) 从 `configs/demo/*` 复制到 `configs/experiments/<name>/exp.yaml`
  2) 先跑 `python -m scripts.config_inspect --config <exp.yaml>` 确认 sources/targets
  3) 再训练（建议先 `--override trainer.num_epochs=1` 做冒烟）
- 新增维护 demo：
  1) 放入 `configs/demo/...`
  2) 在 `configs/config_registry.csv` 增加一行
  3) `python -m scripts.gen_config_atlas` 生成/更新 `docs/CONFIG_ATLAS.md`
  4) `python -m scripts.validate_configs` 必须通过

## 备注
- `streamlit_app.py` 为实验性功能（不作为验证门禁）。
- `dev/test_history/` 为历史 runner（可选，可能需要额外依赖）。

## 提交/PR 建议（便于 review）
- 尽量保持改动聚焦（配置/代码/文档可拆分则拆分）。
- vibecoding（AI 辅助编码）更新遵循 KISS：避免过度工程化、过早抽象与不必要的防御性设计；遵循奥卡姆剃刀原则，
  立足第一性原理，渐进式开发。
- 每次变更附带：
  - 变更清单 + 动机
  - 如何验证（上面的命令）
  - 预期产物（如 `docs/CONFIG_ATLAS.md` 更新、输出目录模式）
