# plot/ → src/plot_factory 迁移计划（KISS / 渐进式 / 高解耦）

## 执行摘要

目标：把 `plot/` 的“脚本集合”收敛为 **离线可复用工具**，并在必要时逐步沉淀为 `src/plot_factory/` 能力区。

核心原则：
- KISS：先做 1–2 张最有价值的图（学习曲线、混淆矩阵）
- 强解耦：基于稳定产物（`manifest.json`、`metrics.csv`），尽量不绑定模型内部结构
- Optional import：缺依赖时降级并写入 eligibility，不崩溃、不影响训练

交付优先级（建议顺序）：
- WP0（post-run 检查）→ WP1（离线绘图）→ WP3（plot_factory 骨架）→ WP4（plotters）

目标：把仓库根目录的 `plot/`（当前为脚本集合、且被 `.gitignore` 忽略）整理为**可维护、可复用、可选启用**的绘图能力。
本计划默认采用“**离线脚本 + 配置文件**”方式落地：不集成到 `main.py`，不引入新的 pipeline。
后续如确有必要，再把核心绘图能力迁移进主仓库能力区 `src/plot_factory/`（供脚本复用）。

- 不破坏单一入口：`python main.py --config <yaml> [--override ...]`
- 不新增 YAML 第 6 个一级 block；开关挂 `trainer.extensions.*`
- 默认不依赖额外绘图库（如 `seaborn/scienceplots`）；缺依赖时**可审计降级**、不崩训练
- 以“最简单可行”为默认（KISS / 奥卡姆剃刀 / 第一性原理 / 渐进式开发）
- 尽量解耦：绘图逻辑不绑定某个 model/task 的内部结构，优先基于**稳定产物**（`manifest.json`、`metrics.csv` 等）

---

## 0) 现状盘点（为什么要迁移）

当前 `plot/` 目录的典型问题（以脚本内容为证）：

- 通过 `sys.path.append`、硬编码绝对路径、`post.*`/`model.*` 等旧路径引用，和 vibench 主仓库结构不一致
- 多处直接 `.cuda()`，CPU 环境会崩溃（不符合“设备可配置”）
- 依赖 `scienceplots/seaborn` 等非核心依赖，且缺少 optional-import 降级
- 输出路径混乱（有的写 `plot/output`、有的写 `./plot_dir`），不利于追溯与自动化收集

因此迁移的核心不是“搬文件”，而是把“脚本集合”收敛成：

1) **post-run 检查脚本（离线工具）**：检查一次 run 完成后“预期产物”是否落盘  
2) **plot 脚本（离线工具）**：对已有 run 目录离线绘图（可用独立 YAML 配置控制）  
3) （后续）**plot_factory（能力区）**：把脚本中稳定的绘图逻辑沉淀到 `src/plot_factory/` 供复用  
4) （可选）**plot extension（门禁）**：如确实需要训练结束自动出图，再考虑加入 callback（保持 best-effort）

---

## 1) 迁移原则（先定边界，避免过度工程化）

### 1.1 KISS（默认最简单可行）
- 先支持 1–2 个最有价值且**输入稳定**的图（例如：学习曲线、混淆矩阵），其余脚本先“归档/保留 TODO”
- 先做“读现有产物绘图”，不要立刻引入复杂的“训练时 hook/pred dump”

### 1.2 强解耦：以“稳定产物”为接口
优先从以下文件读取绘图输入（顺序从稳定到不稳定）：
1) `<run_dir>/artifacts/manifest.json`（SSOT；已由 `ManifestWriterCallback` 生成）
2) `<run_dir>/logs/**/metrics.csv`（CSVLogger 输出；学习曲线/标量图）
3) `<run_dir>/test_result_*.csv`（流水线落盘）
4) （可选增量）`<run_dir>/artifacts/predictions.npz`（后续再引入，专供混淆矩阵/样本可视化）

### 1.3 Optional import + 可审计降级
- `matplotlib`：作为基础依赖（若未来要变为 optional，需要再评估）
- `seaborn/scienceplots/plotly`：必须 optional import；缺失则跳过对应图，并写入 `plot_eligibility.json`

### 1.4 输出位置统一
建议统一写到：
- `<run_dir>/figures/`（与 manifest 字段 `figures_dir` 一致，最小改动）
并（可选）补一个可审计索引：
- `<run_dir>/artifacts/plots/plot_eligibility.json`（每个 plotter ok/skip + reasons）

---

## 2) 目标结构（src/plot_factory）

说明：本计划的 P0/P1 交付以 “scripts + config” 为主；`src/plot_factory/` 作为 P2+ 的收敛目标。

建议的最小目录结构（先骨架，后填充；避免一次性铺太多抽象）：

```
src/plot_factory/
  README.md
  __init__.py
  style.py                  # 统一 matplotlib 风格/字体/后端（安全默认）
  io.py                     # 读取 manifest/metrics/test_result 的小工具（纯函数）
  plotters/
    __init__.py
    learning_curve.py       # P0：学习曲线（只依赖 metrics.csv）
    confusion_matrix.py     # P1：混淆矩阵（依赖 predictions.npz；无则 skip）
  legacy/                   # 先归档旧脚本（不保证可用），逐步消化
    README.md
```

配套（可选）：
- `src/trainer_factory/extensions/plot.py`：`PlotWriterCallback`（与 `manifest.py` 同风格，必须 best-effort）
- `scripts/plot_run.py`：离线绘图 CLI（读 run_dir/manifest，执行 plotters）

---

## 3) 一步步落地计划（建议按 PR/WP 切分）

> 每一步都满足：不影响现有 demo/tests；脚本默认只读 run_dir，不改变训练流程。

### WP0：Post-run 预期产物检查（先解决“有没有产物”）
实现：
- 新增脚本：`scripts/uxfd_postrun.py`
  - 支持 `--config <yaml>`（独立配置，不混入 vibench 5-block）
  - 发现 run 目录（扫描 `**/artifacts/manifest.json` 或用户显式给 run_dirs）
  - 检查“必需/可选”产物是否存在（pattern + glob）
  - 结果写入 `<run_dir>/artifacts/plots/plot_eligibility.json`（仅记录；不影响训练）
- 新增示例配置：`paper/UXFD_paper/merge_uxfd/12_23/uxfd_postrun_config_example.yaml`

默认建议检查项（可在 config 覆盖）：
- 必需：
  - `artifacts/manifest.json`
  - `config_snapshot.yaml`（若当前 run 会写）
- 可选（存在就记录，不存在也不算失败）：
  - `logs/**/metrics.csv`
  - `test_result_*.csv`
  - `figures/`（若开启绘图）

验收：
- `python scripts/uxfd_postrun.py --config paper/UXFD_paper/merge_uxfd/12_23/uxfd_postrun_config_example.yaml`
  在没有任何 run 的情况下不会崩溃；有 run 时输出汇总并写 eligibility。

---

### WP1：离线绘图（先服务“已有 run 目录”，不改训练）
实现（仍在 `scripts/uxfd_postrun.py` 内，或拆成 `scripts/plot_run.py`）：
- P0：学习曲线（仅依赖 `logs/**/metrics.csv`）
- P1：混淆矩阵（仅当存在 `artifacts/predictions.npz` 时才画；否则 skip 并写 reason）

验收：
- 对任意已有 run 目录：
  - 有 `metrics.csv` 则生成 `figures/learning_curve.png`
  - 无 predictions 时，混淆矩阵不报错，只写 skip

---

### WP2：盘点与映射（整理 legacy，逐步迁移为 plotters）
产物：
- 一张“脚本 → plotter”的映射表 + 依赖/输入/输出说明（放在 `src/plot_factory/legacy/README.md` 或本文件追加）

建议先把 `plot/` 脚本按“是否值得产品化”分 3 类：
1) **P0 立刻产品化**：学习曲线、（可选）混淆矩阵  
2) **P1 需要补产物才能画**：attention/滤波器演化/特征可视化  
3) **Legacy 仅归档**：强依赖旧 `model.*`/`post.*` 路径且难以复用的脚本

验收：
- 确定最终输出目录：`figures/` vs `artifacts/plots/`（二选一，建议 `figures/`）

---

### WP3：plot_factory 骨架（供脚本复用，不做 main 集成）
实现：
- 新增 `src/plot_factory/style.py`：统一 `configure_matplotlib(...)`
  - 移除绝对字体路径；中文字体采用“若系统有则用，否则不报错”
  - 强制 headless backend：在无显示环境使用 `Agg`（避免服务器崩）
- 新增 `src/plot_factory/io.py`：读取 `manifest.json`、发现 `metrics.csv`、发现 `test_result_*.csv`
- 新增 `src/plot_factory/README.md`：职责边界、最小用法、输出约定

验收：
- `python -c "import src.plot_factory"` 不报错
- 不修改任何 pipeline/trainer 行为

---

### WP4：P0 plotters（只依赖稳定输入）
实现：
1) `learning_curve.py`
   - 读 `<run_dir>/logs/**/metrics.csv`（CSVLogger）
   - 画 train/val loss、acc 等（有啥画啥），缺列则跳过
2) `confusion_matrix.py`（先做“可选”版本）
   - 若存在 `<run_dir>/artifacts/predictions.npz`（包含 `y_true`、`y_pred` 或 `logits`）则绘制
   - 若不存在：写 skip reason（不要在这里强行引入训练时 dump）

验收：
- CLI 跑完后：`figures/` 下出现 `learning_curve.png`（或同等命名）
- 混淆矩阵在缺少 predictions 时也不报错，只写 skip

---

### WP5：训练器扩展（可选启用，默认不做）
实现：
- 新增 `src/trainer_factory/extensions/plot.py`：
  - `PlotWriterCallback(on_test_end/on_fit_end)` 调用 plot_factory 核心逻辑
  - 开关：`trainer.extensions.plot.enable`（默认 false 或 true 都行，但必须 safe）
  - 仅主进程写文件（参考 `ManifestWriterCallback` 的 `is_main_process`）
- 更新 `src/trainer_factory/extensions/README.md`：列出 `plot`

验收：
- 关闭 plot：训练流程无变化
- 开启 plot：训练结束后 `figures/` 生成图；失败不影响训练

---

### WP6：补齐“可画图所需的最小产物”（按需再做，避免过早）
当且仅当你确认“混淆矩阵/样本可视化必须在主流程自动产出”时，再做此步：

- 在 task/test_step 或 callback 中增加一个**最小**的预测落盘：
  - `<run_dir>/artifacts/predictions.npz`（例如：`y_true`、`y_pred`、可选 `logits`）
- 同步更新 `manifest.json` 增加字段 `predictions_path`

验收：
- `confusion_matrix.py` 在无 seaborn 时也能画（matplotlib fallback），并且 CPU 可跑

---

### WP7：消化/迁移 legacy 脚本（长期，逐个吃掉）
策略：
- 先把 `plot/` 原脚本复制到 `src/plot_factory/legacy/`，只做最小清理：
  - 移除 `sys.path` hack
  - 删除 `.cuda()` 强制（改为跟随输入 device）
  - 修复 `post.*`/`model.*` 引用（指向 `src.plot_factory` / `src.model_factory`）
- 每次只“产品化”一个脚本的核心价值，变成 `plotters/*.py`

完成标志：
- `plot/` 目录清空并删除（或仅保留输出，不再存代码）
- `.gitignore` 不再忽略 `plot/` 代码（但继续忽略绘图输出目录）

---

## 4) 迁移映射建议（把现有 plot 脚本拆成可复用 plotters）

| 现有脚本 | 主要价值 | 问题 | 建议归宿 |
|---|---|---|---|
| `plot/A1_plot_config.py` | 统一风格/字体 | 硬编码字体路径、依赖 scienceplots | `src/plot_factory/style.py`（P0） |
| `plot/pretraining_plot.py` | 预训练预测可视化 | sys.path/绝对路径、与主入口脱钩 | 先做 CLI/plotter（P1+） |
| `plot/A4_confusion_plus_noise_task.py` | 混淆矩阵、噪声鲁棒曲线 | `.cuda()`、旧路径 `post.*` | 拆成 `confusion_matrix.py` + `robustness_snr.py`（后续） |
| `plot/A5_plot_filters.py` | 滤波器响应/演化 | 引用旧 `model.Signal_processing` | 等 UXFD/SP 模块稳定后再迁移（P2） |
| `plot/A7_features.py` | 特征可视化 | 强依赖模型接口 | 后续结合 explain/report 的产物再做（P2+） |
| `plot/A10_model_parse.py` | 模型结构/attention 解析 | 强依赖旧模型类 | 更适合做 explain artifacts，不建议直接 plot_factory P0 做 |

---

## 5) 需要你确认的关键决策（避免返工）

| 决策点 | 选项 | 推荐 | 影响 |
|--------|------|------|------|
| 绘图输出目录 | `<run_dir>/figures/` / `<run_dir>/artifacts/plots/` | `<run_dir>/figures/` | 与 manifest 字段 `figures_dir` 一致、最少改动 |
| plot 默认开关 | `trainer.extensions.plot.enable: false/true` | `false` | 默认不自动出图，避免影响主流程 |
| predictions 落盘 | 主流程自动 / 手动（仅对已有文件画图） | 手动 | 若要自动落盘需做 WP6（侵入性更高） |
