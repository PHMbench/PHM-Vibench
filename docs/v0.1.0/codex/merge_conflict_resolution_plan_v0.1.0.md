# v0.1.0 → main 合并冲突解决计划（已执行归档）

> 目标：在合并 `release/v0.1.0` → `main` 时，给出**逐类文件**的取舍策略，  
> 尽量保持 v0.1.0 的核心设计（config / factory / pipeline），并避免引入新 bug。  
> 本文件现作为「已执行的操作记录」，便于回顾和后续版本参考。

**实际执行情况（Summary）**

- 你已在本地按照本计划完成了 `release/v0.1.0` → `main` 的合并；
- 合并逻辑遵循本文件中的取舍策略：
  - 核心代码与配置（`main.py`、各 Pipeline、config_utils、data_factory、model_factory、task_factory、trainer、utils 等）统一采用 v0.1.0 版本；
  - FS/GFS sampler 与 ID 搜索逻辑使用 v0.1.0 修复方案；
  - config 目录采用 base + demo + reference 的新结构；
  - 多任务脚本整体迁移到 `dev/script/**` 目录，老的 `script/Vibench_paper/foundation model/**` 保留或清理按照计划执行；
  - 测试配置迁移到 `dev/test_history/**`；
  - v0.0.9 时代的 demo/config 移动到 `configs/v0.0.9/**` 归档。

下面的内容保留为当时使用的操作步骤，后续若有新的 release 需要类似的 merge，可以复用此模式。

---

## 1. 操作前置（在 main 上）

1. 确保本地有 `release/v0.1.0`：

   ```bash
   git branch
   # 应能看到 release/v0.1.0
   ```

2. 切换到 `main` 并拉最新远程：

   ```bash
   git checkout main
   git pull origin main
   ```

3. 开始合并（会触发冲突）：

   ```bash
   git merge --no-ff release/v0.1.0 -m "release: v0.1.0"
   ```

4. 用下面的文件分类策略逐个解决冲突，最后：

   ```bash
   git status        # 确认无 Unmerged
   git add .
   git commit        # 完成 merge commit
   git push origin main
   git tag -a v0.1.0 -m "PHM-Vibench v0.1.0"
   git push origin v0.1.0
   ```

---

## 2. 核心代码 / 配置：统一采用 v0.1.0 版本（theirs）

这些是 v0.1.0 的“核心设计变更”，一律选择 **release/v0.1.0 侧（theirs）**，避免退回旧实现。

在冲突状态下，对下列文件执行：

```bash
git checkout --theirs main.py
git checkout --theirs src/Pipeline_01_default.py
git checkout --theirs src/Pipeline_02_pretrain_fewshot.py
git checkout --theirs src/Pipeline_03_multitask_pretrain_finetune.py
git checkout --theirs src/configs/config_utils.py
git checkout --theirs src/data_factory/data_factory.py
git checkout --theirs src/data_factory/dataset_task/ID_dataset.py
git checkout --theirs src/data_factory/samplers/Get_sampler.py
git checkout --theirs src/model_factory/model_factory.py
git checkout --theirs src/model_factory/ISFM/M_01_ISFM.py
git checkout --theirs src/model_factory/ISFM/task_head/H_01_Linear_cla.py
git checkout --theirs src/task_factory/task_factory.py
git checkout --theirs src/task_factory/task/MT/multi_task_lightning.py
git checkout --theirs src/trainer_factory/Default_trainer.py
git checkout --theirs src/utils/utils.py
git checkout --theirs dev/test_history/run_tests.py
git checkout --theirs dev/test_history/pytest.ini
git checkout --theirs dev/test_history/requirements-test.txt
```

> 说明：
> - 这些文件承载了 v0.1.0 的 config 合并逻辑、pipeline 接口、sampler 修复、ISFM/TaskFactory 重构等；
> - main 上的版本较旧 / 不完整，统一采用 v0.1.0 可以最大程度保证新设计一致。

---

## 3. 文档：保留新结构，旧文档迁移到 `docs/past`

合并时常见冲突包括：

- `docs/app_usage.md`
- `docs/developer_guide.md`
- `docs/multi_task_phm_foundation_model.md`
- `docs/multitask_pretrain_finetune_guide.md`
- `docs/streamlit_prompt.md`

在 v0.1.0 中，这些文件已被整体迁移到 `docs/past/`，目的是：

- 避免旧文档与新 config/pipeline 描述混淆；
- 将过去的说明保留成“历史档案”，新版本只指向 v0.1.0 文档。

**推荐策略：**

- 对根目录 `docs/*.md` 的冲突，采用 **release/v0.1.0 的方案**（即允许它们被移动到 `docs/past`）：

  ```bash
  git checkout --theirs docs/app_usage.md
  git checkout --theirs docs/developer_guide.md
  git checkout --theirs docs/multi_task_phm_foundation_model.md
  git checkout --theirs docs/multitask_pretrain_finetune_guide.md
  git checkout --theirs docs/streamlit_prompt.md
  ```

- 同时保留 `docs/v0.1.0/**` 与 `docs/v0.1.0/done/**` 中的新文档（这些在 merge 过程中不会与 main 冲突，`--theirs` 已隐含保持）。

> 最终结果：  
> - 旧的长文档在 `docs/past/`；  
> - v0.1.0 相关说明集中在 `docs/v0.1.0/`；  
> - 根目录 README 只链接到 v0.1.0 更新摘要。

---

## 4. README / AGENTS / .gitignore：以 v0.1.0 为主，必要时手工微调

1. `AGENTS.md`
   - main 中曾被删除，v0.1.0 中有更新版本；
   - 建议采用 **v0.1.0 版本**：

     ```bash
     git checkout --theirs AGENTS.md
     ```

2. `README.md` / `README_CN.md`
   - v0.1.0 中已对目录结构、config 系统、demo 流水线做了说明；
   - 建议直接采用 **v0.1.0 版本**：

     ```bash
     git checkout --theirs README.md
     git checkout --theirs README_CN.md
     ```

3. `.gitignore`
   - 冲突主要来自旧的测试/脚本目录 vs 新的 `dev/**` / `docs/v0.1.0/**` / `configs/v0.0.9/**`；
   - 建议以 **v0.1.0 版本为基础**：

     ```bash
     git checkout --theirs .gitignore
     ```

   - 若你在 main 上之前手动加过忽略规则，合并后可以用：

     ```bash
     git diff HEAD~1 .gitignore
     ```

     检查是否需要手工补回少量规则（此处不强制，且不会影响运行 correctness）。

---

## 5. 多任务脚本 & 论文脚本：采用 v0.1.0 的整理方案

冲突较多的部分集中在多任务脚本与论文脚本路径上，例如：

- `script/Vibench_paper/foundation model/*`
- `dev/script/Vibench_paper/foundation model/*`
- `dev/script/Vibench_paper/foundation_model/*`

在 v0.1.0 中，这些脚本被系统性迁移到 `dev/script/**`，并统一为更规范的目录结构（通常去掉路径中的空格，如 `foundation_model`）。

**原则：**

- 运行核心 demo（6 个 v0.1.0 demo）不依赖这些脚本，它们主要是论文实验批量脚本；
- v0.1.0 中的脚本布局更合理，建议 **整体采用 release/v0.1.0 的版本，保留 dev/script 下的文件，删除旧的 script 目录中的重复脚本**。

**冲突解决建议：**

1. 对 `dev/script/Vibench_paper/**` 中的冲突文件，一律：

   ```bash
   git checkout --theirs dev/script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml
   git checkout --theirs dev/script/Vibench_paper/foundation_model/multitask_B_06_TimesNet.yaml
   git checkout --theirs dev/script/Vibench_paper/foundation_model/multitask_B_08_PatchTST.yaml
   git checkout --theirs dev/script/Vibench_paper/foundation_model/multitask_B_09_FNO.yaml
   git checkout --theirs dev/script/Vibench_paper/foundation_model/run.sbatch
   git checkout --theirs dev/script/Vibench_paper/foundation_model/run_multitask_a100.sh
   git checkout --theirs dev/script/Vibench_paper/foundation_model/run_multitask_experiments.sh
   git checkout --theirs dev/script/Vibench_paper/foundation_model/test_a100_local.sh
   git checkout --theirs dev/script/Vibench_paper/foundation_model/test_multitask.sh
   git checkout --theirs dev/script/Vibench_paper/plans/fix_multitask_model_init_20250831.md
   ```

2. 对老的 `script/Vibench_paper/foundation model/*` 冲突路径：

   - 这些脚本已经有等价版本被迁移到 `dev/script/...`；
   - 建议接受 **删除**，即不做额外 checkout，保持 v0.1.0 的目录结构（`script/` 中不再保留这些文件）。

> 合并后可以执行：
>
> ```bash
> ls script/Vibench_paper
> ls dev/script/Vibench_paper
> ```
>
> 确认只保留 `dev/script` 下的脚本，未来你要跑论文脚本就从 dev 下调用即可。

---

## 6. 其他冲突文件

除上述核心类别外，还有少量冲突文件（例如 `dev/main_test.py` 等），建议统一采用 v0.1.0 版本：

```bash
git checkout --theirs dev/main_test.py
```

如果在执行 `git checkout --theirs <file>` 后 `git diff --name-only --diff-filter=U` 仍有残余文件未解决，可按以下策略处理：

- 若是新增加的 dev 辅助文件（`dev/**`、`docs/v0.1.0/**` 等）：
  - 通常只在 release/v0.1.0 中存在，直接 `git checkout --theirs` 即可；
- 若是 main 上你不在意的小脚本（例如临时测试脚本）：
  - 可以用 `git checkout --ours <file>` 保留 main 版本，或直接删除后 `git add`，不会影响核心 v0.1.0 行为。

---

## 7. 最终检查与提交

完成上述 `git checkout --theirs` / `--ours` 操作后：

1. 检查是否还有未解决冲突：

   ```bash
   git status
   git diff --name-only --diff-filter=U
   ```

   - 若 `--diff-filter=U` 没有输出，说明冲突已全部解决。

2. 将变更加入暂存区并提交 merge：

   ```bash
   git add .
   git commit    # merge commit，message 可以保留 "release: v0.1.0"
   git push origin main
   ```

3. 打 tag：

   ```bash
   git tag -a v0.1.0 -m "PHM-Vibench v0.1.0"
   git push origin v0.1.0
   ```

---

## 8. 合并后的验证建议（可选）

合并完成后，可选择性再跑一遍最小 demo 验证，确保 main 上行为与在 `release/v0.1.0` 上一致：

```bash
python main.py --config configs/demo/01_cross_domain/cwru_dg.yaml --override trainer.num_epochs=1 --override data.num_workers=0
python main.py --config configs/demo/02_cross_system/multi_system_cddg.yaml --override trainer.num_epochs=1 --override data.num_workers=0
python main.py --config configs/demo/03_fewshot/cwru_protonet.yaml --override trainer.num_epochs=1 --override data.num_workers=0
python main.py --config configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml --override trainer.num_epochs=1 --override data.num_workers=0
python main.py --config configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml --override trainer.num_epochs=1 --override data.num_workers=0
python main.py --config configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml --override trainer.num_epochs=1 --override data.num_workers=0
```

如需进一步精简或指定某些脚本/文档保留 main 版本，我可以根据你对这些文件的偏好再给一份“差异最小化”的变体计划。 
