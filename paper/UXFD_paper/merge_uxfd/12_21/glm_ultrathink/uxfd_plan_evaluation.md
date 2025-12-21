# UXFD 合并研究/重构计划评估

## 评估结论

基于对 `init_plan.md` 和 `model_alignment_plan.md` 的详细审阅，本计划在技术架构设计上相当清晰和深入，但在项目管理和执行细节方面存在一些需要澄清的地方。总体而言，这是一个技术上经过深思熟虑的整合方案。

## 详细评估

### 1. 技术实施细节 ✅ (优秀)

**优点：**
- **架构设计清晰**：TSPN_UXFD 的设计很好地平衡了保持上游 TSPN 结构和增加可配置性的需求
- **技术路径明确**：从 OperatorRegistry、LayoutSpec/Adapter、HookStore 到 explain_factory，整个技术链条逻辑连贯
- **配置映射策略合理**：paper_id → operator_graph 的映射方式，以及同时支持上游式和 stage 化配置的兼容性设计
- **风险识别准确**：对布局不一致、依赖污染、解释模块接入等关键风险点有充分认识

**需要澄清的技术细节：**
- OperatorRegistry 的具体实现位置和命名规范（如 `E_`, `B_`, `H_` 前缀是否适用于算子）
- 2D 分支的具体实现细节（路线 A 和 B 的切换机制）
- metadata schema 的具体字段定义和验证逻辑

### 2. 项目管理规划 ⚠️ (需要加强)

**优点：**
- PR 划分合理：5个 PR 的职责边界清晰
- 验收标准明确：每个 PR 都有可执行的验收命令
- 风险识别全面：列出了 8+ 条风险及缓解措施

**需要完善的项目管理要素：**
- **时间规划缺失**：没有预计的时间线和里程碑
- **资源分配不明**：谁负责哪个 PR，需要多少人力
- **依赖关系图**：PR 之间的硬依赖和软依赖关系需要更明确
- **回滚策略细节**：每个 PR 的回滚方案需要更具体

### 3. PR 划分和验收标准 ✅ (清晰)

每个 PR 的职责定义清晰：
- PR0：submodule 落位（基础设施）
- PR1：TSPN 骨架（核心模型）
- PR2：配置映射（paper 接入）
- PR3：explain_factory（解释能力）
- PR4：collect/report（证据链）
- PR5：agent_factory（TODO 落盘）

验收标准具体且可执行，特别是 `python -m scripts.config_inspect` 等命令的运用。

### 4. Submodule 管理策略 ⚠️ (基本清晰，需要补充)

**清晰的方面：**
- 目录结构定义明确
- submodule 初始化策略与现有 paper/ 对齐
- VIBENCH.md 的作用和位置明确

**需要补充：**
- submodule 的版本管理策略（tag vs commit hash）
- submodule 更新流程（如何同步上游更新）
- CI/CD 如何处理 submodule（是否需要 shallow clone）

### 5. 配置映射策略 ✅ (设计合理)

paper_id → preset → operator_graph 的映射策略设计合理，特别是：
- 同时支持上游配置格式和新的 stage 化格式
- 生成脚本 `gen_uxfd_min_configs` 的设计降低了维护成本
- 版本化支持（paper_id@v1）考虑了演进需求

## 改进建议

### 高优先级改进

1. **补充实施时间表**
   - 每个 PR 的预计工期
   - 关键里程碑日期
   - 并行执行的可能性

2. **明确回滚策略**
   - 每个 PR 的回滚命令
   - 数据迁移的回滚方案
   - 配置兼容性保证

3. **细化验收标准**
   - 每个验收命令的期望输出
   - 性能基准（如果适用）
   - 文档更新要求

### 中优先级改进

4. **补充依赖关系图**
   - PR 之间的硬依赖
   - 可以并行执行的部分
   - 阻塞点识别

5. **增加资源规划**
   - 每个 PR 的复杂度评估
   - 所需技能组合
   - 代码审查安排

6. **完善风险缓解措施**
   - 每个风险的具体监控指标
   - 触发条件和应对流程
   - 负责人指定

## Git 分步提交策略

基于 PR 划分，建议采用以下分步提交策略：

### PR0: Submodule 落位（准备阶段）

```bash
# 1. 创建目录结构
git add paper/UXFD_paper/README.md
git add paper/UXFD_paper/README_SUBMODULE.md
git commit -m "feat: 添加 UXFD Paper submodule 入口结构

- 建立 paper/UXFD_paper/ 目录结构
- 添加 submodule 初始化指南
- 定义 7 个 paper_id 与目录映射关系

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 2. 添加第一个 submodule（示例）
git submodule add <repository_url> paper/UXFD_paper/fusion_1d2d
git commit -m "feat: 添加 fusion_1d2d 作为第一个 UXFD submodule

- 添加 1D-2D_fusion_explainable 作为 submodule
- 更新 .gitmodules 配置
- 添加 VIBENCH.md 模板

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### PR1: TSPN 骨架（核心模型）

```bash
# 1. 添加 OperatorRegistry 基础设施
git add src/model_factory/operator_registry.py
git add src/model_factory/operators/base.py
git commit -m "feat: 添加 OperatorRegistry 基础设施

- 实现 OperatorRegistry 注册机制
- 定义 base operator 接口和 LayoutSpec
- 支持算子的版本化管理

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 2. 实现 TSPN_UXFD 模型
git add src/model_factory/X_model/uxfd_tspn/
git add src/model_factory/model_registry.csv
git commit -m "feat: 实现 TSPN_UXFD 可配置模型

- 添加 TSPN_UXFD 兼容上游 TSPN 结构
- 实现 operator_graph 配置解析
- 集成 HookStore 机制

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 3. 添加最小算子集
git add src/model_factory/operators/
git commit -m "feat: 添加最小可用算子集合

- 实现 SP_1D 基础信号处理算子
- 添加 FE 特征提取算子
- 实现 LayoutAdapter 算子

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### PR2: 配置映射

```bash
# 1. 配置适配器
git add src/configs/config_adapter.py
git commit -m "feat: 添加 UXFD 配置适配器

- 实现 paper_id 到 operator_graph 的映射
- 支持上游配置格式转换
- 添加配置版本管理

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 2. 配置预设
git add configs/presets/uxfd/
git add scripts/gen_uxfd_min_configs.py
git commit -m "feat: 添加 UXFD 配置预设和生成脚本

- 为 7 篇 paper 添加预设配置
- 实现配置自动生成脚本
- 添加配置验证功能

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### PR3: Explain Factory

```bash
# 1. 核心框架
git add src/explain_factory/__init__.py
git add src/explain_factory/metadata_reader.py
git add src/explain_factory/eligibility.py
git commit -m "feat: 添加 explain_factory 核心框架

- 实现 metadata 统一读取机制
- 添加 ExplainReady 可用性检查
- 定义 metadata schema

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 2. 解释器实现
git add src/explain_factory/explainers/
git commit -m "feat: 实现核心解释器集合

- 添加 router_weights 内生解释器
- 实现 timefreq 时频解释器
- 添加 gradients 后验解释器
- 支持解释器降级策略

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### PR4: Collect/Report

```bash
# 1. 收集脚本
git add scripts/collect_uxfd_runs.py
git commit -m "feat: 添加 UXFD 运行收集脚本

- 实现 manifest.json 生成
- 添加跨 run 汇总功能
- 支持导出 CSV/JSON 格式

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# 2. manifest 集成
git add src/utils/manifest.py
git commit -m "feat: 集成 manifest 生成到主流程

- 在训练结束时自动生成 manifest
- 统一产物索引格式
- 支持跨版本兼容

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### PR5: Agent Factory

```bash
# 1. Agent 框架
git add src/agent_factory/__init__.py
git add src/agent_factory/distiller.py
git commit -m "feat: 添加 agent_factory 基础框架

- 实现 TODO 内容蒸馏机制
- 定义 frontmatter schema
- 预留 LLM 接入接口

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### 提交最佳实践

1. **每个 PR 多次提交**：按功能模块分批提交，保持每次提交的原子性
2. **清晰的提交信息**：使用 `feat:`、`fix:` 等前缀，说明改动内容
3. **添加 Co-Authored-By**：保留 AI 协助的可追溯性
4. **阶段性标签**：
   ```bash
   git tag -a v0.1-uxfd-pr0 -m "完成 PR0: Submodule 落位"
   git tag -a v0.1-uxfd-pr1 -m "完成 PR1: TSPN 骨架"
   ```
5. **分支管理**：
   ```bash
   # 为每个 PR 创建独立分支
   git checkout -b feature/uxfd-pr0
   # 完成后合并到主开发分支
   git checkout main
   git merge feature/uxfd-pr0 --no-ff
   ```

## 总结

这是一个技术上非常成熟的整合计划，对 PHM-Vibench 和 UXFD 的理解都很深入。技术方案设计合理，考虑了兼容性、可扩展性和可维护性。主要需要在项目管理维度进行补充，特别是时间规划、资源分配和更详细的执行流程。

建议在实施前：
1. 先完善时间表和资源分配
2. 制定更详细的回滚方案
3. 准备好每个 PR 的具体实施清单
4. 建立定期的进度回顾机制