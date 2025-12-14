# PHM-Vibench Bug识别执行计划

## 概述
本执行计划基于GLM（Granular Level Manual）的4个步骤，提供详细的执行指导。

## 当前状态
- 分支：`lqfix_25-12`
- 工作目录：`docs/LQ_fix/12_14/`
- 目标：系统性识别和文档化所有Bug

## 执行流程

### 阶段1：Preflight（准备阶段）- 0.5天

#### 1.1 环境检查
```bash
# 确认分支
git status --porcelain=v1 -b
git branch --show-current

# 检查工作区是否干净
git status
```

#### 1.2 创建目录结构
```bash
# 主目录
mkdir -p docs/LQ_fix/12_14/{bugs,reports,archive}

# 扫描日志目录
mkdir -p docs/LQ_fix/12_14/reports/scan_logs
```

#### 1.3 初始化文档文件
```bash
# 创建空的主文档
cat > docs/LQ_fix/12_14/BUGS.md << 'EOF'
# PHM-Vibench Bug总览

## 快速导航
- [Bug索引](BUG_INDEX.md)
- [Bug模板](BUG_TEMPLATES.md)
- [汇总报告](reports/BUG_SUMMARY.md)

## 优先级统计
- P0（致命）：0
- P1（严重）：0
- P2（中等）：0
- P3（轻微）：0

## 模块分布
- [配置系统](bugs/configuration.md)：0
- [数据工厂](bugs/data_factory.md)：0
- [模型工厂](bugs/model_factory.md)：0
- [任务工厂](bugs/task_factory.md)：0
- [训练器工厂](bugs/trainer_factory.md)：0
- [管道系统](bugs/pipelines.md)：0
- [文档问题](bugs/docs.md)：0

---

*最后更新：2025-12-14*
EOF

# 创建Bug模板
cat > docs/LQ_fix/12_14/BUG_TEMPLATES.md << 'EOF'
# Bug报告模板

## Bug ID命名规则
格式：`BUG-YYYYMMDD-XXX`（例如：BUG-20251214-001）

## 标准Bug报告格式
```markdown
## BUG-YYYYMMDD-XXX: 简短描述
- **优先级**: P0/P1/P2/P3
- **类型**: configuration/data/model/task/trainer/pipelines/perf/docs/other
- **状态**: open/triaged/verified/fix_planned/won't_fix/fixed
- **影响范围**: 受影响的模块/功能/数据集
- **复现步骤**:
  1. 命令：`python main.py --config xxx.yaml`
  2. 配置：xxx
  3. 覆盖参数：`--override key=value`
- **期望行为**: 描述应该发生什么
- **实际行为**: 描述实际发生了什么
- **定位线索**:
  - 文件：`src/xxx/yyy.py:123`
  - 函数：`function_name()`
  - 错误信息：`xxx`
- **临时绕过**: 如果有临时解决方案
- **相关Issue**: #issue_number
```

## 优先级定义
- **P0（致命）**: 系统崩溃、数据丢失、完全无法使用
- **P1（严重）**: 核心功能失效、主要工作流阻塞
- **P2（中等）**: 功能受限、有临时方案、性能问题
- **P3（轻微）**: 文档错误、UI问题、非核心功能缺陷
EOF

# 创建Bug索引
cat > docs/LQ_fix/12_14/BUG_INDEX.md << 'EOF'
# Bug索引

## 按优先级
### P0 - 致命
（暂无）

### P1 - 严重
（暂无）

### P2 - 中等
（暂无）

### P3 - 轻微
（暂无）

## 按类型
- **configuration**: [配置系统](bugs/configuration.md)
- **data**: [数据工厂](bugs/data_factory.md)
- **model**: [模型工厂](bugs/model_factory.md)
- **task**: [任务工厂](bugs/task_factory.md)
- **trainer**: [训练器工厂](bugs/trainer_factory.md)
- **pipelines**: [管道系统](bugs/pipelines.md)
- **docs**: [文档问题](bugs/docs.md)
- **perf**: 性能问题
- **other**: 其他问题

## 按日期
- 2025-12-14：0个

---

*索引格式：BUG-ID | 标题 | 优先级 | 类型 | 状态*
EOF

# 创建模块Bug文档
for module in configuration data_factory model_factory task_factory trainer_factory pipelines docs; do
    cat > "docs/LQ_fix/12_14/bugs/${module}.md" << EOF
# ${module^} 模块 Bug

## 概述
本文档记录 ${module} 模块相关的 Bug。

---

*最后更新：2025-12-14*
EOF
done
```

### 阶段2：Auto Scan（自动扫描）- 1天

#### 2.1 扫描命令执行
```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench

# 1. TODO/FIXME/HACK
rg -n "\b(TODO|FIXME|HACK)\b" src/ configs/ dev/ docs/ test/ \
  | tee docs/LQ_fix/12_14/reports/scan_logs/rg_todo_fixme_hack.txt

# 2. 可疑异常处理
rg -n "except\s*:" src/ \
  | tee docs/LQ_fix/12_14/reports/scan_logs/rg_except_bare.txt

rg -n "except\s+Exception" src/ \
  | tee docs/LQ_fix/12_14/reports/scan_logs/rg_except_exception.txt

rg -n "except\s+BaseException" src/ \
  | tee docs/LQ_fix/12_14/reports/scan_logs/rg_except_baseexception.txt

rg -n "except\b.*:\s*pass\b" -S src/ \
  | tee docs/LQ_fix/12_14/reports/scan_logs/rg_except_pass.txt

# 3. assert/raise
rg -n "\bassert\b|\braise\b" src/ \
  | tee docs/LQ_fix/12_14/reports/scan_logs/rg_assert_raise.txt

# 4. 路径/环境变量
rg -n "(/home/|\$HOME|PROJECT_HOME|data_dir|output_dir|save/)" -S src/ configs/ \
  | tee docs/LQ_fix/12_14/reports/scan_logs/rg_paths_envs.txt
```

#### 2.2 扫描结果初步分析
创建分析脚本：
```bash
cat > docs/LQ_fix/12_14/reports/analyze_scan_results.py << 'EOF'
#!/usr/bin/env python3
"""分析扫描结果并生成初步Bug列表"""

import os
from collections import defaultdict

def analyze_scan_results():
    results_dir = "docs/LQ_fix/12_14/reports/scan_logs"
    bugs = defaultdict(list)

    # 分析TODO/FIXME
    if os.path.exists(f"{results_dir}/rg_todo_fixme_hack.txt"):
        with open(f"{results_dir}/rg_todo_fixme_hack.txt") as f:
            for line in f:
                if "TODO" in line or "FIXME" in line:
                    # 提取文件路径和行号
                    parts = line.split(":")
                    if len(parts) >= 2:
                        file_path = parts[0]
                        line_num = parts[1]
                        bugs["todo"].append((file_path, line_num, line.strip()))

    # 分析异常处理
    for pattern in ["bare", "exception", "baseexception", "pass"]:
        filename = f"rg_except_{pattern}.txt"
        if os.path.exists(f"{results_dir}/{filename}"):
            with open(f"{results_dir}/{filename}") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        bugs[f"except_{pattern}"].append((parts[0], parts[1], line.strip()))

    # 输出分析结果
    print("扫描结果分析：")
    for category, items in bugs.items():
        print(f"\n{category}: {len(items)} 条")
        for item in items[:5]:  # 只显示前5条
            print(f"  - {item[0]}:{item[1]}")

if __name__ == "__main__":
    analyze_scan_results()
EOF

python docs/LQ_fix/12_14/reports/analyze_scan_results.py > docs/LQ_fix/12_14/reports/scan_analysis.txt
```

### 阶段3：Manual Review（手动审查）- 2天

#### 3.1 审查检查清单

创建审查记录模板：
```bash
cat > docs/LQ_fix/12_14/reports/manual_review_template.md << 'EOF'
# 手动审查记录

## 模块：[模块名]
### 审查日期：2025-12-14
### 审查人：[姓名]

#### 检查点1：[检查点名称]
- 状态：□ 通过 □ 问题 □ 需要进一步调查
- 问题描述：
- 影响：
- 相关文件：

#### 检查点2：[检查点名称]
...

#### 发现的Bug：
1. BUG-ID:
   - 优先级：
   - 描述：

#### 建议改进：
1.
EOF
```

#### 3.2 各模块审查重点

**配置系统 (`src/configs/`)**
- 配置加载逻辑
- 默认值处理
- 类型安全

**数据工厂 (`src/data_factory/`)**
- 数据集注册机制
- 内存管理
- 文件路径处理

**模型工厂 (`src/model_factory/`)**
- 组件命名一致性
- 检查点加载
- 依赖管理

**任务工厂 (`src/task_factory/`)**
- 批处理格式
- 损失函数
- 多任务处理

**训练器工厂 (`src/trainer_factory/`)**
- 设备管理
- 日志系统
- 检查点保存

**管道系统 (`Pipeline_*.py`)**
- 多阶段执行
- 配置继承
- 错误处理

### 阶段4：Triage & Docs（分级建档）- 1.5天

#### 4.1 Bug分级流程
1. **自动扫描结果 → Bug**
   - P0：裸 except、系统关键路径TODO
   - P1：异常处理不当、重要配置问题
   - P2：一般TODO、文档问题
   - P3：代码注释、非关键路径

2. **手动审查结果 → Bug**
   - 根据影响范围和复现难度确定优先级

#### 4.2 Bug建档步骤
对于每个发现的Bug：

1. **创建Bug ID**
   ```bash
   # 获取下一个Bug编号
   last_num=$(grep -c "BUG-20251214" docs/LQ_fix/12_14/BUG_INDEX.md)
   bug_id="BUG-20251214-$(printf "%03d" $((last_num + 1)))"
   ```

2. **添加到模块文档**
   - 编辑对应的 `bugs/module.md`
   - 使用标准格式添加Bug

3. **更新索引**
   - 在 `BUG_INDEX.md` 中添加条目
   - 更新统计数据

4. **更新主页面**
   - 在 `BUGS.md` 中更新计数

#### 4.3 生成汇总报告
```bash
cat > docs/LQ_fix/12_14/reports/BUG_SUMMARY.md << 'EOF'
# Bug汇总报告

## 总览
- 发现时间：2025-12-14
- Bug总数：0
- P0：0 个
- P1：0 个
- P2：0 个
- P3：0 个

## 模块分布
| 模块 | P0 | P1 | P2 | P3 | 总计 |
|------|----|----|----|----|------|
| 配置系统 | 0 | 0 | 0 | 0 | 0 |
| 数据工厂 | 0 | 0 | 0 | 0 | 0 |
| 模型工厂 | 0 | 0 | 0 | 0 | 0 |
| 任务工厂 | 0 | 0 | 0 | 0 | 0 |
| 训练器工厂 | 0 | 0 | 0 | 0 | 0 |
| 管道系统 | 0 | 0 | 0 | 0 | 0 |
| 文档问题 | 0 | 0 | 0 | 0 | 0 |

## 关键发现
1.
2.
3.

## 修复建议优先级
1. [P0/P1级Bug的修复建议]
2.
3.

## 复现要求
- 需要数据集：是/否
- 需要GPU：是/否
- 特殊环境变量：是/否

---
*报告生成时间：2025-12-14*
EOF
```

## 执行时间线

| 时间 | 任务 | 产出 |
|------|------|------|
| Day 1 上午 | Preflight准备 | 目录结构、空文档 |
| Day 1 下午 | Auto Scan | 扫描结果、初步分析 |
| Day 2 上午 | Manual Review (1) | 配置、数据、模型工厂 |
| Day 2 下午 | Manual Review (2) | 任务、训练器、管道 |
| Day 3 上午 | Manual Review (3) | 文档、完成审查 |
| Day 3 下午 | Triage & Docs | Bug分级、建档 |
| Day 4 上午 | 汇总报告 | 最终报告、索引更新 |

## 注意事项

1. **备份原始数据**：所有扫描输出保存在 `scan_logs/`
2. **标准化格式**：严格遵循Bug模板
3. **及时更新**：每发现一个Bug立即更新索引
4. **交叉验证**：P0/P1级Bug需要二次确认
5. **文档同步**：最终要更新README中的链接

## 完成标准

- [ ] 所有扫描命令执行完成
- [ ] 每个模块审查记录完整
- [ ] 所有发现的Bug已建档
- [ ] Bug索引和统计更新
- [ ] 汇总报告生成
- [ ] 所有文档链接有效