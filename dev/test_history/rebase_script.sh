#!/bin/bash

# HSE提交整理脚本
# 目标：将17个提交整理为7个清晰的提交

echo "开始重新组织HSE相关提交..."

# 重置到417befb之前的状态
git reset --hard 417befb

# 按新的顺序重新应用提交
echo "重新应用HSE核心功能实现..."

# 1. HSE核心功能实现（合并为1个提交）
git cherry-pick 52361f5  # 基础提交
git cherry-pick bfa05b4 --no-commit
git cherry-pick 6532f10 --no-commit
git cherry-pick deb8971 --no-commit
git cherry-pick 1ca9d7f --no-commit
git cherry-pick 8a84564 --no-commit
git cherry-pick 0de238c --no-commit

# 提交HSE核心功能
git commit -m "feat(hse): Implement HSE Industrial Contrastive Learning core components

- Add HSE heterogeneous contrastive learning implementation plan
- Implement project templates for requirements, design, structure, tasks, and tech stack
- Add MomentumEncoder and ProjectionHead for contrastive learning
- Introduce comprehensive contrastive loss functions (InfoNCE, Triplet, SupConLoss, etc.)
- Implement HSE Prompt guided contrastive learning configurations
- Add Two-Stage Training Controller for HSE Prompt-Guided Contrastive Learning
- Introduce E_01_HSE_v2 for prompt-guided hierarchical signal embedding
- Add comprehensive performance benchmarking for HSE Prompt components"

echo "HSE核心功能实现已合并"

# 2. HSE Pipeline和实验框架（合并为1个提交）
git cherry-pick 4d25a0d  # 基础提交
git cherry-pick df0184c --no-commit
git cherry-pick ec28971 --no-commit
git cherry-pick 8201067 --no-commit

# 提交HSE Pipeline
git commit -m "feat(pipeline): Add HSE experiment pipeline and specifications

- Add HSE Paper Pipeline with requirements, design, and tasks for automated execution
- Implement unified metric learning pipeline with complete user guide
- Simplify HSE specifications for unified metric learning approach
- Add generated task commands for HSE paper pipeline"

echo "HSE Pipeline和实验框架已合并"

# 3. Claude工具（保持独立）
echo "重新应用Claude工具提交..."
git cherry-pick b330048  # auto_commit_config.yaml
git cherry-pick 73e5ec6  # claude_auto_commit.py
git cherry-pick bca0be7  # claude_commit.sh
git cherry-pick c10e3b5  # 文档更新

echo "Claude工具提交已保持独立"

# 4. 验证和监控工具（合并为1个提交）
git cherry-pick 33fccad  # 基础提交
git cherry-pick 9f941ee --no-commit

# 提交验证工具
git commit -m "feat(validation): Add comprehensive validation and metrics reporting tools

- Add MetricsMarkdownReporter and SystemMetricsTracker for comprehensive metrics reporting
- Implement OneEpochValidator for comprehensive 1-epoch training validation
- Support system-level metrics analysis and performance comparison
- Include diagnostic insights and debugging capabilities"

echo "验证和监控工具已合并"

echo "提交重新组织完成！"
echo "新的提交结构："
git log --oneline -10