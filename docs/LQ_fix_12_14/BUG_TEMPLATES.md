# Bug报告模板

## Bug ID命名规则
格式：`BUG-YYYYMMDD-XXX`（例如：BUG-20251214-001）

## 标准Bug报告格式

```markdown
## BUG-YYYYMMDD-XXX: 简短描述

- **优先级**: P0/P1/P2/P3
- **类型**: configuration/data/model/task/trainer/pipelines/perf/docs/other
- **状态**: open/triaged/verified/fix_planned/won't_fix/fixed
- **模块**: 具体模块名
- **文件**: `path/to/file:line_number`
- **发现时间**: YYYY-MM-DD HH:MM:SS

### 问题描述
详细描述Bug的问题和影响

### 代码位置
```python
# 展示相关代码片段
```

### 期望行为
描述应该发生的正确行为

### 实际行为
描述实际发生的错误行为

### 影响范围
- 受影响的功能/模块
- 影响的用户群体
- 可能的后果

### 复现步骤
1. 执行命令
2. 配置参数
3. 操作步骤

### 修复建议
提供具体的修复代码或方案

### 相关Issue
- #issue_number
- 相关的Bug链接

---
```

## 优先级定义

### P0（致命）
- 系统崩溃、数据丢失
- 安全漏洞
- 完全无法使用核心功能

### P1（严重）
- 核心功能无法使用
- 主要工作流阻塞
- 性能严重下降

### P2（中等）
- 功能部分受限
- 有临时解决方案
- 性能问题但不阻塞使用

### P3（轻微）
- 文档错误
- UI/UX问题
- 非核心功能的小缺陷

## Bug类型分类

### configuration
- 配置加载错误
- 类型安全问题
- 默认值问题
- YAML解析问题

### data
- 数据加载失败
- 内存管理问题
- 数据集特定问题
- 路径处理错误

### model
- 模型创建失败
- 组件不匹配
- 检查点加载问题
- 导入依赖问题

### task
- 任务执行错误
- 批处理格式问题
- 损失函数问题
- 采样策略问题

### trainer
- 训练流程中断
- GPU/CPU切换问题
- 日志记录问题
- 检查点保存问题

### pipelines
- 多阶段执行问题
- 配置继承错误
- 资源管理问题
- 错误传播问题

### performance
- 内存泄漏
- 性能瓶颈
- 资源使用不当

### docs
- 文档不一致
- 示例错误
- 链接失效

### other
- 其他未分类问题

## 状态流转

```
open → triaged → verified → fix_planned → fixed
  ↓         ↓
won't_fix (任何阶段)
```

## 示例Bug报告

```markdown
## BUG-20251214-001: H5文件句柄关闭时使用裸except

- **优先级**: P0
- **类型**: exception_handling
- **状态**: open
- **模块**: data_factory
- **文件**: `src/data_factory/H5DataDict.py:40`
- **发现时间**: 2025-12-14 16:23:29

### 问题描述
在关闭HDF5文件句柄时使用了裸except语句，可能会隐藏重要的异常信息，导致文件句柄泄漏或其他问题未被及时发现。

### 代码位置
```python
# src/data_factory/H5DataDict.py:38-41
try:
    self.h5f.close()
except:
    pass  # 忽略关闭时的异常
```

### 期望行为
应该捕获特定的异常类型（如OSError、ValueError），并至少记录日志以便调试。

### 实际行为
使用裸except捕获所有异常（包括SystemExit、KeyboardInterrupt），完全静默忽略。

### 影响范围
- 所有使用HDF5数据集的操作
- 可能导致文件句柄泄漏
- 难以调试文件关闭相关的问题

### 复现步骤
1. 使用包含无效或损坏的HDF5文件
2. 尝试关闭文件句柄
3. 异常被静默忽略

### 修复建议
```python
import logging

try:
    self.h5f.close()
except (OSError, ValueError) as e:
    logging.warning(f"Failed to close HDF5 file {self.h5_file}: {e}")
```

---
```

## 贡献指南

### 报告新Bug
1. 检查是否已有相关Bug
2. 使用正确的模板创建报告
3. 提供清晰的复现步骤
4. 标记正确的优先级和类型

### 修复Bug
1. 分配Bug给自己
2. 更新状态为fix_planned
3. 实施修复并添加测试
4. 更新状态为fixed
5. 关联相关的Pull Request

### 验证Bug修复
1. 根据复现步骤验证
2. 检查是否有回归
3. 更新Bug状态
4. 添加验证记录