---
name: Bug 报告（中文）
about: 报告可复现的 PHM-Vibench 缺陷
title: "[BUG] "
labels: bug
assignees: ''
---

## 提交前检查

- [ ] 我已搜索现有 [Issues](https://github.com/PHMbench/PHM-Vibench/issues)。
- [ ] 我已在当前 commit 或明确的 release tag 上复现问题。
- [ ] 我已阅读[中文贡献指南](../../CONTRIBUTING_CN.md)。
- [ ] 这不是安全漏洞；安全问题应遵循 [SECURITY.md](../../SECURITY.md)。

## 问题描述

说明缺陷和影响，并指出它涉及 config、data、model、task、trainer、CLI、
Streamlit、checkpoint 还是 artifact。

## 复现信息

1. 仓库 commit 或 tag：
2. 配置文件：
3. CLI overrides：
4. 数据来源或 fixture：
5. 完整命令：
6. 可稳定复现的步骤：

请使用维护入口，例如：

```bash
python main.py \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

请提供合法可共享的最小配置或数据 fixture，并移除凭据、私有数据和本机秘密。

## 预期行为

说明预期输出、状态变化、错误、指标或 artifact。

## 实际行为

请附退出码和完整 traceback/log：

```text
在此粘贴日志
```

## 环境

```text
操作系统：
CPU/GPU：
Python：
PyTorch：
CUDA runtime/driver：
PyTorch Lightning：
其他相关包：
```

可使用：

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

## 其他证据

仅在有助于复现或定位问题时附截图、输出目录、checkpoint 或相关 Issue。
