# AI Tools

本项目使用 [tmux-ai-cli](https://github.com/liq22/tmux-ai-cli) 管理多个 AI 工具（Claude、Gemini、Codex）。

---

## 安装

```bash
cd ../tmux-ai-cli
./install.sh
```

---

## 快速使用

```bash
ai list           # 列出所有实例
ai new claude     # 创建 claude 实例
ai c1             # 快捷进入 claude-1
ai master         # 统一视图（单终端切换）
```

---

## VS Code 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Alt+1` | claude-1 |
| `Ctrl+Alt+2` | claude-2 |
| `Ctrl+Alt+3` | gemini-1 |
| `Ctrl+Alt+4` | gemini-2 |
| `Ctrl+Alt+5` | codex-1 |
| `Ctrl+Alt+Shift+N` | 新建 claude |
| `Ctrl+Alt+Shift+M` | 新建 gemini |

---

## tmux 快捷键

| 按键 | 功能 |
|------|------|
| `Ctrl+B` 然后 `d` | 断开会话（AI 继续运行） |
| `Ctrl+B` 然后 `1/2/3` | 切换窗口（master 模式） |

---

详细用法见 **[tmux-ai-cli 文档](https://github.com/liq22/tmux-ai-cli)**
