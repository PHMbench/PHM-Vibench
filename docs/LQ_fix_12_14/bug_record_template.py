#!/usr/bin/env python3
"""
Bug记录生成器
用于快速创建标准化的Bug记录
"""

import os
import json
from datetime import datetime

class BugRecorder:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.bugs_dir = os.path.join(base_dir, "docs/LQ_fix_12_14/bugs")
        self.next_bug_id = self._get_next_bug_id()

    def _get_next_bug_id(self):
        """获取下一个Bug ID"""
        index_file = os.path.join(self.base_dir, "docs/LQ_fix_12_14/BUG_INDEX.md")
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                content = f.read()
                # 查找已有的Bug数量
                bug_count = content.count("BUG-20251214-")
                return bug_count + 1
        return 1

    def create_bug(self, priority, bug_type, title, file_path, line_num,
                   description, repro_steps, expected, actual, fix_suggestion):
        """创建Bug记录"""
        bug_id = f"BUG-20251214-{self.next_bug_id:03d}"
        self.next_bug_id += 1

        # 确定模块
        module = self._determine_module(file_path)

        # 创建Bug内容
        bug_content = f"""## {bug_id}: {title}

- **优先级**: {priority}
- **类型**: {bug_type}
- **状态**: open
- **模块**: {module}
- **文件**: `{file_path}:{line_num}`
- **发现时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 问题描述
{description}

### 复现步骤
{repro_steps}

### 期望行为
{expected}

### 实际行为
{actual}

### 修复建议
{fix_suggestion}

### 相关代码
```python
# {file_path}:{line_num}
# [需要手动添加相关代码片段]
```

---
"""

        # 保存到模块文件
        module_file = os.path.join(self.bugs_dir, f"{module}.md")

        # 如果文件不存在，创建它
        if not os.path.exists(module_file):
            with open(module_file, 'w') as f:
                f.write(f"""# {module.replace('_', ' ').title()} 模块 Bug

## 概述
本文档记录 {module} 模块相关的 Bug。

---

""")

        # 追加Bug
        with open(module_file, 'a') as f:
            f.write(bug_content + "\n")

        # 更新索引
        self._update_index(bug_id, title, priority, bug_type, module)

        return bug_id

    def _determine_module(self, file_path):
        """确定模块"""
        if 'configs' in file_path:
            return 'configuration'
        elif 'data_factory' in file_path:
            return 'data_factory'
        elif 'model_factory' in file_path:
            return 'model_factory'
        elif 'task_factory' in file_path:
            return 'task_factory'
        elif 'trainer_factory' in file_path:
            return 'trainer_factory'
        elif file_path.startswith('Pipeline_'):
            return 'pipelines'
        else:
            return 'docs'

    def _update_index(self, bug_id, title, priority, bug_type, module):
        """更新Bug索引"""
        index_file = os.path.join(self.base_dir, "docs/LQ_fix_12_14/BUG_INDEX.md")

        # 读取现有内容
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                lines = f.readlines()
        else:
            lines = ["# Bug索引\n\n"]

        # 找到插入位置（在按日期部分）
        insert_pos = -1
        for i, line in enumerate(lines):
            if "## 按日期" in line:
                insert_pos = i + 2
                break

        # 如果没找到，在文件末尾添加
        if insert_pos == -1:
            lines.append("\n## 按日期\n\n")
            insert_pos = len(lines)

        # 插入新的Bug条目
        new_line = f"- {datetime.now().strftime('%Y-%m-%d')}：{bug_id} | {title} | {priority} | {bug_type} | {module}\n"
        lines.insert(insert_pos, new_line)

        # 写回文件
        with open(index_file, 'w') as f:
            f.writelines(lines)

def quick_create_bug(base_dir, scan_result_line):
    """从扫描结果快速创建Bug"""
    recorder = BugRecorder(base_dir)

    # 解析扫描结果
    if ':' not in scan_result_line:
        print("无法解析扫描结果格式")
        return

    parts = scan_result_line.split(':', 2)
    file_path = parts[0]
    line_num = parts[1]
    content = parts[2] if len(parts) > 2 else ""

    # 根据内容判断Bug类型和优先级
    if "except:" in scan_result_line:
        priority = "P0"
        bug_type = "exception_handling"
        title = f"裸except语句可能隐藏异常"
        description = f"代码中使用了裸except语句，可能会隐藏重要的异常信息，使调试变得困难。"
        fix_suggestion = "应该指定具体的异常类型，并记录异常信息。"

    elif "except...: pass" in scan_result_line:
        priority = "P1"
        bug_type = "exception_handling"
        title = f"异常被静默忽略"
        description = f"代码捕获异常但只是pass，可能导致错误状态被忽略。"
        fix_suggestion = "应该至少记录日志，或者处理异常状态。"

    elif "TODO" in scan_result_line or "FIXME" in scan_result_line:
        priority = "P2"
        bug_type = "code_todo"
        title = f"待实现的TODO/FIXME"
        description = f"代码中还有未完成的TODO/FIXME项。"
        fix_suggestion = "需要完成TODO中描述的功能或修复FIXME中的问题。"

    else:
        priority = "P2"
        bug_type = "other"
        title = f"需要关注的代码模式"
        description = f"扫描发现的可疑代码模式。"
        fix_suggestion = "需要进一步分析并确定是否需要修复。"

    # 创建Bug
    bug_id = recorder.create_bug(
        priority=priority,
        bug_type=bug_type,
        title=title,
        file_path=file_path,
        line_num=line_num,
        description=description,
        repro_steps=f"1. 查看文件 `{file_path}` 第 {line_num} 行\n2. 执行相关代码路径",
        expected="异常应该被正确处理或记录",
        actual=content,
        fix_suggestion=fix_suggestion
    )

    print(f"创建Bug记录: {bug_id}")
    return bug_id

# 示例使用
if __name__ == "__main__":
    base_dir = "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench"

    # 示例：从扫描结果创建Bug
    example_scan = "src/example.py:42: except:"
    quick_create_bug(base_dir, example_scan)