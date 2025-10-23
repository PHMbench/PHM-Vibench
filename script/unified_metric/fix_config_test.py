#!/usr/bin/env python3
"""
修复test_config_fixed.py的语法错误
"""

import os

def fix_syntax_error():
    file_path = 'script/unified_metric/existing_test_files/test_config_fixed.py'

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 修复第80-82行的语法错误
    # 重新写这些行，确保没有隐藏字符
    lines[79] = '        print(f"  - 修改时间: {info.get(\'modified_at\', \'未知\')}")\n'
    lines[80] = '        print(f"  - 原始路径: {info.get(\'original_data_dir\', \'未知\')}")\n'
    lines[81] = '        print(f"  - 新路径: {info.get(\'new_data_dir\', \'未知\')}")\n'

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"✅ 已修复 {file_path} 的语法错误")

    # 验证修复
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        compile(content, file_path, 'exec')
        print("✅ 语法检查通过")
        return True
    except SyntaxError as e:
        print(f"❌ 仍有语法错误: {e}")
        return False

if __name__ == "__main__":
    fix_syntax_error()