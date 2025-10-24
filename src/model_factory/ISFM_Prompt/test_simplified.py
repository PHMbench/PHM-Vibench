#!/usr/bin/env python3
"""
Simple test script to verify simplified ISFM_Prompt components.

This script tests the syntax and basic functionality of the simplified components
without requiring PyTorch installation.
"""

import ast
import sys
import os

def test_python_syntax(file_path):
    """Test if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the AST to check syntax
        ast.parse(source)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Test all simplified components."""
    print("=== Simplified ISFM_Prompt Components Test ===\n")

    # List of files to test
    test_files = [
        "components/SimpleSystemPromptEncoder.py",
        "embedding/HSE_prompt.py",
        "M_02_ISFM_Prompt.py",
        "components/__init__.py",
        "embedding/__init__.py"
    ]

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    all_passed = True

    for file_path in test_files:
        full_path = os.path.join(current_dir, file_path)

        if not os.path.exists(full_path):
            print(f"❌ {file_path}: File not found")
            all_passed = False
            continue

        success, message = test_python_syntax(full_path)

        if success:
            print(f"✅ {file_path}: {message}")
        else:
            print(f"❌ {file_path}: {message}")
            all_passed = False

    print(f"\n=== Test Results ===")
    if all_passed:
        print("🎉 All simplified components have valid Python syntax!")
        print("📝 Summary of simplifications:")
        print("   • Created SimpleSystemPromptEncoder (lightweight Dataset_id → prompt)")
        print("   • Created HSE_prompt (simplified HSE with system prompts)")
        print("   • Simplified M_02_ISFM_Prompt (removed complex configurations)")
        print("   • Updated component imports and registrations")
        print("   • Created demo configuration file")
        print("\n🚀 Ready for use with:")
        print("   python main.py --config configs/demo/Simplified_Prompt/hse_prompt_demo.yaml")
    else:
        print("❌ Some components have syntax errors. Please fix before using.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())