# 🤖 Claude Auto-Commit System - Complete User Guide

> **Author**: PHM-Vibench Team  
> **Date**: 2025-09-12  
> **Version**: 1.0  
> **Purpose**: Comprehensive guide for using Claude's intelligent auto-commit system

## 📋 Table of Contents

1. [Quick Start (2 minutes)](#quick-start-2-minutes)
2. [Installation & Setup](#installation--setup)
3. [Basic Usage - Step by Step](#basic-usage---step-by-step)
4. [Commit Modes](#commit-modes)
5. [Configuration Customization](#configuration-customization)
6. [Git Aliases Reference](#git-aliases-reference)
7. [Shell Wrapper Commands](#shell-wrapper-commands)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)
11. [Examples & Scenarios](#examples--scenarios)
12. [Safety & Security](#safety--security)

---

## Quick Start (2 minutes)

### ✅ Prerequisites Check
```bash
# Check if you're in a git repository
git status

# Expected: Should show git status, not "not a git repository"
```

### 🚀 Enable Auto-Commit
```bash
# Method 1: Shell wrapper (recommended)
./script/claude_commit.sh enable

# Method 2: Direct Python script
python script/claude_auto_commit.py --enable
```

### 🧪 Test the System
```bash
# Create a test file
echo "# Test file" > test_auto_commit.md

# Run auto-commit
./script/claude_commit.sh commit

# Check results
git log --oneline -3
```

**Expected Output:**
```
abc1234 🤖 [Claude] docs: Update test_auto_commit.md
```

### ✅ Success Indicators
- ✅ Commits have `🤖 [Claude]` prefix
- ✅ Meaningful commit messages
- ✅ Files grouped intelligently
- ✅ No errors in output

---

## Installation & Setup

### 📁 Required Files
The auto-commit system consists of these files:
```
script/
├── claude_auto_commit.py      # Main Python script (executable)
├── claude_commit.sh           # Shell wrapper (executable)
└── claude_auto_commit_user_guide.md  # This guide

.claude/
├── auto_commit_config.yaml    # Configuration file
└── settings.local.json        # Claude Code permissions
```

### 🔧 System Requirements
- **Git repository** (initialized)
- **Python 3.6+** with PyYAML
- **Bash shell** (for wrapper script)
- **Write permissions** in the repository

### ⚙️ Initial Configuration Check
```bash
# Check if configuration exists
ls -la .claude/auto_commit_config.yaml

# If missing, it will be created automatically on first run
python script/claude_auto_commit.py --status
```

### 🔒 Permissions Setup
The system requires these git operations to be allowed in `.claude/settings.local.json`:
```json
"allow": [
  "Bash(git add:*)",
  "Bash(git commit:*)",
  "Bash(git status:*)",
  "Bash(git log:*)",
  "Bash(python script/claude_auto_commit.py:*)",
  "Bash(./script/claude_commit.sh:*)"
]
```

---

## Basic Usage - Step by Step

### Step 1: Check System Status
```bash
# Check if auto-commit is enabled and working
./script/claude_commit.sh status
```

**Expected Output:**
```
🤖 Claude Auto-Commit Status:
   Enabled: ✅ Yes
   Mode: grouped
   Config: .claude/auto_commit_config.yaml
   Pending changes: 0 files
```

### Step 2: Enable Auto-Commit (if needed)
```bash
# Enable the system
./script/claude_commit.sh enable
```

**Expected Output:**
```
✅ Auto-commit enabled
```

### Step 3: Make Changes
Create, modify, or delete files as needed during your Claude session.

### Step 4: Preview Changes (Recommended)
```bash
# See what would be committed without actually committing
./script/claude_commit.sh dry-run
```

**Example Output:**
```
Would commit 5 files:
  A script/new_feature.py
  M src/model_factory/existing_model.py
  A docs/user_guide.md
  M .claude/settings.local.json
  A configs/new_config.yaml

Grouped into 3 commit(s):
  scripts: 1 files
  docs: 1 files  
  configs: 2 files
```

### Step 5: Run Auto-Commit
```bash
# Commit with intelligent grouping (recommended)
./script/claude_commit.sh commit grouped

# Or use other modes
./script/claude_commit.sh commit atomic   # One file per commit
./script/claude_commit.sh commit batch    # All files in one commit
```

**Expected Output:**
```
[Claude Auto-Commit] Running auto-commit in grouped mode...
🤖 [Claude] feat: Update new_feature.py
🤖 [Claude] docs: Update user_guide.md
🤖 [Claude] chore: Update 2 config files
[Success] Auto-commit completed successfully
```

### Step 6: Review Commits
```bash
# View recent Claude commits
./script/claude_commit.sh log

# Or use git alias
git claude-log
```

**Expected Output:**
```
abc1234 🤖 [Claude] chore: Update 2 config files  
def5678 🤖 [Claude] docs: Update user_guide.md
ghi9012 🤖 [Claude] feat: Update new_feature.py
```

### Step 7: Push When Ready (Manual)
```bash
# Review all changes before pushing
git claude-status

# Push to remote (manual step for safety)
git push origin your-branch-name
```

---

## Commit Modes

### 🎯 Atomic Mode
**One file per commit** - Maximum granularity

```bash
./script/claude_commit.sh commit atomic
```

**Use when:**
- Making unrelated changes to multiple files
- Want maximum rollback granularity
- Debugging or experimental changes

**Example:**
```
abc1234 🤖 [Claude] feat: Add new_model.py
def5678 🤖 [Claude] docs: Update README.md
ghi9012 🤖 [Claude] fix: Update bug_fix.py
```

### 📦 Batch Mode
**All files in one commit** - Maximum simplicity

```bash
./script/claude_commit.sh commit batch
```

**Use when:**
- All changes are tightly related
- Want to minimize commit count
- Working on a single feature

**Example:**
```
abc1234 🤖 [Claude] feat: Update 5 files for user authentication feature

Files changed:
- src/auth/login.py
- src/auth/register.py
- configs/auth_config.yaml
- docs/auth_guide.md
- tests/test_auth.py
```

### 🎨 Grouped Mode (Recommended)
**Smart grouping by file type** - Balance of granularity and organization

```bash
./script/claude_commit.sh commit grouped
```

**Use when:**
- Want intelligent organization (default)
- Making diverse changes across the project
- Need good commit history readability

**Example:**
```
abc1234 🤖 [Claude] feat: Update authentication scripts
def5678 🤖 [Claude] docs: Update documentation  
ghi9012 🤖 [Claude] chore: Update configuration files
```

---

## Configuration Customization

### 📝 Configuration File Location
```
.claude/auto_commit_config.yaml
```

### 🔧 Key Configuration Options

#### Enable/Disable System
```yaml
enabled: true                    # Set to false to disable
mode: grouped                    # atomic, batch, or grouped
max_files_per_commit: 10         # Safety limit
```

#### Commit Message Customization
```yaml
commit_prefix: "🤖 [Claude]"     # Prefix for all commits

commit_templates:
  feat: "feat: {description}"    # New features
  fix: "fix: {description}"      # Bug fixes  
  docs: "docs: {description}"    # Documentation
  chore: "chore: {description}"  # Configuration, etc.
```

#### File Grouping Rules
```yaml
group_rules:
  scripts: 'script/.*\.py$'      # Python scripts
  models: 'src/model_factory/.*' # Model files
  configs: '.*\.(yaml|yml|json)$' # Config files
  docs: '.*\.md$'                # Documentation
```

#### Exclusion Patterns
```yaml
exclusions:
  - '.*\.tmp$'                   # Temporary files
  - '__pycache__/.*'             # Python cache
  - '.*\.log$'                   # Log files
  - '.*\.pyc$'                   # Compiled Python
```

### 🛠️ Editing Configuration
```bash
# Edit configuration file
nano .claude/auto_commit_config.yaml

# Test new configuration
./script/claude_commit.sh test

# View current configuration
./script/claude_commit.sh config
```

---

## Git Aliases Reference

The system creates convenient git aliases:

### `git claude-commit`
**Run auto-commit directly**
```bash
git claude-commit                # Use default mode
git claude-commit --mode atomic  # Specify mode
git claude-commit --dry-run      # Preview only
```

### `git claude-status`  
**Quick status check**
```bash
git claude-status
```
Equivalent to: `git status --short`

### `git claude-log`
**View Claude commits only**
```bash
git claude-log
```
Shows last 10 commits with `🤖 [Claude]` prefix

### `git claude-undo`
**Undo last commit (soft reset)**
```bash
git claude-undo
```
Equivalent to: `git reset --soft HEAD~1`

### `git claude-review`
**Review staged changes**
```bash
git claude-review
```
Equivalent to: `git diff --cached`

---

## Shell Wrapper Commands

### 🔧 System Management

#### `./script/claude_commit.sh enable`
**Enable auto-commit system**
```bash
./script/claude_commit.sh enable
```

#### `./script/claude_commit.sh disable`
**Disable auto-commit system**  
```bash
./script/claude_commit.sh disable
```

#### `./script/claude_commit.sh status`
**Check system status**
```bash
./script/claude_commit.sh status
```

#### `./script/claude_commit.sh test`
**Test system functionality**
```bash
./script/claude_commit.sh test
```

### 📝 Commit Operations

#### `./script/claude_commit.sh commit [mode]`
**Run auto-commit**
```bash
./script/claude_commit.sh commit           # Default (grouped)
./script/claude_commit.sh commit atomic    # One file per commit
./script/claude_commit.sh commit batch     # All files together
./script/claude_commit.sh commit grouped   # Smart grouping
```

#### `./script/claude_commit.sh dry-run`
**Preview changes without committing**
```bash
./script/claude_commit.sh dry-run
```

### 📊 Information Commands

#### `./script/claude_commit.sh log`
**View recent Claude commits**
```bash
./script/claude_commit.sh log
```

#### `./script/claude_commit.sh config`
**View current configuration**
```bash
./script/claude_commit.sh config
```

#### `./script/claude_commit.sh help`
**Show usage information**
```bash
./script/claude_commit.sh help
```

---

## Advanced Features

### 🔍 Dry Run Mode
**Preview commits without executing**

Always run dry-run before important commits:
```bash
# Preview what would be committed
./script/claude_commit.sh dry-run

# Review the plan, then commit
./script/claude_commit.sh commit grouped
```

### 📊 Detailed Logging
**Monitor all auto-commit operations**

```bash
# View log file
cat .claude/auto_commit.log

# Tail log in real-time
tail -f .claude/auto_commit.log
```

### ⚙️ Custom Commit Messages
**Override automatic messages**

```bash
# Commit with custom message (future feature)
python script/claude_auto_commit.py --message "Custom commit message"
```

### 🚀 Integration with Claude Sessions
**Automatic triggering (when configured)**

Set up hooks in `.claude/settings.local.json`:
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write|Edit",
      "hooks": [{
        "type": "command",
        "command": "./script/claude_commit.sh commit grouped"
      }]
    }]
  }
}
```

---

## Troubleshooting

### ❌ Common Errors & Solutions

#### **Error: "Not in a git repository"**
```bash
# Solution: Initialize git repository
git init
git add .
git commit -m "Initial commit"
```

#### **Error: "Permission denied"**
```bash
# Solution: Make scripts executable
chmod +x script/claude_auto_commit.py
chmod +x script/claude_commit.sh
```

#### **Error: "ModuleNotFoundError: No module named 'yaml'"**
```bash
# Solution: Install PyYAML
pip install PyYAML
```

#### **Error: "Nothing to commit"**
```bash
# Check for changes
git status

# If no changes, this is expected behavior
./script/claude_commit.sh status
```

#### **Error: "Commit failed with return code 1"**
```bash
# Check git configuration
git config user.name
git config user.email

# Set if missing
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 🔧 Reset and Recovery

#### **Undo Last Auto-Commit**
```bash
# Soft reset (keeps changes)
git claude-undo

# Or manually
git reset --soft HEAD~1
```

#### **Undo Multiple Commits**
```bash
# Reset last 3 commits (keeps changes)
git reset --soft HEAD~3

# Hard reset (DANGER: loses changes)
git reset --hard HEAD~3
```

#### **Reset Configuration**
```bash
# Backup current config
cp .claude/auto_commit_config.yaml .claude/auto_commit_config.yaml.bak

# Delete config to regenerate defaults
rm .claude/auto_commit_config.yaml

# Run to recreate defaults
python script/claude_auto_commit.py --status
```

### 🕐 Performance Issues

#### **Slow Commits**
```bash
# Check repository size
du -sh .git

# Clean up if needed
git gc --aggressive
```

#### **Too Many Small Commits**
```bash
# Switch to batch mode temporarily
python script/claude_auto_commit.py --mode batch

# Or squash commits later
git rebase -i HEAD~10
```

---

## Best Practices

### ✅ When to Use Auto-Commit

**✅ Good Use Cases:**
- During active development sessions
- Making multiple related changes
- Documenting progress incrementally  
- Experimenting with configurations
- Refactoring sessions

**❌ When to Commit Manually:**
- Final production commits
- Important milestone commits
- Commits requiring detailed messages
- Breaking changes that need explanation
- Commits that will be pushed immediately

### 🔄 Workflow Integration

#### **Recommended Workflow**
1. **Start session**: Enable auto-commit
2. **Work actively**: Let auto-commit handle incremental saves
3. **Review progress**: Use `git claude-log` periodically
4. **Consolidate**: Squash commits before pushing (optional)
5. **Push**: Manual push to remote after review

#### **Branch Strategy**
```bash
# Use feature branches for auto-commit
git checkout -b feature/new-functionality

# Work with auto-commit enabled
./script/claude_commit.sh enable

# Make changes, auto-commit handles the rest
# ...

# Review before merging
git claude-log
git diff main

# Merge when ready
git checkout main
git merge feature/new-functionality
```

### 📏 Commit Message Quality

#### **Auto-Generated Messages Are:**
- ✅ Consistent in format
- ✅ Properly categorized (feat/fix/docs)
- ✅ Include file context
- ✅ Clearly marked as Claude-generated

#### **Consider Manual Messages For:**
- Major architectural changes
- Breaking API changes
- Important bug fixes
- Release commits

### 🔒 Security Considerations

#### **Safe Auto-Commit Practices**
```yaml
# Exclude sensitive files in config
exclusions:
  - '.*\.key$'
  - '.*\.env$' 
  - 'secrets/.*'
  - '.*_secret\..*'
```

#### **Review Before Pushing**
```bash
# Always review commits before pushing
git claude-log
git diff origin/main

# Check for sensitive information
git log --patch --grep="🤖 \[Claude\]"
```

---

## Examples & Scenarios

### 📚 Example 1: Adding New Features

**Scenario:** Adding a new data processing module

```bash
# Start with clean state
git claude-status

# Enable auto-commit
./script/claude_commit.sh enable

# Create new files during development
# - src/data_processing/new_module.py
# - tests/test_new_module.py
# - docs/data_processing_guide.md
# - configs/processing_config.yaml

# Preview changes
./script/claude_commit.sh dry-run
```

**Expected Output:**
```
Would commit 4 files:
  A src/data_processing/new_module.py
  A tests/test_new_module.py  
  A docs/data_processing_guide.md
  A configs/processing_config.yaml

Grouped into 4 commit(s):
  scripts: 1 files
  tests: 1 files
  docs: 1 files
  configs: 1 files
```

```bash
# Commit the changes
./script/claude_commit.sh commit grouped

# Review results
git claude-log
```

**Result:**
```
abc1234 🤖 [Claude] chore: Update processing_config.yaml
def5678 🤖 [Claude] docs: Update data_processing_guide.md
ghi9012 🤖 [Claude] test: Update test_new_module.py
jkl3456 🤖 [Claude] feat: Update new_module.py
```

### 🔧 Example 2: Bulk Documentation Updates

**Scenario:** Updating multiple documentation files

```bash
# Modified files:
# - README.md
# - docs/installation.md
# - docs/usage.md
# - docs/api_reference.md
# - CHANGELOG.md

# Use batch mode for related changes
./script/claude_commit.sh commit batch
```

**Result:**
```
abc1234 🤖 [Claude] docs: Update 5 documentation files

Files changed:
- CHANGELOG.md
- README.md
- docs/api_reference.md
- docs/installation.md
- docs/usage.md

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 🛠️ Example 3: Refactoring Session

**Scenario:** Refactoring model architecture

```bash
# Many files modified during refactoring
./script/claude_commit.sh dry-run
```

**Expected Output:**
```
Would commit 12 files:
  M src/model_factory/base_model.py
  M src/model_factory/transformer_model.py
  M src/model_factory/cnn_model.py
  M tests/test_base_model.py
  M tests/test_transformer.py
  M configs/model_config.yaml
  M docs/model_architecture.md
  A src/model_factory/utils.py
  D src/model_factory/deprecated_model.py
  ...

Grouped into 6 commit(s):
  models: 4 files
  tests: 2 files
  configs: 1 files
  docs: 1 files
  utils: 1 files
  deprecated: 1 files
```

```bash
# Commit in grouped mode for organization
./script/claude_commit.sh commit grouped
```

### ⚙️ Example 4: Configuration Changes

**Scenario:** Updating various configuration files

```bash
# Files changed:
# - .claude/settings.local.json
# - configs/training_config.yaml
# - configs/model_config.yaml
# - docker-compose.yml

# Use atomic mode to separate different config types
./script/claude_commit.sh commit atomic
```

**Result:**
```
abc1234 🤖 [Claude] chore: Update docker-compose.yml
def5678 🤖 [Claude] chore: Update model_config.yaml
ghi9012 🤖 [Claude] chore: Update training_config.yaml
jkl3456 🤖 [Claude] chore: Update settings.local.json
```

---

## Safety & Security

### 🔒 What Gets Committed Automatically

#### **✅ Included by Default:**
- Source code files (`.py`, `.js`, `.cpp`, etc.)
- Documentation (`.md`, `.rst`, `.txt`)
- Configuration files (`.yaml`, `.json`, `.toml`)
- Test files (`test_*.py`, `*_test.js`)
- Data schemas and metadata

#### **❌ Excluded by Default:**
- Temporary files (`.tmp`, `.cache`, `.swp`)
- Build artifacts (`__pycache__/`, `node_modules/`)
- Log files (`.log`)
- System files (`.DS_Store`, `Thumbs.db`)
- Large data files (`.h5`, `.pkl`, `.npz`)
- Compiled files (`.pyc`, `.class`)

### 🛡️ Security Features

#### **Automatic Exclusions**
```yaml
# These patterns are excluded automatically
exclusions:
  - '.*\.key$'          # Key files
  - '.*\.env$'          # Environment files
  - 'secrets/.*'        # Secrets directory
  - '.*_secret\..*'     # Secret files
  - '.*\.log$'          # Log files
```

#### **Safety Limits**
```yaml
# Prevents accidental large commits
max_files_per_commit: 10
max_commits_per_session: 50
min_time_between_commits: 30  # seconds
```

#### **Manual Overrides**
```bash
# Always preview large changes
./script/claude_commit.sh dry-run

# Use atomic mode for risky changes
./script/claude_commit.sh commit atomic
```

### 🚫 Push Protection

#### **Auto-Commit Never Pushes**
- Commits are **local only**
- **Manual push required** for safety
- Remote operations **explicitly denied** in permissions

#### **Review Before Pushing**
```bash
# Review all Claude commits
git claude-log

# Check diff against remote
git diff origin/main

# Review specific commits
git show abc1234

# Push only when ready
git push origin feature-branch
```

### 🔄 Recovery Options

#### **Undo Recent Commits**
```bash
# Undo last commit (keeps changes)
git claude-undo

# Undo last 3 commits
git reset --soft HEAD~3
```

#### **Emergency Reset**
```bash
# Create backup branch first
git branch backup-$(date +%Y%m%d-%H%M%S)

# Hard reset (DANGER: loses uncommitted changes)
git reset --hard HEAD~5

# Or reset to specific commit
git reset --hard abc1234
```

---

## 🎓 Conclusion

The Claude Auto-Commit System provides:

### ✅ **Benefits**
- **🤖 Intelligent automation** of commit processes
- **📦 Smart grouping** of related files
- **🔒 Safety features** and exclusion patterns  
- **🎯 Multiple modes** for different workflows
- **📝 Consistent messaging** with proper categorization
- **🛡️ Security safeguards** against accidental pushes

### 🎯 **Best Results When**
- Used during active development sessions
- Combined with periodic manual review
- Configured for your specific project needs
- Used with feature branches
- Followed by manual push after review

### 📞 **Support & Issues**
- Check logs: `.claude/auto_commit.log`
- Test system: `./script/claude_commit.sh test`
- Reset config: Delete `.claude/auto_commit_config.yaml`
- Emergency recovery: `git claude-undo`

---

**🚀 Ready to boost your development workflow with intelligent auto-commits!**

*This guide covers all features of the Claude Auto-Commit System v1.0. For updates and advanced usage, check the latest version of this guide.*