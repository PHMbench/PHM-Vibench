#!/usr/bin/env python3
"""
Claude Auto-Commit System

Automatically commits changes made during Claude sessions with intelligent
grouping and meaningful commit messages.

Features:
- Smart file grouping based on project structure
- Intelligent commit message generation
- Multiple commit modes (atomic, batch, grouped)
- Configurable exclusions and patterns
- Comprehensive logging

Usage:
    python script/claude_auto_commit.py [options]

Author: PHM-Vibench Team (via Claude Code)
Date: 2025-09-12
"""

import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import argparse
import logging
import sys
import re
import os

class ClaudeAutoCommit:
    """Manages automatic git commits for Claude sessions."""
    
    def __init__(self, config_path: str = ".claude/auto_commit_config.yaml"):
        """Initialize the auto-commit system."""
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_logging()
        
    def load_config(self):
        """Load configuration or use defaults."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.get_default_config()
            self.save_default_config()
            
    def get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'enabled': True,
            'mode': 'grouped',  # atomic, batch, or grouped
            'max_files_per_commit': 5,
            'commit_prefix': '🤖 [Claude]',
            'group_rules': {
                'specs': r'\.claude/specs/.*\.md$',
                'scripts': r'script/.*\.py$',
                'configs': r'.*\.(yaml|yml|json)$',
                'docs': r'.*\.md$',
                'tests': r'test.*\.py$',
                'models': r'src/model_factory/.*',
                'tasks': r'src/task_factory/.*',
                'utils': r'src/utils/.*',
                'data': r'data/.*'
            },
            'commit_templates': {
                'feat': 'feat: {description}',
                'fix': 'fix: {description}',
                'docs': 'docs: {description}',
                'refactor': 'refactor: {description}',
                'test': 'test: {description}',
                'chore': 'chore: {description}',
                'style': 'style: {description}',
                'perf': 'perf: {description}'
            },
            'exclusions': [
                r'.*\.tmp$',
                r'.*\.cache$',
                r'__pycache__/.*',
                r'\.git/.*',
                r'.*\.pyc$',
                r'\.DS_Store$',
                r'.*\.swp$',
                r'.*\.log$'
            ],
            'advanced': {
                'add_claude_signature': True,
                'include_file_list': True,
                'max_commit_message_length': 500,
                'auto_stage': True
            }
        }
        
    def save_default_config(self):
        """Save default configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"✅ Created default configuration: {self.config_path}")
        
    def setup_logging(self):
        """Setup logging for auto-commit operations."""
        log_file = Path('.claude/auto_commit.log')
        log_file.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [Claude Auto-Commit] - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, 'a', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def is_git_repository(self) -> bool:
        """Check if current directory is a git repository."""
        try:
            subprocess.run(['git', 'rev-parse', '--git-dir'], 
                         check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
            
    def get_git_status(self) -> Tuple[List[str], List[str], List[str]]:
        """Get current git status."""
        if not self.is_git_repository():
            self.logger.error("Not in a git repository")
            return [], [], []
            
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git status failed: {e}")
            return [], [], []
        
        modified = []
        added = []
        deleted = []
        
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
                
            status = line[:2]
            file_path = line[3:]
            
            # Skip excluded files
            if self.is_excluded(file_path):
                continue
                
            if status == '??':
                added.append(file_path)
            elif 'M' in status:
                modified.append(file_path)
            elif 'D' in status:
                deleted.append(file_path)
                
        return modified, added, deleted
        
    def is_excluded(self, file_path: str) -> bool:
        """Check if file should be excluded from auto-commit."""
        for pattern in self.config.get('exclusions', []):
            if re.match(pattern, file_path):
                return True
        return False
        
    def group_files(self, files: List[str]) -> Dict[str, List[str]]:
        """Group files by type based on patterns."""
        groups = {}
        
        for file in files:
            file_path = Path(file)
            group_found = False
            
            # Check against group rules
            for group_name, pattern in self.config['group_rules'].items():
                if re.match(pattern, str(file_path)):
                    if group_name not in groups:
                        groups[group_name] = []
                    groups[group_name].append(file)
                    group_found = True
                    break
                    
            if not group_found:
                # Default group based on extension or directory
                if file_path.suffix:
                    group_name = file_path.suffix[1:]  # Remove dot
                else:
                    # Use parent directory name
                    parent_parts = file_path.parent.parts
                    group_name = parent_parts[-1] if parent_parts else 'other'
                    
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(file)
                
        return groups
        
    def detect_commit_type(self, files: List[str]) -> str:
        """Detect the type of commit based on files and content."""
        file_paths = [f.lower() for f in files]
        
        # Check for specific patterns
        if any('test' in f for f in file_paths):
            return 'test'
        elif any(f.endswith('.md') for f in file_paths):
            return 'docs'
        elif any('fix' in f for f in file_paths):
            return 'fix'
        elif any('refactor' in f for f in file_paths):
            return 'refactor'
        elif any('config' in f or f.endswith(('.yaml', '.yml', '.json')) for f in file_paths):
            return 'chore'
        else:
            # Default to feat for new functionality
            return 'feat'
            
    def generate_commit_message(self, files: List[str], action: str = 'Update') -> str:
        """Generate intelligent commit message."""
        if not files:
            return f"{self.config['commit_prefix']} Empty commit"
            
        commit_type = self.detect_commit_type(files)
        
        # Generate description based on files
        if len(files) == 1:
            file_path = Path(files[0])
            if action == 'Add':
                description = f"Add {file_path.name}"
            elif action == 'Remove':
                description = f"Remove {file_path.name}"
            else:
                description = f"Update {file_path.name}"
        else:
            # Find common directory or group files by type
            paths = [Path(f) for f in files]
            
            # Try to find common parent directory
            try:
                common_parts = []
                all_parts = [p.parts for p in paths]
                
                if all_parts:
                    min_length = min(len(parts) for parts in all_parts)
                    for i in range(min_length):
                        parts_at_i = [parts[i] for parts in all_parts]
                        if len(set(parts_at_i)) == 1:
                            common_parts.append(parts_at_i[0])
                        else:
                            break
                            
                if common_parts and len(common_parts) > 0:
                    common_path = '/'.join(common_parts)
                    if len(common_path) > 30:  # Truncate long paths
                        common_path = '...' + common_path[-27:]
                    description = f"{action} {len(files)} files in {common_path}/"
                else:
                    # Group by file type
                    extensions = [p.suffix for p in paths if p.suffix]
                    if extensions and len(set(extensions)) == 1:
                        ext = extensions[0][1:]  # Remove dot
                        description = f"{action} {len(files)} {ext} files"
                    else:
                        description = f"{action} {len(files)} files"
                        
            except Exception:
                description = f"{action} {len(files)} files"
                
        # Use template
        template = self.config['commit_templates'].get(commit_type, '{description}')
        message = template.format(description=description)
        
        # Add prefix if configured
        if self.config.get('commit_prefix'):
            message = f"{self.config['commit_prefix']} {message}"
            
        # Add file list if configured and not too many files
        if (self.config['advanced'].get('include_file_list', True) and 
            len(files) > 1 and len(files) <= 10):
            message += "\n\nFiles changed:"
            for f in sorted(files):
                message += f"\n- {f}"
                
        # Add Claude signature if configured
        if self.config['advanced'].get('add_claude_signature', True):
            message += "\n\n🤖 Generated with Claude Code\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
            
        # Truncate if too long
        max_length = self.config['advanced'].get('max_commit_message_length', 500)
        if len(message) > max_length:
            message = message[:max_length-3] + '...'
            
        return message
        
    def stage_files(self, files: List[str]) -> bool:
        """Stage files for commit."""
        if not files:
            return True
            
        try:
            for file in files:
                subprocess.run(['git', 'add', file], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stage files: {e}")
            return False
            
    def commit_files(self, files: List[str], message: Optional[str] = None) -> bool:
        """Commit a list of files."""
        if not files:
            self.logger.info("No files to commit")
            return True
            
        try:
            # Stage files if auto_stage is enabled
            if self.config['advanced'].get('auto_stage', True):
                if not self.stage_files(files):
                    return False
                    
            # Generate message if not provided
            if not message:
                action = self.detect_file_action(files)
                message = self.generate_commit_message(files, action)
                
            # Commit with message
            subprocess.run(['git', 'commit', '-m', message], check=True)
            
            self.logger.info(f"✅ Committed {len(files)} files")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Commit failed: {e}")
            return False
            
    def detect_file_action(self, files: List[str]) -> str:
        """Detect the primary action being performed on files."""
        # This is a simplified version - could be enhanced
        # to actually check git diff status
        modified, added, deleted = self.get_git_status()
        
        if any(f in added for f in files):
            return 'Add'
        elif any(f in deleted for f in files):
            return 'Remove'
        else:
            return 'Update'
            
    def auto_commit(self, mode: Optional[str] = None) -> bool:
        """Perform auto-commit based on mode."""
        if not self.config.get('enabled', True):
            self.logger.info("Auto-commit is disabled")
            return False
            
        if not self.is_git_repository():
            self.logger.error("Not in a git repository")
            return False
            
        mode = mode or self.config.get('mode', 'grouped')
        
        # Get current status
        modified, added, deleted = self.get_git_status()
        all_files = modified + added + deleted
        
        if not all_files:
            self.logger.info("No changes to commit")
            return True
            
        self.logger.info(f"Found {len(all_files)} changed files")
        success = True
        
        if mode == 'atomic':
            # Commit each file separately
            for file in all_files:
                action = 'Add' if file in added else 'Remove' if file in deleted else 'Update'
                message = self.generate_commit_message([file], action)
                if not self.commit_files([file], message):
                    success = False
                    
        elif mode == 'batch':
            # Commit all files together
            action = 'Update' if modified else 'Add' if added else 'Remove'
            message = self.generate_commit_message(all_files, action)
            if not self.commit_files(all_files, message):
                success = False
                
        elif mode == 'grouped':
            # Group files and commit each group
            groups = self.group_files(all_files)
            
            for group_name, group_files in groups.items():
                self.logger.info(f"Committing {group_name} group ({len(group_files)} files)")
                
                # Determine action for this group
                group_added = [f for f in group_files if f in added]
                group_deleted = [f for f in group_files if f in deleted]
                group_modified = [f for f in group_files if f in modified]
                
                if group_added and not group_modified and not group_deleted:
                    action = 'Add'
                elif group_deleted and not group_modified and not group_added:
                    action = 'Remove'
                else:
                    action = 'Update'
                    
                description = f"{action} {group_name} files" if len(group_files) > 1 else f"{action} {Path(group_files[0]).name}"
                message = self.generate_commit_message(group_files, description)
                
                if not self.commit_files(group_files, message):
                    success = False
                    
        else:
            self.logger.error(f"Unknown mode: {mode}")
            return False
            
        if success:
            self.logger.info(f"✅ Auto-commit completed successfully in {mode} mode")
        else:
            self.logger.error(f"❌ Some commits failed in {mode} mode")
            
        return success
        
    def enable(self):
        """Enable auto-commit."""
        self.config['enabled'] = True
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print("✅ Auto-commit enabled")
        
    def disable(self):
        """Disable auto-commit."""
        self.config['enabled'] = False
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print("❌ Auto-commit disabled")
        
    def status(self):
        """Show current status."""
        enabled = self.config.get('enabled', True)
        mode = self.config.get('mode', 'grouped')
        
        print(f"🤖 Claude Auto-Commit Status:")
        print(f"   Enabled: {'✅ Yes' if enabled else '❌ No'}")
        print(f"   Mode: {mode}")
        print(f"   Config: {self.config_path}")
        
        if self.is_git_repository():
            modified, added, deleted = self.get_git_status()
            total = len(modified) + len(added) + len(deleted)
            print(f"   Pending changes: {total} files")
            if total > 0:
                print(f"     Modified: {len(modified)}")
                print(f"     Added: {len(added)}")
                print(f"     Deleted: {len(deleted)}")
        else:
            print("   ❌ Not in a git repository")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Claude Auto-Commit System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script/claude_auto_commit.py                    # Run auto-commit with default mode
  python script/claude_auto_commit.py --mode grouped     # Run with grouped mode
  python script/claude_auto_commit.py --enable           # Enable auto-commit
  python script/claude_auto_commit.py --disable          # Disable auto-commit
  python script/claude_auto_commit.py --status           # Show status
        """
    )
    
    parser.add_argument('--mode', choices=['atomic', 'batch', 'grouped'],
                       help='Commit mode (default: from config)')
    parser.add_argument('--enable', action='store_true',
                       help='Enable auto-commit')
    parser.add_argument('--disable', action='store_true',
                       help='Disable auto-commit')
    parser.add_argument('--status', action='store_true',
                       help='Show status')
    parser.add_argument('--config', default='.claude/auto_commit_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be committed without actually committing')
    
    args = parser.parse_args()
    
    try:
        # Create auto-commit instance
        auto_commit = ClaudeAutoCommit(config_path=args.config)
        
        # Handle different actions
        if args.enable:
            auto_commit.enable()
        elif args.disable:
            auto_commit.disable()
        elif args.status:
            auto_commit.status()
        elif args.dry_run:
            # Show what would be committed
            modified, added, deleted = auto_commit.get_git_status()
            all_files = modified + added + deleted
            
            if not all_files:
                print("No changes to commit")
            else:
                print(f"Would commit {len(all_files)} files:")
                for f in sorted(all_files):
                    status = 'A' if f in added else 'D' if f in deleted else 'M'
                    print(f"  {status} {f}")
                    
                # Show grouping
                groups = auto_commit.group_files(all_files)
                print(f"\nGrouped into {len(groups)} commit(s):")
                for group_name, group_files in groups.items():
                    print(f"  {group_name}: {len(group_files)} files")
        else:
            # Run auto-commit
            success = auto_commit.auto_commit(mode=args.mode)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()