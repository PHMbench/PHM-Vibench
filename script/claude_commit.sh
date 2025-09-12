#!/bin/bash
# Claude Auto-Commit Shell Wrapper
# Simple interface for auto-commit operations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[Claude Auto-Commit]${NC} $1"
}

print_error() {
    echo -e "${RED}[Error]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[Warning]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[Info]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[Success]${NC} $1"
}

# Check if Python script exists
SCRIPT_PATH="$(dirname "$0")/claude_auto_commit.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "Claude auto-commit script not found at: $SCRIPT_PATH"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "🤖 Claude Auto-Commit Shell Wrapper"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  enable              Enable auto-commit"
    echo "  disable             Disable auto-commit"
    echo "  status              Show current status"
    echo "  commit [mode]       Run auto-commit with optional mode"
    echo "  dry-run             Show what would be committed"
    echo "  log                 Show recent auto-commit log"
    echo "  config              Show configuration"
    echo "  help                Show this help message"
    echo ""
    echo "Commit modes:"
    echo "  atomic              Commit each file separately"
    echo "  batch               Commit all files together"
    echo "  grouped             Group related files (default)"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 commit grouped"
    echo "  $0 dry-run"
    echo "  $0 enable"
}

# Function to show recent commits
show_log() {
    print_info "Recent Claude commits:"
    git log --oneline -10 --grep="🤖 \[Claude\]" --color=always 2>/dev/null || {
        print_warning "No Claude commits found or not in a git repository"
    }
}

# Function to show configuration
show_config() {
    if [ -f ".claude/auto_commit_config.yaml" ]; then
        print_info "Current configuration (.claude/auto_commit_config.yaml):"
        echo ""
        cat .claude/auto_commit_config.yaml
    else
        print_warning "Configuration file not found. Run with --config to create default."
    fi
}

# Main execution
case "${1:-help}" in
    enable)
        print_status "Enabling auto-commit..."
        python "$SCRIPT_PATH" --enable
        ;;
    disable)
        print_status "Disabling auto-commit..."
        python "$SCRIPT_PATH" --disable
        ;;
    status)
        python "$SCRIPT_PATH" --status
        ;;
    commit)
        MODE="${2:-grouped}"
        print_status "Running auto-commit in $MODE mode..."
        python "$SCRIPT_PATH" --mode "$MODE"
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            print_success "Auto-commit completed successfully"
        else
            print_error "Auto-commit failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
        ;;
    dry-run|dryrun)
        print_status "Dry run - showing what would be committed..."
        python "$SCRIPT_PATH" --dry-run
        ;;
    log)
        show_log
        ;;
    config)
        show_config
        ;;
    help|-h|--help)
        show_usage
        ;;
    test)
        print_status "Testing auto-commit system..."
        
        # Check if we're in a git repository
        if ! git rev-parse --git-dir &>/dev/null; then
            print_error "Not in a git repository"
            exit 1
        fi
        
        # Check Python script
        if python "$SCRIPT_PATH" --status &>/dev/null; then
            print_success "✓ Python script works"
        else
            print_error "✗ Python script has issues"
            exit 1
        fi
        
        # Check configuration
        if [ -f ".claude/auto_commit_config.yaml" ]; then
            print_success "✓ Configuration file exists"
        else
            print_warning "⚠ Configuration file missing (will be created automatically)"
        fi
        
        # Show status
        echo ""
        python "$SCRIPT_PATH" --status
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac