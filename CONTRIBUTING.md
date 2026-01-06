# Contributing to PHM-Vibench

<div align="center">
  <p>
    <a href="CONTRIBUTING.md"><strong>English</strong></a> |
    <a href="CONTRIBUTING_CN.md">中文</a>
  </p>
</div>

We welcome contributions to PHM-Vibench! This guide will help you understand how to contribute effectively to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contribution Guidelines](#contribution-guidelines)
4. [Adding Components](#adding-components)
   - [New Datasets](#new-datasets)
   - [New Models](#new-models)
   - [New Tasks](#new-tasks)
   - [New Pipelines](#new-pipelines)
5. [Configuration System](#configuration-system)
6. [Code Standards](#code-standards)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Pull Request Process](#pull-request-process)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git
- Basic understanding of deep learning and time-series analysis

### Architecture Overview

PHM-Vibench uses a **factory design pattern** for modular extension:

```
PHM-Vibench/
├── src/
│   ├── data_factory/      # Dataset loading and preprocessing
│   ├── model_factory/     # Models (embeddings, backbones, heads)
│   ├── task_factory/      # Training logic and metrics
│   └── trainer_factory/   # PyTorch Lightning Trainer wiring
├── configs/               # YAML configuration files
└── docs/                  # Documentation
```

For detailed architecture, see [`CLAUDE.md`](CLAUDE.md).

## Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/your-username/PHM-Vibench.git
cd PHM-Vibench
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Tests**
```bash
python -m pytest test/
```

5. **Offline Smoke Test** (no downloads required)
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

## Contribution Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **New Components**: Datasets, models, tasks, pipelines
2. **Bug Fixes**: Fixing issues in existing code
3. **Documentation**: Improving docs, examples, and tutorials
4. **Performance Improvements**: Optimizations and efficiency gains
5. **Configuration**: New experiment configs and presets

### Before You Start

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create an Issue**: For new features or significant changes
3. **Discuss First**: For major changes, discuss with maintainers
4. **Read CLAUDE.md**: Understand project architecture and change strategy

### Key Design Principles

- **Configuration-First**: All experiments defined via YAML configs
- **Factory Pattern**: Register components, don't hardcode imports
- **Single Source of Truth**: Update registry → atlas → docs
- **本科生能跑 + 博士生能改**: Keep it accessible yet extensible

## Adding Components

### New Datasets

See [`src/data_factory/contributing.md`](src/data_factory/contributing.md) for detailed guide.

**Quick steps**:
1. Create dataset class in `src/data_factory/dataset_task/`
2. Register in `src/data_factory/dataset_task/__init__.py`
3. Add metadata entry to `data/metadata.xlsx`
4. Create config in `configs/base/data/`

### New Models

See [`src/model_factory/contributing.md`](src/model_factory/contributing.md) for detailed guide.

**Model components follow registry-style IDs**:
- Embeddings: `E_**_*`
- Backbones: `B_**_*`
- Heads: `H_**_*`

**Quick steps**:
1. Create model class with NumPy-style docstrings
2. Register in appropriate `__init__.py`
3. Add config preset in `configs/demo/`

### New Tasks

**Task types** (selected by `task.type` + `task.name`):
- `DG`: Domain Generalization
- `CDDG`: Cross-Dataset Domain Generalization
- `FS`/`GFS`: Few-shot / Generalized Few-shot
- `ID`: ID-based ingestion
- `MT`: Multi-task

**Quick steps**:
1. Create task in `src/task_factory/task/<TYPE>/`
2. Inherit from base task class
3. Register in `src/task_factory/task/<TYPE>/__init__.py`

### New Pipelines

Pipelines assemble factories in fixed order:
1. Load config
2. Build data
3. Build model
4. Build task
5. Build trainer

**Quick steps**:
1. Create `src/Pipeline_<name>.py`
2. Select via YAML: `pipeline: <name>`

## Configuration System

PHM-Vibench uses **v5.x 5-block config model**:
- `environment` / `data` / `model` / `task` / `trainer`

### Adding New Configs

1. **Create config YAML** in `configs/demo/` or `configs/experiments/`
2. **Add to registry**: Update `configs/config_registry.csv`
3. **Regenerate atlas**: `python -m scripts.gen_config_atlas`
4. **Validate**: `python -m scripts.validate_configs`

### Config Composition Rules (low → high precedence)

1. `base_configs.*` YAML files
2. Demo YAML's own block overrides
3. Optional local override `configs/local/local.yaml`
4. CLI `--override key=value`

### Inspection Tools

```bash
# View resolved config + sources + targets
python -m scripts.config_inspect --config <yaml> --override key=value

# Validate all configs
python -m scripts.validate_configs

# Generate CONFIG_ATLAS.md
python -m scripts.gen_config_atlas
```

## Code Standards

### Python Style Guide

We follow PEP 8 with modifications:

1. **Line Length**: 100 characters maximum
2. **Imports**: Group imports (standard, third-party, local)
3. **Naming**:
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_CASE`

### Docstring Standards

Use NumPy-style docstrings:

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """Brief description.

    Longer description explaining purpose and behavior.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str, optional
        Description of param2 (default: "default")

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When param1 is negative

    Examples
    --------
    >>> result = function_name(5, "test")
    >>> print(result)
    True
    """
```

## Testing

### Test Structure

```
test/
├── test_end_to_end_integration.py
├── test_parameter_consistency.py
└── ...
```

### Running Tests

```bash
# Run all maintained tests
python -m pytest test/

# Run specific test file
python -m pytest test/test_parameter_consistency.py

# Run with coverage
python -m pytest test/ --cov=src --cov-report=html
```

### Writing Tests

```python
import pytest
import torch
from argparse import Namespace

class TestYourComponent:
    """Test suite for YourComponent."""

    @pytest.fixture
    def config(self):
        """Configuration for testing."""
        return Namespace(
            param1="value1",
            param2=42
        )

    def test_basic_functionality(self, config):
        """Test basic functionality."""
        # Arrange
        component = YourComponent(config)

        # Act
        result = component.method()

        # Assert
        assert result is not None
```

## Documentation

### Documentation Requirements

1. **API Documentation**: NumPy-style docstrings
2. **Usage Examples**: Working code examples
3. **Config Documentation**: Update registry and atlas
4. **Bilingual Support**: English and Chinese (`_CN.md` suffix)

### Documentation Structure

```
PHM-Vibench/
├── README.md / README_CN.md           # Main project README
├── CONTRIBUTING.md / CONTRIBUTING_CN.md  # This file
├── CLAUDE.md                           # Architecture and change strategy
├── AGENTS.md                           # Development runbook
├── configs/README.md                   # Config system guide
├── docs/
│   ├── CONFIG_ATLAS.md                 # Generated config reference
│   ├── developer_guide.md
│   └── testing.md
└── src/
    ├── data_factory/README_CN.md
    ├── model_factory/README_CN.md
    └── task_factory/README_CN.md
```

## Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass
2. **Check Style**: Follow code standards
3. **Update Documentation**: Add/update relevant docs
4. **Add Tests**: Include tests for new functionality
5. **Update Registry**: For new configs, update `config_registry.csv`

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Configuration addition

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Configs validated (if applicable)

## Documentation
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Registry/atlas updated (if applicable)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and validations
2. **Code Review**: Maintainers review code quality and design
3. **Documentation Review**: Check docs are complete and accurate
4. **Approval**: At least one maintainer approval required

### After Approval

1. **Squash and Merge**: We typically squash commits
2. **Update Changelog**: Maintainers update the changelog
3. **Release Notes**: Significant changes included in release notes

## Required Change Order

For config-related changes:
1. Registry → 2. Atlas → 3. Inspect → 4. Schema validate → 5. README → 6. CI/tests

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Maintain professional communication

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **CLAUDE.md**: For architecture and change strategy
- **AGENTS.md**: For development commands

## Contact

- **Maintainers**: [Qi Li](https://github.com/liq22), [Xuan Li](https://github.com/Xuan423)
- **GitHub**: [PHM-Vibench Repository](https://github.com/PHMbench/PHM-Vibench)

Thank you for contributing to PHM-Vibench!
