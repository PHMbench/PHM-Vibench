# Contributing to PHM-Vibench Model Factory

We welcome contributions to the PHM-Vibench Model Factory! This guide will help you understand how to contribute effectively to the project.

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Adding New Models](#adding-new-models)
5. [Code Standards](#code-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Git
- Basic understanding of deep learning and time-series analysis

### Development Setup

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
pip install -r requirements-dev.txt  # Development dependencies
```

4. **Install in Development Mode**
```bash
pip install -e .
```

5. **Run Tests**
```bash
python -m pytest tests/
```

## 📝 Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **New Model Implementations**: SOTA models for time-series analysis
2. **Bug Fixes**: Fixing issues in existing code
3. **Documentation**: Improving docs, examples, and tutorials
4. **Performance Improvements**: Optimizations and efficiency gains
5. **Feature Enhancements**: New functionality and utilities

### Before You Start

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create an Issue**: For new features or significant changes
3. **Discuss First**: For major changes, discuss with maintainers
4. **Follow Standards**: Adhere to our coding and documentation standards

## 🏗️ Adding New Models

### Model Implementation Template

```python
"""
New Model Implementation for PHM-Vibench Model Factory

Brief description of the model and its key features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any


class Model(nn.Module):
    """Model Name for time-series analysis.
    
    Detailed description of the model, its architecture,
    and key innovations.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, hidden dimension (default: 256)
        - num_layers : int, number of layers (default: 6)
        - dropout : float, dropout probability (default: 0.1)
        - num_classes : int, number of output classes (for classification)
        - output_dim : int, output dimension (for regression)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)
        
    Output Shape
    ------------
    torch.Tensor
        For classification: (batch_size, num_classes)
        For regression: (batch_size, seq_len, output_dim)
        
    References
    ----------
    Author et al. "Paper Title" Conference/Journal Year.
    Supporting references for key components.
    Adapted for time-series industrial signals with specific modifications.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters with defaults
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.num_layers = getattr(args, 'num_layers', 6)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Model architecture implementation
        # ...
        
        # Task-specific heads
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
            self.task_type = 'classification'
        else:
            self.regressor = nn.Linear(self.hidden_dim, self.output_dim)
            self.task_type = 'regression'
    
    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)
            
        Returns
        -------
        torch.Tensor
            Output tensor shape depends on task type
        """
        # Implementation
        # ...
        
        return output


if __name__ == "__main__":
    # Test implementation
    def test_model():
        """Test model with different configurations."""
        print("Testing Model...")
        
        # Test configuration
        from argparse import Namespace
        
        args = Namespace(
            input_dim=3,
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            num_classes=5
        )
        
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 64
        x = torch.randn(batch_size, seq_len, args.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"Input: {x.shape}, Output: {output.shape}")
        assert output.shape == (batch_size, args.num_classes)
        
        print("✅ Model tests passed!")
        return True
    
    test_model()
```

### Model Registration

Add your model to the appropriate category's `__init__.py`:

```python
# In src/model_factory/CategoryName/__init__.py
from .YourModel import Model as YourModel

__all__ = [
    # ... existing models
    "YourModel"
]
```

### Model Documentation

Create a README for your model category if it doesn't exist:

```markdown
# Your Model Category

Brief description of the model category and its applications.

## Available Models

### YourModel
**Paper**: Author et al. "Paper Title" Conference Year

Brief description of the model and its key features.

**Configuration**:
```python
args = Namespace(
    model_name='YourModel',
    input_dim=3,
    hidden_dim=256,
    # ... other parameters
)
```
```

## 📏 Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

1. **Line Length**: 100 characters maximum
2. **Imports**: Group imports (standard, third-party, local)
3. **Naming**: 
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_CASE`

### Code Quality Tools

```bash
# Format code
black src/ tests/ examples/

# Check style
flake8 src/ tests/ examples/

# Type checking
mypy src/

# Sort imports
isort src/ tests/ examples/
```

### Docstring Standards

Use NumPy-style docstrings:

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
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

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_models/      # Model-specific tests
│   ├── test_utils/       # Utility function tests
│   └── test_factory/     # Factory function tests
├── integration/          # Integration tests
├── performance/          # Performance benchmarks
└── fixtures/            # Test data and fixtures
```

### Writing Tests

```python
import pytest
import torch
from argparse import Namespace
from src.model_factory import build_model


class TestYourModel:
    """Test suite for YourModel."""
    
    @pytest.fixture
    def model_args(self):
        """Model configuration for testing."""
        return Namespace(
            model_name='YourModel',
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            num_classes=4
        )
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return torch.randn(8, 32, 3)  # (batch, seq_len, features)
    
    def test_model_creation(self, model_args):
        """Test model can be created successfully."""
        model = build_model(model_args)
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_forward_pass(self, model_args, sample_data):
        """Test forward pass produces correct output shape."""
        model = build_model(model_args)
        output = model(sample_data)
        
        expected_shape = (8, 4)  # (batch_size, num_classes)
        assert output.shape == expected_shape
    
    def test_parameter_count(self, model_args):
        """Test model has reasonable parameter count."""
        model = build_model(model_args)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Reasonable bounds for parameter count
        assert 1000 < param_count < 10_000_000
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, model_args, batch_size):
        """Test model works with different batch sizes."""
        model = build_model(model_args)
        x = torch.randn(batch_size, 32, 3)
        output = model(x)
        
        assert output.shape[0] == batch_size
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models/test_your_model.py

# Run with coverage
pytest --cov=src tests/

# Run performance tests
pytest tests/performance/ -v
```

## 📚 Documentation

### Documentation Requirements

1. **Model Documentation**: Comprehensive docstrings
2. **Usage Examples**: Working code examples
3. **API Reference**: Parameter descriptions
4. **Performance Benchmarks**: Speed and accuracy metrics

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## 🔄 Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass
2. **Check Style**: Run code quality tools
3. **Update Documentation**: Add/update relevant docs
4. **Add Tests**: Include tests for new functionality
5. **Benchmark Performance**: For new models, include benchmarks

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Performance benchmarks included (for new models)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Reviewers test functionality
4. **Documentation Review**: Check docs are complete and accurate
5. **Approval**: At least one maintainer approval required

### After Approval

1. **Squash and Merge**: We typically squash commits
2. **Update Changelog**: Maintainers update the changelog
3. **Release Notes**: Significant changes included in release notes

## 🏷️ Release Process

### Versioning

We use Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update changelog
3. Run full test suite
4. Build and test documentation
5. Create release tag
6. Publish to PyPI (if applicable)

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Maintain professional communication

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private matters or security issues

## 📞 Contact

- **Maintainers**: List of current maintainers
- **Email**: phm-vibench@example.com
- **GitHub**: [PHM-Vibench Repository](https://github.com/PHMbench/PHM-Vibench)

Thank you for contributing to PHM-Vibench! 🎉
