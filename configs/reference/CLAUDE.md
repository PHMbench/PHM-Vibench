# Reference Configurations - CLAUDE.md

This module provides architecture guidance for legacy reference configurations in PHM-Vibench. For current usage, see [@README.md].

## Overview

The `configs/reference/` directory contains **legacy configurations** maintained for backward compatibility and reference.

## Status

⚠️ **This directory is planned for migration/removal.**

For new experiments, use:
- **`configs/demo/`** - Maintained template configurations
- **`configs/experiments/`** - Your local experiments

## Migration Guide

If you have code using reference configs:

### Old Way (Reference)
```yaml
# Old path (deprecated)
 configs/reference/legacy_experiment.yaml
```

### New Way (Demo)
```yaml
# Use demo templates
 configs/demo/01_cross_domain/cwru_dg.yaml
```

### Or Create Your Own
```yaml
# In configs/experiments/
base_configs:
  - configs/base/environment.yaml
  - configs/demo/01_cross_domain/cwru_dg.yaml

# Your overrides
data:
  batch_size: 128
```

## Purpose of Reference Configs

- **Historical**: Shows how configs were structured in older versions
- **Debugging**: Helps with legacy code issues
- **Migration**: Guide for updating old experiments

## Related Documentation

- [@README.md] - Reference Configuration List
- [@../README.md] - Configuration System
- [@../demo/CLAUDE.md] - Current Demo Configurations
