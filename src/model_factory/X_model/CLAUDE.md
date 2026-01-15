# X_model - CLAUDE.md

This module provides architecture guidance for XAI (Explainable AI) and auxiliary models in PHM-Vibench. For available models, see [@README.md].

## Overview

The X_model directory contains models designed for:
- **Explainable AI**: Interpretable predictions
- **Auxiliary tasks**: Supporting main model training
- **Analysis tools**: Signal interpretation and visualization

## Architecture Focus

Unlike other model directories focused on prediction accuracy, X_model emphasizes:
- **Interpretability**: Understanding model decisions
- **Visualization**: Making signal features visible
- **Debugging**: Tools for model analysis

## Typical Use Cases

1. **Attention Visualization**: Show which time steps are important
2. **Feature Importance**: Identify contributing frequency bands
3. **Saliency Maps**: Visualize input influence on output

## Design Philosophy

### Transparency First
- Models should be interpretable
- Decision process should be visible
- Support for post-hoc analysis

### Integration with Main Models
X_model components can work alongside:
- ISFM for attention analysis
- CNN for feature visualization
- Any model for explainability

## Configuration

X_model configurations depend on the specific analysis task. Refer to [@README.md] for model-specific options.

## Related Documentation

- [@README.md] - Available Models and Configuration
- [@../README.md] - Model Factory Overview
