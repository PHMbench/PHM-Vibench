"""
Analysis tools for ContrastiveIDTask experiments.

This module provides comprehensive analysis tools including:
- Ablation studies for hyperparameter analysis
- Visualization tools for feature analysis and training curves  
- Baseline method comparison tools
"""

from .ablation_studies import AblationStudyRunner
from .visualization_tools import VisualizationTools
from .baseline_comparison import BaselineComparison

__all__ = [
    'AblationStudyRunner',
    'VisualizationTools', 
    'BaselineComparison'
]