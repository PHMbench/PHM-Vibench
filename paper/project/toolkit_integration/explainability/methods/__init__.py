"""
Explanation methods module

Contains different types of explanation methods organized by category.
"""

# Import intrinsic explainers
from .intrinsic.signal_path_explainer import SignalPathExplainer
from .intrinsic.path_analysis_explainer import PathAnalysisExplainer
from .intrinsic.operator_weight_explainer import OperatorWeightExplainer

# Import post-hoc explainers
from .posthoc.gradcam_explainer import GradCAMExplainer
from .posthoc.shap_explainer import SHAPExplainer

# Import existing captum wrapper for compatibility
try:
    from .posthoc.captum_wrapper import CaptumWrapper
except ImportError:
    CaptumWrapper = None

__all__ = [
    "SignalPathExplainer",
    "PathAnalysisExplainer",
    "OperatorWeightExplainer",
    "GradCAMExplainer",
    "SHAPExplainer"
]

# Optional imports
if CaptumWrapper is not None:
    __all__.append("CaptumWrapper")