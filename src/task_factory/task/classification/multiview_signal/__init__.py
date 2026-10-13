"""Source-frozen multiview signal readouts; numerical helpers do not access data."""

from .features import (ProgramSpec, SignalFeatureSpec, extract_signal_features,
                       feature_names, program_features, view_indices)
from .readouts import (OffsetReadout, ReadoutState, baseline_quality,
                       fit_readouts, project_brier, source_guard)
from .metrics import (aggregate_metrics, equal_unit_weights, group_means,
                      paired_brier_summary, per_unit_metrics)
from .runtime import PHASES, execute

__all__ = [
    "ProgramSpec", "SignalFeatureSpec", "extract_signal_features", "feature_names",
    "program_features", "view_indices", "OffsetReadout", "ReadoutState",
    "baseline_quality", "fit_readouts", "project_brier", "source_guard",
    "aggregate_metrics", "equal_unit_weights", "group_means", "paired_brier_summary",
    "per_unit_metrics",
    "PHASES", "execute",
]
