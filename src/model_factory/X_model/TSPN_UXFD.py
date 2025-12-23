"""TSPN_UXFD

Compatibility wrapper for UXFD merge.

This model intentionally stays close to the upstream UXFD `TSPN.py` structure:
`SignalProcessingLayer → FeatureExtractorlayer → Classifier`.

Implementation note:
- Today this wrapper reuses the existing `src/model_factory/X_model/TSPN.py` code path.
- Future UXFD modularization should live under `src/model_factory/X_model/UXFD/`.
"""

from __future__ import annotations

from .TSPN import Model as _TSPNModel


class Model(_TSPNModel):
    """Alias of the existing TSPN implementation for stable registry naming."""

    pass

