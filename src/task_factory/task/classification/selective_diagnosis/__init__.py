"""Finite-unit selective diagnosis, certificates and explicit feature-track execution.

The data archive supplies declared physical-unit identities. Software validation
cannot establish their physical independence or upstream feature fit provenance.
"""
from .core import (box_bounds, calibrate, calibration_capacity, class_certificate,
                   matched_risk, max_box_average, risk_coverage, unit_weights)

__all__ = ['box_bounds', 'calibrate', 'calibration_capacity', 'class_certificate',
           'matched_risk', 'max_box_average', 'risk_coverage', 'unit_weights']

from .execution import PHASES, execute
