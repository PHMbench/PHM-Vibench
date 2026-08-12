"""
Post-hoc explanation methods

These methods provide explanations after the model has been trained,
without requiring modifications to the model architecture.
"""

from .captum_wrapper import CaptumWrapper

__all__ = ["CaptumWrapper"]