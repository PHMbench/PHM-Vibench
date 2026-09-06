"""Dataset binding for the P09 reliability-conditioned GFS task.

The method consumes the maintained window representation. Episode membership
and support/query ordering remain the sampler's responsibility and are audited
at the protocol-lock stage.
"""

from .Classification_dataset import set_dataset

__all__ = ["set_dataset"]
