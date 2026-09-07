"""Public PHMFactory package surface.

The v0.3 compatibility release exposes a stable package name while the mature
runtime remains under :mod:`src`.  Public modules should import through this
package; internal migration of the protected runtime is intentionally deferred.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.3.0rc1"
