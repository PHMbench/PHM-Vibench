"""Public PHMFactory package surface.

The v0.3 compatibility release exposes a stable package name while the mature
runtime remains under :mod:`src`.  Public modules should import through this
package; internal migration of the protected runtime is intentionally deferred.
"""

from __future__ import annotations

__all__ = ["__version__", "installed_build_identity"]

__version__ = "0.3.0rc1"


def installed_build_identity() -> dict:
    """Identify the installed distribution's RECORD, without invoking Git.

    RECORD is installation metadata, not a claim that an uncommitted checkout is
    a release or that data/model artifacts were validated. A source import must
    never borrow the identity of another installed distribution.
    """
    import hashlib
    from importlib.metadata import PackageNotFoundError, distribution
    from pathlib import Path

    current = Path(__file__).resolve()
    try:
        dist = distribution("phmfactory")
    except PackageNotFoundError:
        return {"kind": "source_checkout", "version": __version__, "record_sha256": None}
    record = dist.read_text("RECORD")
    packaged = Path(dist.locate_file("phmfactory/__init__.py")).resolve()
    if packaged != current or record is None:
        return {"kind": "source_checkout", "version": __version__, "record_sha256": None}
    return {"kind": "installed_distribution", "version": dist.version,
            "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest()}
