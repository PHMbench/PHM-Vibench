"""Dependency-bounded generative model namespace.

Models are resolved dynamically by ``model.type`` / ``model.name``. Keep this
package free of eager model imports so selecting one generative model never
requires unrelated research backbones or optional dependencies.
"""

__all__: list[str] = []
