from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")

class Registry:
    """Simple name-to-object registry with decorator support."""

    def __init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers ``obj`` under ``name``."""
        def decorator(obj: T) -> T:
            self._items[name] = obj
            return obj
        return decorator

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"{name} is not registered")
        return self._items[name]

    def available(self) -> Dict[str, T]:
        return dict(self._items)

__all__ = ["Registry"]
