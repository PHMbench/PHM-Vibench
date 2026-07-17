"""Transformer family models loaded lazily by the model factory."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "TransformerDummy": (".Transformer_Dummy", "Model"),
    "Informer": (".Informer", "Model"),
    "Autoformer": (".Autoformer", "Model"),
    "PatchTST": (".PatchTST", "Model"),
    "Linformer": (".Linformer", "Model"),
    "ConvTransformer": (".ConvTransformer", "Model"),
    "TSLTransformer": (".TSLTransformer", "Model"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(module_name, __name__), symbol)
