from __future__ import annotations

import importlib
import inspect


def test_pipeline06_imports_without_optional_generative_dependencies() -> None:
    module = importlib.import_module("src.Pipeline_06_Generative_Modeling")

    assert callable(module.pipeline)
    assert module.STAGE_NAMES == frozenset({"train", "sample", "eval"})


def test_pipeline06_has_no_independent_cli_entrypoint() -> None:
    module = importlib.import_module("src.Pipeline_06_Generative_Modeling")
    source = inspect.getsource(module)

    assert "argparse" not in source
    assert "if __name__ ==" not in source
    assert "--config_path" not in source
