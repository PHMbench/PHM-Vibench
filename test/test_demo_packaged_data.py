from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from phmfactory.commands import demo


def test_demo_uses_bundled_data_with_an_ephemeral_writable_cache() -> None:
    observed: list[argparse.Namespace] = []
    cache_paths: list[Path] = []
    custom_data_dir = "/tmp/custom-phm-data"

    def run_experiment(args: argparse.Namespace) -> str:
        observed.append(args)
        cache_override = next(
            item for item in args.override if item.startswith("data.cache_dir=")
        )
        cache_path = Path(cache_override.split("=", 1)[1])
        assert cache_path.is_dir()
        cache_paths.append(cache_path)
        return "ok"

    result = demo.run(
        ["--override", f"data.data_dir={custom_data_dir}"],
        experiment_runner=run_experiment,
    )

    assert result == "ok"
    args = observed[0]
    packaged_data_dir = Path(str(resources.files("data"))).resolve()
    packaged_override = f"data.data_dir={packaged_data_dir}"
    user_override = f"data.data_dir={custom_data_dir}"

    assert (packaged_data_dir / "metadata_dummy.csv").is_file()
    assert (packaged_data_dir / "raw" / "Dummy_Data" / "dummy1.csv").is_file()
    assert args.override[:3] == list(demo.DEFAULT_OVERRIDES)
    assert args.override.index(packaged_override) < args.override.index(user_override)
    assert args.override[-1] == user_override
    assert cache_paths and not cache_paths[0].exists()
