from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from phmfactory.commands import demo


def test_demo_uses_bundled_data_and_preserves_user_override_order() -> None:
    observed: list[argparse.Namespace] = []
    custom_data_dir = "/tmp/custom-phm-data"

    result = demo.run(
        ["--override", f"data.data_dir={custom_data_dir}"],
        experiment_runner=lambda args: observed.append(args) or "ok",
    )

    assert result == "ok"
    args = observed[0]
    packaged_data_dir = Path(str(resources.files("data"))).resolve()
    assert (packaged_data_dir / "metadata_dummy.csv").is_file()
    assert (packaged_data_dir / "raw" / "Dummy_Data" / "dummy1.csv").is_file()
    assert args.override[0] == f"data.data_dir={packaged_data_dir}"
    assert args.override[-1] == f"data.data_dir={custom_data_dir}"
