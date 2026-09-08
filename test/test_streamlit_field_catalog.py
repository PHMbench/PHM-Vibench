from __future__ import annotations

from pathlib import Path

from apps.streamlit import config_service as cs


CATALOG = Path(__file__).parents[1] / "apps" / "streamlit" / "field_catalog.yaml"


def test_common_controls_use_only_current_maintained_paths() -> None:
    catalog = cs.load_catalog(CATALOG)
    paths = {spec.key: spec.paths for spec in catalog.fields}

    assert paths == {
        "device": ("trainer.device",),
        "devices": ("trainer.devices",),
        "epochs": ("trainer.num_epochs",),
        "iterations": ("environment.iterations",),
        "seed": ("environment.seed",),
        "batch_size": ("data.batch_size",),
        "learning_rate": ("task.lr",),
        "num_workers": ("data.num_workers",),
        "test_after_fit": ("trainer.test_after_fit",),
        "data_dir": ("data.data_dir",),
        "metadata_file": ("data.metadata_file",),
        "output_dir": ("environment.output_dir",),
    }


def test_quick_start_exposes_the_small_reproducible_surface() -> None:
    catalog = cs.load_catalog(CATALOG)
    quick = {spec.key for spec in catalog.fields if spec.quick_start}

    assert quick == {
        "device",
        "devices",
        "epochs",
        "iterations",
        "seed",
        "batch_size",
        "learning_rate",
        "num_workers",
    }


def test_legacy_parameter_aliases_are_not_public_controls() -> None:
    catalog = cs.load_catalog(CATALOG)
    public_paths = {path for spec in catalog.fields for path in spec.paths}

    assert public_paths.isdisjoint(
        {
            "task.epochs",
            "task.batch_size",
            "task.num_workers",
            "trainer.learning_rate",
            "trainer.lr",
            "trainer.batch_size",
            "trainer.num_workers",
            "model.learning_rate",
            "output_dir",
        }
    )
