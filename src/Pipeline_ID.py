"""Pipeline wrapper for ID datasets."""

from src.Pipeline_01_default import pipeline as default_pipeline


def pipeline(args):
    """Run the default pipeline. Configs should specify id_data_factory."""
    return default_pipeline(args)
