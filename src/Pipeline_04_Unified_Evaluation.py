"""Unified two-stage evaluation pipeline."""

import argparse
from typing import Any, Dict

from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator
from src.utils.config.pipeline_adapters import adapt_p04
from src.utils.utils import close_lab


def pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the canonical unified orchestrator or propagate its failure."""
    try:
        unified = adapt_p04(args.config_path, getattr(args, "local_config", None))
        orchestrator = TwoStageOrchestrator(unified)
        summary = orchestrator.run_complete()
        return {"results": summary, "unified": True}
    finally:
        close_lab()


if __name__ == "__main__":
    """
    Direct execution for testing purposes.
    Normally this pipeline is called via main.py --pipeline Pipeline_04_unified_metric
    """
    parser = argparse.ArgumentParser(description="Unified Metric Learning Pipeline")
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to unified metric learning configuration file')
    parser.add_argument('--notes', type=str, default='',
                       help='Experiment notes')

    args = parser.parse_args()

    # Run the pipeline
    results = pipeline(args)

    print("✅ Pipeline completed successfully")
    print(f"📊 Summary: {results}")
