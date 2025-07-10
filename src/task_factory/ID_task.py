import torch
from typing import Any, Dict, Tuple
from .Default_task import Default_task
from .utils.data_processing import prepare_batch


class task(Default_task):
    """Task that dynamically processes data from :class:`ID_dataset`."""

    def _shared_step(
        self,
        batch: Tuple,
        stage: str,
        task_id: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Override to preprocess raw batches on-the-fly."""
        if 'x' not in batch:
            batch = prepare_batch(batch, self.args_data)
        return super()._shared_step(batch, stage, task_id)

