from __future__ import annotations

from src.task_factory.task.generative._experimental_one_step import (
    ExperimentalOneStepFlowTask,
)


class OtNfmTask(ExperimentalOneStepFlowTask):
    method_id = "ot_nfm_experimental"


task = OtNfmTask
