from __future__ import annotations

from src.task_factory.task.generative._experimental_one_step import (
    ExperimentalOneStepFlowTask,
)


class MeanflowTask(ExperimentalOneStepFlowTask):
    method_id = "meanflow_imf_experimental"


task = MeanflowTask
