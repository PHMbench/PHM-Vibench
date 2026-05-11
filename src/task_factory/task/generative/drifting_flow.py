from __future__ import annotations

from src.task_factory.task.generative._experimental_one_step import (
    ExperimentalOneStepFlowTask,
)


class DriftingFlowTask(ExperimentalOneStepFlowTask):
    method_id = "drifting_flow_experimental"


task = DriftingFlowTask
