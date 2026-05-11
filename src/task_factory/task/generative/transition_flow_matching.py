from __future__ import annotations

from src.task_factory.task.generative._experimental_one_step import (
    ExperimentalOneStepFlowTask,
)


class TransitionFlowMatchingTask(ExperimentalOneStepFlowTask):
    method_id = "transition_flow_matching_experimental"


task = TransitionFlowMatchingTask
