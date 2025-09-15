from .H_01_Linear_cla import H_01_Linear_cla
from .H_02_distance_cla import H_02_distance_cla
from .H_03_Linear_pred import H_03_Linear_pred
from .H_09_multiple_task import H_09_multiple_task # 新增导入
from .H_04_VIB_pred import H_04_VIB_pred
from .H_10_ProjectionHead import H_10_ProjectionHead
from .multi_task_head import MultiTaskHead

__all__ = ["H_01_Linear_cla",
            "H_02_distance_cla",
              "H_03_Linear_pred",
                "H_09_multiple_task",
                "H_04_VIB_pred",
                "H_10_ProjectionHead",
                "MultiTaskHead"] # 新增到 __all__
