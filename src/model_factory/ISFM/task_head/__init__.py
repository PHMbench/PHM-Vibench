from .H_01_Linear_cla import H_01_Linear_cla
from .H_02_distance_cla import H_02_distance_cla
from .H_03_Linear_pred import H_03_Linear_pred
from .H_05_RUL_pred import H_05_RUL_pred
from .H_06_Anomaly_det import H_06_Anomaly_det
from .H_09_multiple_task import H_09_multiple_task # 新增导入
from .H_04_VIB_pred import H_04_VIB_pred
from .multi_task_head import MultiTaskHead

__all__ = ["H_01_Linear_cla",
            "H_02_distance_cla",
              "H_03_Linear_pred",
                "H_05_RUL_pred",
                "H_06_Anomaly_det",
                "H_09_multiple_task",
                "H_04_VIB_pred",
                "MultiTaskHead"] # 新增到 __all__
