"""
Lightweight fault-domain knowledge base.

This module keeps the public ``FaultKnowledgeBase`` import available without
requiring the absent legacy ``knowledge`` package.
"""

from __future__ import annotations

from typing import Dict, List, Union


class FaultKnowledgeBase:
    """Minimal deterministic knowledge base for local toolkit operation."""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Union[List[str], str]]] = {
            "内圈故障": {
                "description": "轴承内圈可能存在局部损伤或磨损。",
                "causes": ["材料疲劳", "润滑不足", "冲击载荷", "安装偏差"],
                "actions": ["复核包络谱", "检查润滑状态", "安排停机复检"],
            },
            "外圈故障": {
                "description": "轴承外圈可能存在点蚀、裂纹或磨损。",
                "causes": ["载荷集中", "污染颗粒", "润滑退化"],
                "actions": ["检查轴承座", "复核外圈特征频率", "制定维修窗口"],
            },
            "不平衡": {
                "description": "转子质量分布不均可能导致转频振动增强。",
                "causes": ["积灰", "磨损", "装配偏心"],
                "actions": ["清洁转子", "执行动平衡校正"],
            },
            "不对中": {
                "description": "轴系或联轴器不对中可能导致转频倍频异常。",
                "causes": ["安装误差", "基础松动", "热变形"],
                "actions": ["检查联轴器", "复测轴线同心度"],
            },
        }

    def get_fault_info(self, fault_type: str) -> Dict[str, Union[List[str], str]]:
        """Return known information for a fault type."""
        return self._records.get(
            fault_type,
            {
                "description": "暂无该故障类型的专用知识。",
                "causes": [],
                "actions": ["结合信号证据和维护记录进行人工复核"],
            },
        )

    def list_fault_types(self) -> List[str]:
        """Return supported fault labels."""
        return list(self._records)
