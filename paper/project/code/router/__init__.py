# 路由器模块

try:
    from .statistical_router import StatisticalRouter
except ImportError:
    from statistical_router import StatisticalRouter

__all__ = [
    'StatisticalRouter'
]