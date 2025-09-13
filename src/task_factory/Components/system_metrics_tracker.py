"""
系统特定指标追踪器
分离追踪不同系统的指标，避免指标混合影响调试
"""
import numpy as np
from typing import Dict, List, Any
import torch


class SystemMetricsTracker:
    """分离追踪不同系统的指标"""
    
    def __init__(self):
        """初始化追踪器"""
        self.system_metrics = {}  # {system_id: {metric_name: [values]}}
        self.batch_count = 0  # 批次计数
        
    def update(self, system_id: Any, metrics_dict: Dict[str, Any]):
        """更新特定系统的指标
        
        Args:
            system_id: 系统标识符
            metrics_dict: 指标字典 {metric_name: value}
        """
        # 确保system_id可以作为字典键
        sys_key = str(system_id)
        
        if sys_key not in self.system_metrics:
            self.system_metrics[sys_key] = {}
            
        for metric_name, metric_value in metrics_dict.items():
            # 处理tensor类型的指标值
            if torch.is_tensor(metric_value):
                metric_value = metric_value.detach().cpu().numpy()
                if metric_value.ndim == 0:  # 标量tensor
                    metric_value = float(metric_value)
                    
            # 跳过无效值
            if metric_value is None or (isinstance(metric_value, float) and 
                                       (np.isnan(metric_value) or np.isinf(metric_value))):
                continue
                
            # 存储指标值
            if metric_name not in self.system_metrics[sys_key]:
                self.system_metrics[sys_key][metric_name] = []
            self.system_metrics[sys_key][metric_name].append(metric_value)
        
        self.batch_count += 1
    
    def compute_epoch_metrics(self, aggregation='mean') -> Dict[str, Dict[str, float]]:
        """计算每个系统的epoch级指标
        
        Args:
            aggregation: 聚合方式，'mean', 'median', 'last'等
            
        Returns:
            {system_id: {metric_name: aggregated_value}}
        """
        results = {}
        
        for sys_id, metrics in self.system_metrics.items():
            results[sys_id] = {}
            
            for metric_name, values in metrics.items():
                if not values:
                    continue
                    
                # 根据聚合方式计算最终值
                if aggregation == 'mean':
                    results[sys_id][metric_name] = float(np.mean(values))
                elif aggregation == 'median':
                    results[sys_id][metric_name] = float(np.median(values))
                elif aggregation == 'last':
                    results[sys_id][metric_name] = float(values[-1])
                elif aggregation == 'max':
                    results[sys_id][metric_name] = float(np.max(values))
                elif aggregation == 'min':
                    results[sys_id][metric_name] = float(np.min(values))
                else:
                    # 默认使用平均值
                    results[sys_id][metric_name] = float(np.mean(values))
        
        return results
    
    def get_system_count(self) -> int:
        """获取追踪的系统数量"""
        return len(self.system_metrics)
    
    def get_systems(self) -> List[str]:
        """获取所有系统ID"""
        return list(self.system_metrics.keys())
    
    def clear(self):
        """清空所有追踪数据"""
        self.system_metrics.clear()
        self.batch_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total_systems': self.get_system_count(),
            'total_batches': self.batch_count,
            'systems': self.get_systems()
        }
        
        # 计算每个系统的批次数
        batch_counts = {}
        for sys_id, metrics in self.system_metrics.items():
            if metrics:
                # 假设所有指标的长度相同
                first_metric = list(metrics.values())[0]
                batch_counts[sys_id] = len(first_metric)
        
        stats['batches_per_system'] = batch_counts
        return stats
    
    def print_summary(self, phase: str = ""):
        """打印追踪器摘要信息"""
        stats = self.get_statistics()
        
        print(f"\n=== SystemMetricsTracker Summary {phase}===")
        print(f"Total Systems: {stats['total_systems']}")
        print(f"Total Batches: {stats['total_batches']}")
        
        if stats['systems']:
            print("Systems tracked:", ", ".join(stats['systems']))
            
        if stats['batches_per_system']:
            print("Batches per system:")
            for sys_id, count in stats['batches_per_system'].items():
                print(f"  - System {sys_id}: {count} batches")
        
        print("=" * (30 + len(phase)))


if __name__ == '__main__':
    """单元测试"""
    print("=== Testing SystemMetricsTracker ===")
    
    # 创建追踪器
    tracker = SystemMetricsTracker()
    
    # 模拟添加指标
    tracker.update('system_1', {
        'classification_acc': 0.95,
        'classification_f1': 0.93,
        'anomaly_auroc': 0.87
    })
    
    tracker.update('system_1', {
        'classification_acc': 0.94,
        'classification_f1': 0.92,
        'anomaly_auroc': 0.89
    })
    
    tracker.update('system_5', {
        'classification_acc': 0.85,
        'classification_f1': 0.82,
        'anomaly_auroc': 0.02  # 异常值
    })
    
    # 打印统计信息
    tracker.print_summary("Test")
    
    # 计算epoch指标
    epoch_metrics = tracker.compute_epoch_metrics()
    print("\nEpoch Metrics:")
    for sys_id, metrics in epoch_metrics.items():
        print(f"System {sys_id}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n✓ SystemMetricsTracker test completed!")