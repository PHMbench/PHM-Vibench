"""
评估指标工具模块，包含各种任务的评估指标实现
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    auc, mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, Any, Union, List, Optional, Tuple, Callable


# 分类指标
def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor], 
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """计算分类任务的评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签 (如果是原始模型输出，需要先进行argmax)
        y_prob: 预测概率 (用于计算AUC等指标)，形状为(n_samples, n_classes)
        class_names: 类别名称列表，用于命名指标
    
    Returns:
        包含各类评估指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor) and y_prob is not None:
        y_prob = y_prob.cpu().numpy()
    
    # 处理概率输出，如果传入的是原始模型输出
    if y_prob is None and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_prob = y_pred
        y_pred = np.argmax(y_pred, axis=1)
    
    # 计算基础指标
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # 计算每个类别的指标
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]
    else:
        assert len(class_names) == n_classes, "类别名称列表长度与类别数不匹配"
    
    # 计算每个类别的精确率、召回率和F1
    for i, class_name in enumerate(class_names):
        metrics[f"precision_{class_name}"] = precision_score(
            y_true, y_pred, labels=[i], average='macro', zero_division=0
        )
        metrics[f"recall_{class_name}"] = recall_score(
            y_true, y_pred, labels=[i], average='macro', zero_division=0
        )
        metrics[f"f1_{class_name}"] = f1_score(
            y_true, y_pred, labels=[i], average='macro', zero_division=0
        )
    
    # 对于二分类或多分类问题，计算AUC
    if y_prob is not None:
        if n_classes == 2:
            # 二分类问题，直接计算AUC
            if y_prob.shape[1] == 2:  # 如果有两个概率输出列
                y_prob_positive = y_prob[:, 1]
            else:  # 如果只有一个概率输出
                y_prob_positive = y_prob
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob_positive)
                # 计算PR AUC
                precision, recall, _ = precision_recall_curve(y_true, y_prob_positive)
                metrics["auc_pr"] = auc(recall, precision)
            except ValueError:
                # 如果所有样本都是同一个类别，会引发错误
                metrics["auc_roc"] = float('nan')
                metrics["auc_pr"] = float('nan')
        else:
            # 多分类问题，计算宏观AUC（每个类别一对多）
            try:
                metrics["auc_roc"] = roc_auc_score(
                    np.eye(n_classes)[y_true], y_prob, multi_class='ovr', average='macro'
                )
            except ValueError:
                metrics["auc_roc"] = float('nan')
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    return metrics


# 异常检测指标
def compute_anomaly_metrics(
    y_true: Union[np.ndarray, torch.Tensor], 
    y_pred: Union[np.ndarray, torch.Tensor],
    anomaly_score: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pos_label: int = 1,  # 异常类别的标签
) -> Dict[str, float]:
    """计算异常检测任务的评估指标
    
    Args:
        y_true: 真实标签 (0表示正常，1表示异常)
        y_pred: 预测标签 (0表示正常，1表示异常)
        anomaly_score: 异常分数，用于计算AUC等指标
        pos_label: 正类标签，默认为1（异常）
    
    Returns:
        包含各类评估指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(anomaly_score, torch.Tensor) and anomaly_score is not None:
        anomaly_score = anomaly_score.cpu().numpy()
    
    # 计算基础指标
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }
    
    # 计算AUC
    if anomaly_score is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, anomaly_score)
            # 计算PR AUC
            precision, recall, _ = precision_recall_curve(y_true, anomaly_score, pos_label=pos_label)
            metrics["auc_pr"] = auc(recall, precision)
        except ValueError:
            # 如果所有样本都是同一个类别，会引发错误
            metrics["auc_roc"] = float('nan')
            metrics["auc_pr"] = float('nan')
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    # 计算特定于异常检测的指标
    # 假阳性率 (FPR)
    tn, fp, fn, tp = cm.ravel()
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 假发现率 (FDR)
    metrics["false_discovery_rate"] = fp / (fp + tp) if (fp + tp) > 0 else 0
    
    return metrics


# RUL预测指标
def compute_rul_metrics(
    y_true: Union[np.ndarray, torch.Tensor], 
    y_pred: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5  # α-λ准确度的参数
) -> Dict[str, float]:
    """计算RUL预测任务的评估指标
    
    Args:
        y_true: 真实RUL值
        y_pred: 预测RUL值
        alpha: α-λ准确度的参数
    
    Returns:
        包含各类评估指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 确保形状一致
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # 计算基础回归指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 计算RUL特定指标
    
    # 平均相对误差百分比 (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
    
    # 评分函数 1：惩罚早期预测的指数评分（PHM数据挑战中常用）
    errors = y_pred - y_true
    s1 = np.sum(np.exp(errors / 13) - 1) if np.any(errors > 0) else 0  # 早期预测 (过早预测故障)
    s2 = np.sum(np.exp(-errors / 10) - 1) if np.any(errors < 0) else 0  # 晚期预测 (过晚预测故障)
    score = s1 + s2
    
    # α-λ准确度：预测值在真实值的±α%范围内的比例
    alpha_lambda_accuracy = np.mean(
        np.abs(y_pred - y_true) <= (alpha * y_true)
    )
    
    # 偏差（评估预测模型是否系统性地过高或过低估计RUL）
    bias = np.mean(y_pred - y_true)
    
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape if not np.isinf(mape) else float('nan'),
        "score": score,
        "alpha_lambda_accuracy": alpha_lambda_accuracy,
        "bias": bias
    }
    
    return metrics


# 注册的指标函数
_METRIC_FUNCTIONS = {
    "classification": compute_classification_metrics,
    "anomaly": compute_anomaly_metrics,
    "rul": compute_rul_metrics,
}


def get_metric_function(task_type: str) -> Callable:
    """获取指定任务类型的评估指标函数
    
    Args:
        task_type: 任务类型，可选值为'classification', 'anomaly', 'rul'
        
    Returns:
        对应的评估指标函数
    
    Raises:
        ValueError: 如果指定的任务类型不受支持
    """
    if task_type not in _METRIC_FUNCTIONS:
        raise ValueError(f"不支持的任务类型: {task_type}，"
                        f"支持的任务类型有: {list(_METRIC_FUNCTIONS.keys())}")
    return _METRIC_FUNCTIONS[task_type]


def compute_metrics(
    task_type: str,
    y_true: Union[np.ndarray, torch.Tensor], 
    y_pred: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Dict[str, float]:
    """计算指定任务类型的评估指标
    
    Args:
        task_type: 任务类型，可选值为'classification', 'anomaly', 'rul'
        y_true: 真实值
        y_pred: 预测值
        **kwargs: 传递给具体评估指标函数的其他参数
    
    Returns:
        包含各类评估指标的字典
    """
    metric_func = get_metric_function(task_type)
    return metric_func(y_true, y_pred, **kwargs)