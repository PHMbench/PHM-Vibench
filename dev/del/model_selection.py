"""
模型选择工具模块，用于模型验证、超参数优化和模型选择
"""
import numpy as np
import pandas as pd
import torch
import os
from copy import deepcopy
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, 
    GridSearchCV, RandomizedSearchCV
)


class ModelValidator:
    """模型验证器，用于K折交叉验证和留一验证
    """
    
    def __init__(
        self,
        trainer_factory,
        n_splits: int = 5,
        stratify: bool = True,
        group_column: Optional[str] = None,
        random_state: int = 42,
        save_path: str = 'results/validation'
    ):
        """初始化模型验证器
        
        Args:
            trainer_factory: 训练器工厂，用于创建训练器实例
            n_splits: 折数
            stratify: 是否进行分层抽样
            group_column: 分组列名，若提供则进行分组K折交叉验证
            random_state: 随机种子
            save_path: 结果保存路径
        """
        self.trainer_factory = trainer_factory
        self.n_splits = n_splits
        self.stratify = stratify
        self.group_column = group_column
        self.random_state = random_state
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    def _get_kfold(self, X, y, groups=None):
        """获取K折交叉验证迭代器
        
        Args:
            X: 特征数据
            y: 标签数据
            groups: 分组数据
            
        Returns:
            K折交叉验证迭代器
        """
        if groups is not None:
            return GroupKFold(n_splits=self.n_splits)
        elif self.stratify and y is not None:
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
    
    def cross_validate(self, configs: Dict[str, Any], dataset, task_type: str = 'classification') -> Dict[str, Any]:
        """进行K折交叉验证
        
        Args:
            configs: 配置字典
            dataset: 数据集实例或数据加载器元组(X, y)
            task_type: 任务类型，用于选择评估指标
            
        Returns:
            包含验证结果的字典
        """
        # 处理数据集
        if hasattr(dataset, 'get_all_data'):
            X, y = dataset.get_all_data()
            groups = None
        elif isinstance(dataset, tuple) and len(dataset) >= 2:
            X, y = dataset[:2]
            groups = dataset[2] if len(dataset) > 2 else None
        else:
            raise ValueError("数据集格式不支持，需要实现get_all_data方法或提供(X, y[, groups])元组")
        
        # 获取K折交叉验证迭代器
        kfold = self._get_kfold(X, y, groups)
        
        # 准备结果存储
        all_results = []
        fold_metrics = {}
        
        # 开始K折交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y, groups)):
            print(f"\n{'='*50}\n第 {fold+1}/{self.n_splits} 折交叉验证\n{'='*50}")
            
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建数据集
            fold_dataset = self._create_fold_dataset(X_train, y_train, X_val, y_val)
            
            # 更新配置
            fold_configs = deepcopy(configs)
            fold_configs['dataset'] = fold_dataset
            
            # 创建训练器
            trainer = self.trainer_factory(fold_configs)
            
            # 执行训练和评估
            fold_path = os.path.join(self.save_path, f"fold_{fold+1}")
            os.makedirs(fold_path, exist_ok=True)
            
            result = trainer(fold_configs, fold_path, fold)
            all_results.append(result)
            
            # 保存折指标
            for k, v in result.items():
                if k not in fold_metrics:
                    fold_metrics[k] = []
                fold_metrics[k].append(v)
        
        # 计算平均指标和标准差
        summary = {}
        for k, v in fold_metrics.items():
            if isinstance(v[0], (int, float)):
                summary[f"{k}_mean"] = np.mean(v)
                summary[f"{k}_std"] = np.std(v)
        
        # 保存汇总结果
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(self.save_path, 'cv_summary.csv'), index=False)
        
        print(f"\n{'='*50}\n交叉验证完成\n{'='*50}")
        print("平均性能指标:")
        for k, v in summary.items():
            if k.endswith("_mean"):
                metric_name = k[:-5]
                print(f"{metric_name}: {v:.4f} ± {summary[f'{metric_name}_std']:.4f}")
        
        return {
            "fold_results": all_results,
            "summary": summary
        }
    
    def _create_fold_dataset(self, X_train, y_train, X_val, y_val):
        """创建用于交叉验证的数据集
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            包含训练和验证分割的数据集
        """
        # 返回一个实现了必要数据加载方法的简单数据集对象
        class FoldDataset:
            def __init__(self, X_train, y_train, X_val, y_val, batch_size=32):
                self.X_train = torch.tensor(X_train, dtype=torch.float32)
                self.y_train = torch.tensor(y_train)
                self.X_val = torch.tensor(X_val, dtype=torch.float32)
                self.y_val = torch.tensor(y_val)
                self.batch_size = batch_size
                
                # 创建数据集对象
                self.train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
                self.val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
                # 使用测试集作为验证集
                self.test_dataset = self.val_dataset
            
            def get_train_loader(self):
                return torch.utils.data.DataLoader(
                    self.train_dataset, batch_size=self.batch_size, shuffle=True
                )
            
            def get_val_loader(self):
                return torch.utils.data.DataLoader(
                    self.val_dataset, batch_size=self.batch_size, shuffle=False
                )
            
            def get_test_loader(self):
                return torch.utils.data.DataLoader(
                    self.test_dataset, batch_size=self.batch_size, shuffle=False
                )
            
            def get_data_loaders(self, batch_size=None):
                if batch_size is not None:
                    self.batch_size = batch_size
                return self.get_train_loader(), self.get_val_loader(), self.get_test_loader()
        
        return FoldDataset(X_train, y_train, X_val, y_val)


class HyperparameterTuner:
    """超参数调优器，用于网格搜索和随机搜索
    """
    
    def __init__(
        self,
        model_factory,
        param_grid: Dict[str, List],
        n_iter: Optional[int] = None,
        scoring: str = 'accuracy',
        cv: int = 3,
        random_state: int = 42,
        save_path: str = 'results/tuning'
    ):
        """初始化超参数调优器
        
        Args:
            model_factory: 模型工厂，用于创建模型实例
            param_grid: 参数网格，包含要搜索的参数及其可能值
            n_iter: 随机搜索的迭代次数，若为None则进行网格搜索
            scoring: 评分方法
            cv: 交叉验证折数
            random_state: 随机种子
            save_path: 结果保存路径
        """
        self.model_factory = model_factory
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.best_params = None
        self.best_score = None
    
    def fit(self, X, y, groups=None):
        """执行超参数调优
        
        Args:
            X: 特征数据
            y: 标签数据
            groups: 分组数据
            
        Returns:
            最佳参数和最佳分数
        """
        # 创建基础模型
        base_model = self.model_factory()
        
        # 创建搜索器
        if self.n_iter is None:
            # 网格搜索
            searcher = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=2
            )
        else:
            # 随机搜索
            searcher = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=2
            )
        
        # 执行搜索
        print(f"\n{'='*50}\n开始超参数搜索\n{'='*50}")
        searcher.fit(X, y, groups)
        
        # 保存结果
        self.best_params = searcher.best_params_
        self.best_score = searcher.best_score_
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(searcher.cv_results_)
        results_df.to_csv(os.path.join(self.save_path, 'tuning_results.csv'), index=False)
        
        print(f"\n{'='*50}\n超参数搜索完成\n{'='*50}")
        print(f"最佳得分: {self.best_score:.4f}")
        print(f"最佳参数: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def get_best_model(self):
        """获取使用最佳参数创建的模型
        
        Returns:
            配置了最佳参数的模型实例
        """
        if self.best_params is None:
            raise ValueError("必须先调用fit方法执行超参数搜索")
        
        return self.model_factory(**self.best_params)


class EnsembleBuilder:
    """集成模型构建器，用于创建和训练集成模型
    """
    
    def __init__(
        self,
        base_models: List[Callable],
        ensemble_method: str = 'voting',
        weights: Optional[List[float]] = None,
        save_path: str = 'results/ensemble'
    ):
        """初始化集成模型构建器
        
        Args:
            base_models: 基础模型工厂函数列表
            ensemble_method: 集成方法，支持'voting'、'stacking'和'bagging'
            weights: 模型权重，用于加权集成
            save_path: 结果保存路径
        """
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.models = []
    
    def build(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """构建集成模型
        
        Args:
            configs: 配置字典
            
        Returns:
            更新后的配置字典，包含集成模型
        """
        # 创建基础模型实例
        self.models = [model_factory() for model_factory in self.base_models]
        
        # 根据集成方法创建集成模型
        if self.ensemble_method == 'voting':
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            # 判断任务类型
            is_classifier = configs.get('task', {}).get('type', 'classification') == 'classification'
            
            # 创建相应的集成模型
            if is_classifier:
                ensemble_model = VotingClassifier(
                    estimators=[(f"model_{i}", model) for i, model in enumerate(self.models)],
                    voting='soft',
                    weights=self.weights
                )
            else:
                ensemble_model = VotingRegressor(
                    estimators=[(f"model_{i}", model) for i, model in enumerate(self.models)],
                    weights=self.weights
                )
        
        elif self.ensemble_method == 'stacking':
            from sklearn.ensemble import StackingClassifier, StackingRegressor
            
            # 判断任务类型
            is_classifier = configs.get('task', {}).get('type', 'classification') == 'classification'
            
            # 创建相应的集成模型
            if is_classifier:
                from sklearn.linear_model import LogisticRegression
                ensemble_model = StackingClassifier(
                    estimators=[(f"model_{i}", model) for i, model in enumerate(self.models)],
                    final_estimator=LogisticRegression()
                )
            else:
                from sklearn.linear_model import LinearRegression
                ensemble_model = StackingRegressor(
                    estimators=[(f"model_{i}", model) for i, model in enumerate(self.models)],
                    final_estimator=LinearRegression()
                )
        
        elif self.ensemble_method == 'bagging':
            from sklearn.ensemble import BaggingClassifier, BaggingRegressor
            
            # 判断任务类型
            is_classifier = configs.get('task', {}).get('type', 'classification') == 'classification'
            
            # 使用第一个模型作为基础模型
            if is_classifier:
                ensemble_model = BaggingClassifier(
                    base_estimator=self.models[0],
                    n_estimators=len(self.models),
                    random_state=42
                )
            else:
                ensemble_model = BaggingRegressor(
                    base_estimator=self.models[0],
                    n_estimators=len(self.models),
                    random_state=42
                )
        
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
        
        # 更新配置
        configs['model'] = ensemble_model
        
        return configs