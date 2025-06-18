# TODO using IDselector to extend other ID selector
import random
from sklearn.model_selection import train_test_split

class IDSelector:
    """数据ID选择器基类
    
    负责根据不同的选择策略和任务要求，选择适当的训练/测试ID
    """
    
    def __init__(self, metadata, args_task):
        """
        初始化ID选择器
        
        参数:
            metadata: 元数据对象
            args_task: 任务配置参数
        """
        self.metadata = metadata
        self.args_task = args_task
        self.train_val_ids = []
        self.test_ids = []
        self._split_cache = None
        
    def select(self):
        """
        执行ID选择，确定训练/验证和测试数据集的ID
        
        返回:
            tuple: (train_val_ids, test_ids) 训练/验证ID列表和测试ID列表
        """
        # 检查缓存
        cache_id = self._get_cache_id()
        if self._split_cache and self._split_cache.get('id') == cache_id:
            print(f"使用缓存的ID选择结果")
            return self._split_cache['train_val_ids'], self._split_cache['test_ids']
        
        # 如果未指定目标数据集，返回所有ID
        if not hasattr(self.args_task, 'target_system_id') or self.args_task.target_system_id is None:
            self.train_val_ids = list(self.metadata.keys())
            self.test_ids = list(self.metadata.keys())
            print(f"未指定目标数据集ID，使用全部 {len(self.train_val_ids)} 个样本")
            return self.train_val_ids, self.test_ids
        
        # 选择ID
        self._select_ids()
        
        # 打印选择统计信息
        self._print_stats()
        
        # 缓存结果
        self._split_cache = {
            'id': cache_id,
            'train_val_ids': self.train_val_ids,
            'test_ids': self.test_ids
        }
        
        return self.train_val_ids, self.test_ids
    
    def _get_cache_id(self):
        """生成缓存ID"""
        import hashlib
        # 将关键参数转换为字符串并哈希
        params = {
            'target_system_id': getattr(self.args_task, 'target_system_id', None),
            'type': getattr(self.args_task, 'type', None),
            'selector': getattr(self.args_task, 'selector', 'default'),
            'seed': getattr(self.args_task, 'seed', 42)
        }
        
        # 添加子类特定参数
        extra_params = self._get_extra_params()
        params.update(extra_params)
        
        return hashlib.md5(str(params).encode()).hexdigest()
    
    def _get_extra_params(self):
        """获取子类特定的缓存参数，由子类实现"""
        return {}
    
    def _select_ids(self):
        """执行ID选择，由子类实现"""
        raise NotImplementedError("子类必须实现_select_ids方法")
    
    def _print_stats(self):
        """打印ID选择统计信息"""
        if not hasattr(self, 'train_val_ids') or not hasattr(self, 'test_ids'):
            return
        
        # 统计每个数据集的样本数量
        train_stats = self._get_dataset_stats(self.train_val_ids)
        test_stats = self._get_dataset_stats(self.test_ids)
        
        print("\n数据选择结果统计:")
        print(f"总训练/验证样本数: {len(self.train_val_ids)}")
        print(f"总测试样本数: {len(self.test_ids)}")
        
        if train_stats:
            print("\n各数据集样本分布:")
            for dataset_id in sorted(set(list(train_stats.keys()) + list(test_stats.keys()))):
                train_count = train_stats.get(dataset_id, {'count': 0})['count']
                test_count = test_stats.get(dataset_id, {'count': 0})['count']
                total = train_count + test_count
                print(f"  数据集 {dataset_id}: 训练={train_count} ({train_count/total:.1%}), 测试={test_count} ({test_count/total:.1%})")
    
    def _get_dataset_stats(self, id_list):
        """获取指定ID列表的数据集统计信息"""
        stats = {}
        for id in id_list:
            try:
                dataset_id = self.metadata.df.loc[id, 'Dataset_id']
                if dataset_id not in stats:
                    stats[dataset_id] = {'count': 0, 'labels': {}}
                
                stats[dataset_id]['count'] += 1
                
                # 如果有标签信息，统计标签分布
                if 'Label' in self.metadata.df.columns:
                    label = self.metadata.df.loc[id, 'Label']
                    if label not in stats[dataset_id]['labels']:
                        stats[dataset_id]['labels'][label] = 0
                    stats[dataset_id]['labels'][label] += 1
            except:
                # 处理ID不存在的情况
                continue
                
        return stats
    
    def visualize(self, save_path=None):
        """可视化ID选择结果"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            import pandas as pd
            
            # 准备数据
            data = []
            
            # 处理所有ID
            for id in self.train_val_ids:
                try:
                    row = self.metadata.df.loc[id]
                    data.append({
                        'Dataset': row['Dataset_id'],
                        'Domain': row['Domain_id'] if 'Domain_id' in row else 'Unknown',
                        'Split': 'Train',
                        'Label': row['Label'] if 'Label' in row else 'Unknown'
                    })
                except:
                    continue
            
            for id in self.test_ids:
                try:
                    row = self.metadata.df.loc[id]
                    data.append({
                        'Dataset': row['Dataset_id'],
                        'Domain': row['Domain_id'] if 'Domain_id' in row else 'Unknown',
                        'Split': 'Test',
                        'Label': row['Label'] if 'Label' in row else 'Unknown'
                    })
                except:
                    continue
            
            if not data:
                print("没有足够的数据进行可视化")
                return
                
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 创建可视化图表
            plt.figure(figsize=(12, 8))
            sns.countplot(data=df, x='Dataset', hue='Split', palette='Set2')
            plt.title('数据集分布')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_datasets.png")
                plt.close()
            else:
                plt.show()
            
            # 如果有标签信息，绘制标签分布
            if 'Label' in df.columns and df['Label'].nunique() > 1:
                plt.figure(figsize=(12, 8))
                sns.countplot(data=df, x='Label', hue='Split', palette='Set3')
                plt.title('标签分布')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(f"{save_path}_labels.png")
                    plt.close()
                else:
                    plt.show()
                    
        except Exception as e:
            print(f"可视化数据时出错: {e}")


class DGIDSelector(IDSelector):
    """领域泛化ID选择器"""
    
    def _get_extra_params(self):
        """获取领域泛化特定参数"""
        return {
            'source_domain_id': getattr(self.args_task, 'source_domain_id', None),
            'target_domain_id': getattr(self.args_task, 'target_domain_id', None),
            'target_domain_num': getattr(self.args_task, 'target_domain_num', 1)
        }
    
    def _select_ids(self):
        """领域泛化ID选择"""
        task_type = getattr(self.args_task, 'type', 'DG')
        
        if task_type == 'DG':
            self._select_dg()
        elif task_type == 'CDDG':
            self._select_cddg()
        elif task_type == 'CDGD':
            self._select_cdgd()
        else:
            print(f"未知的任务类型: {task_type}，使用默认DG选择")
            self._select_dg()
    
    def _select_dg(self):
        """标准领域泛化选择"""
        # 过滤数据集
        filtered_df = self.metadata.df[
            self.metadata.df['Dataset_id'].isin(self.args_task.target_system_id)]
        
        # 训练域
        train_df = filtered_df[
            filtered_df['Domain_id'].isin(self.args_task.source_domain_id)]
        
        # 测试域
        test_df = filtered_df[
            filtered_df['Domain_id'].isin(self.args_task.target_domain_id)]
        
        self.train_val_ids = list(train_df['Id'])
        self.test_ids = list(test_df['Id'])
        
        print(f"DG选择 - 源域: {self.args_task.source_domain_id}, 目标域: {self.args_task.target_domain_id}")
    
    def _select_cddg(self):
        """跨数据集领域泛化选择"""
        # 筛选出目标数据集
        filtered_df = self.metadata.df[
            self.metadata.df['Dataset_id'].isin(self.args_task.target_system_id)]
        
        # 找出每个数据集的域
        dataset_domains = {}
        for dataset_id in self.args_task.target_system_id:
            dataset_df = filtered_df[filtered_df['Dataset_id'] == dataset_id]
            domains = sorted(dataset_df['Domain_id'].unique())
            domains = [d for d in domains if not pd.isna(d)]
            dataset_domains[dataset_id] = domains
        
        # 为每个数据集选择训练和测试域
        train_domains = {}
        test_domains = {}
        for dataset_id, domains in dataset_domains.items():
            test_count = min(self.args_task.target_domain_num, len(domains))
            train_domains[dataset_id] = domains[:-test_count] if test_count > 0 else domains
            test_domains[dataset_id] = domains[-test_count:] if test_count > 0 else []
        
        # 收集ID
        train_rows = []
        test_rows = []
        for dataset_id in self.args_task.target_system_id:
            # 训练集
            for domain_id in train_domains[dataset_id]:
                train_rows.extend(
                    filtered_df[(filtered_df['Dataset_id'] == dataset_id) & 
                             (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
                )
            # 测试集
            for domain_id in test_domains[dataset_id]:
                test_rows.extend(
                    filtered_df[(filtered_df['Dataset_id'] == dataset_id) & 
                             (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
                )
        
        self.train_val_ids = train_rows
        self.test_ids = test_rows
        
        # 打印信息
        print(f"CDDG选择 - 每个数据集使用最后{self.args_task.target_domain_num}个域作为测试")
        for dataset_id in self.args_task.target_system_id:
            print(f"  数据集 {dataset_id}:")
            print(f"    训练域: {train_domains[dataset_id]}")
            print(f"    测试域: {test_domains[dataset_id]}")
    
    def _select_cdgd(self):
        """自定义领域泛化选择"""
        # 类似于_select_cddg的实现...
        filtered_df = self.metadata.df[
            self.metadata.df['Dataset_id'].isin(self.args_task.target_system_id)]
        
        # 构建分组和ID选择...
        group_by = getattr(self.args_task, 'group_by', 'Domain_id')
        print(f"CDGD选择 - 按 {group_by} 分组")
        
        train_rows = []
        test_rows = []
        # 实现分组选择逻辑...
        
        self.train_val_ids = train_rows
        self.test_ids = test_rows


class FewShotIDSelector(IDSelector):
    """小样本学习ID选择器"""
    
    def _get_extra_params(self):
        """获取小样本学习特定参数"""
        return {
            'n_way': getattr(self.args_task, 'n_way', 5),
            'k_shot': getattr(self.args_task, 'k_shot', 1),
            'n_query': getattr(self.args_task, 'n_query', 15),
            'test_ratio': getattr(self.args_task, 'test_ratio', 0.2),
            'balanced': getattr(self.args_task, 'balanced', True)
        }
    
    def _select_ids(self):
        """小样本学习ID选择"""
        # 过滤数据集
        filtered_df = self.metadata.df[
            self.metadata.df['Dataset_id'].isin(self.args_task.target_system_id)]
        
        # 设置随机种子
        random.seed(getattr(self.args_task, 'seed', 42))
        
        # 检查标签列
        label_col = getattr(self.args_task, 'label_column', 'Label')
        if label_col not in filtered_df.columns:
            print(f"警告: 找不到标签列 '{label_col}'，无法执行小样本采样")
            # 回退到随机选择
            self._random_select(filtered_df)
            return
        
        # 获取所有可用类别
        all_classes = sorted(filtered_df[label_col].unique())
        all_classes = [c for c in all_classes if not pd.isna(c)]
        
        # n-way k-shot设置
        n_way = min(getattr(self.args_task, 'n_way', 5), len(all_classes))
        k_shot = getattr(self.args_task, 'k_shot', 1)
        n_query = getattr(self.args_task, 'n_query', 15)
        
        # 随机选择n_way个类别
        selected_classes = random.sample(all_classes, n_way)
        
        print(f"Few-Shot采样 - {n_way}-way {k_shot}-shot")
        print(f"  选择的类别: {selected_classes}")
        
        # 收集支持集和查询集样本
        support_ids = []
        query_ids = []
        
        for cls in selected_classes:
            # 获取当前类别的所有样本
            cls_samples = filtered_df[filtered_df[label_col] == cls]['Id'].tolist()
            
            if len(cls_samples) <= k_shot:
                # 类别样本不足
                print(f"  警告: 类别 {cls} 的样本数量 ({len(cls_samples)}) 小于 k_shot ({k_shot})")
                support_ids.extend(cls_samples)
                continue
                
            # 随机选择k_shot个样本作为支持集
            cls_support = random.sample(cls_samples, k_shot)
            support_ids.extend(cls_support)
            
            # 剩余样本作为查询集
            cls_query = [s for s in cls_samples if s not in cls_support]
            # 如果指定了查询样本数量，随机选择n_query个
            if n_query > 0 and len(cls_query) > n_query:
                cls_query = random.sample(cls_query, n_query)
            
            query_ids.extend(cls_query)
        
        self.train_val_ids = support_ids
        self.test_ids = query_ids
        
        print(f"  支持集样本数: {len(support_ids)}")
        print(f"  查询集样本数: {len(query_ids)}")
    
    def _random_select(self, df):
        """随机选择样本"""
        all_ids = df['Id'].tolist()
        random.shuffle(all_ids)
        
        test_ratio = getattr(self.args_task, 'test_ratio', 0.2)
        test_size = int(len(all_ids) * test_ratio)
        
        self.test_ids = all_ids[:test_size]
        self.train_val_ids = all_ids[test_size:]


class ImbalancedIDSelector(IDSelector):
    """不平衡数据ID选择器"""
    
    def _get_extra_params(self):
        """获取不平衡数据特定参数"""
        return {
            'imbalance_ratio': getattr(self.args_task, 'imbalance_ratio', 0.1),
            'minority_labels': getattr(self.args_task, 'minority_labels', None),
            'test_ratio': getattr(self.args_task, 'test_ratio', 0.2),
            'stratify': getattr(self.args_task, 'stratify', True)
        }
    
    def _select_ids(self):
        """不平衡数据ID选择"""
        # 过滤数据集
        filtered_df = self.metadata.df[
            self.metadata.df['Dataset_id'].isin(self.args_task.target_system_id)]
        
        # 检查标签列
        label_col = getattr(self.args_task, 'label_column', 'Label')
        if label_col not in filtered_df.columns:
            print(f"警告: 找不到标签列 '{label_col}'，无法执行不平衡采样")
            # 回退到随机选择
            self._random_select(filtered_df)
            return
        
        # 获取所有可用类别及其样本数
        class_counts = filtered_df[label_col].value_counts().to_dict()
        
        # 确定多数类和少数类
        if hasattr(self.args_task, 'minority_labels') and self.args_task.minority_labels:
            # 明确指定少数类
            minority_labels = self.args_task.minority_labels
            majority_labels = [l for l in class_counts.keys() if l not in minority_labels]
        else:
            # 按样本数量自动确定
            median_count = np.median(list(class_counts.values()))
            minority_labels = [l for l, c in class_counts.items() if c < median_count]
            majority_labels = [l for l, c in class_counts.items() if c >= median_count]
        
        # 设置不平衡比例
        imbalance_ratio = getattr(self.args_task, 'imbalance_ratio', 0.1)
        
        print(f"不平衡数据采样 - 比例: {imbalance_ratio}")
        print(f"  多数类: {majority_labels}")
        print(f"  少数类: {minority_labels}")
        
        # 收集每个类别的样本
        label_ids = {label: filtered_df[filtered_df[label_col] == label]['Id'].tolist() 
                    for label in class_counts.keys()}
        
        # 设置随机种子
        random.seed(getattr(self.args_task, 'seed', 42))
        
        # 计算不平衡采样
        majority_samples = []
        minority_samples = []
        
        # 对每个多数类采样
        for label in majority_labels:
            majority_samples.extend(label_ids[label])
        
        # 对每个少数类采样
        target_minority_count = int(len(majority_samples) * imbalance_ratio)
        minority_count_per_class = target_minority_count // len(minority_labels) if minority_labels else 0
        
        for label in minority_labels:
            available = label_ids[label]
            if len(available) <= minority_count_per_class:
                # 如果样本不足，全部使用
                minority_samples.extend(available)
            else:
                # 随机采样
                sampled = random.sample(available, minority_count_per_class)
                minority_samples.extend(sampled)
        
        # 合并所有样本
        all_samples = majority_samples + minority_samples
        
        # 划分训练集和测试集
        test_ratio = getattr(self.args_task, 'test_ratio', 0.2)
        stratify = getattr(self.args_task, 'stratify', True)
        
        if stratify:
            # 使用分层抽样
            sample_labels = [filtered_df.loc[id, label_col] for id in all_samples]
            train_ids, test_ids = train_test_split(
                all_samples, test_size=test_ratio, 
                stratify=sample_labels,
                random_state=getattr(self.args_task, 'seed', 42)
            )
        else:
            # 随机抽样
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * (1 - test_ratio))
            train_ids = all_samples[:split_idx]
            test_ids = all_samples[split_idx:]
        
        self.train_val_ids = train_ids
        self.test_ids = test_ids
        
        # 打印统计信息
        print(f"  训练集样本数: {len(train_ids)}")
        print(f"  测试集样本数: {len(test_ids)}")
    
    def _random_select(self, df):
        """随机选择样本"""
        all_ids = df['Id'].tolist()
        random.shuffle(all_ids)
        
        test_ratio = getattr(self.args_task, 'test_ratio', 0.2)
        test_size = int(len(all_ids) * test_ratio)
        
        self.test_ids = all_ids[:test_size]
        self.train_val_ids = all_ids[test_size:]