import torch
import torch.nn as nn
from .H_01_Linear_cla import H_01_Linear_cla
from .H_02_distance_cla import H_02_distance_cla
from .H_03_Linear_pred import H_03_Linear_pred

class H_09_multiple_task(nn.Module):
    def __init__(self, args_m):
        super().__init__()
        self.args_m = args_m
        self.task_heads = nn.ModuleDict()

        # 根据配置初始化不同的任务头
        # 这里的 'classification' 和 'prediction' 是示例 task_id
        # 您需要根据实际的 args_m.task_list 来决定实例化哪些 head
        
        # 示例：如果配置中定义了分类任务
        if hasattr(args_m, 'classification_head'):
            if args_m.classification_head == 'H_02_distance_cla':
                self.task_heads['classification'] = H_02_distance_cla(args_m)
            elif args_m.classification_head == 'H_01_Linear_cla':
                self.task_heads['classification'] = H_01_Linear_cla(args_m)


        # 示例：如果配置中定义了预测任务
        if hasattr(args_m, 'prediction_head'):
            if args_m.prediction_head == 'H_03_Linear_pred':
                self.task_heads['prediction'] = H_03_Linear_pred(args_m)


    def forward(self, x,
                 system_id=None,
                   task_id=None,
                     return_feature=False,
                       **kwargs):
        if task_id is None:
            # 如果没有提供 task_id，可能需要一个默认行为或抛出错误
            # 例如，默认执行第一个可用的 head 或在配置中指定默认任务
            if 'classification' in self.task_heads:
                task_id = 'classification'
            elif 'prediction' in self.task_heads:
                task_id = 'prediction'
            else:
                raise ValueError("task_id is None and no default task head could be determined.")
            # print(f"[H_09_multiple_task] Warning: task_id is None, defaulting to '{task_id}'.")

        if task_id not in self.task_heads:
            raise ValueError(f"task_id '{task_id}' not found in configured task_heads: {list(self.task_heads.keys())}")

        selected_head = self.task_heads[task_id]
        if task_id == 'classification':
            # 如果是分类任务，可能需要传入 system_id
            if system_id is None:
                raise ValueError("system_id must be provided for classification tasks.")
            return selected_head(x,
                                system_id=system_id,
                                return_feature=return_feature,
                                **kwargs)
        elif task_id == 'prediction':
            # 如果是预测任务，可能不需要 system_id
            return selected_head(x,
                                **kwargs)

if __name__ == '__main__':
    from argparse import Namespace
    import pandas as pd
    args = Namespace(
        output_dim=64,
        classification_head='H_02_distance_cla', # H_01_Linear_cla
        prediction_head='H_03_Linear_pred',
        num_classes={'system1': 10, 'system2': 5},
        num_patches=128,
        d_model=64,
        patch_size_C=3,
        patch_size_L=128,
    )
    metadata = pd.DataFrame({
        'Dataset_id': [1, 2],
        'Sample_rate': [16000, 16000]
    }, index=[0, 1])

    model = H_09_multiple_task(args)
    x = torch.randn(2, 128, 64)  # 假设输入特征维度为128
    output = model(x, system_id='system1', task_id='classification')
    print(output.shape)  # 输出形状应与任务头的输出一致
    # 如果需要返回特征，可以设置 return_feature=True
    output_feature = model(x, system_id='system1', task_id='classification', return_feature=True)
    print(output_feature.shape)  # 输出特征的形状
    # 测试预测任务
    output_pred = model(x, system_id='system1', task_id='prediction', shape=(128, 3))
    print(output_pred.shape)  # 输出预测任务的形状

