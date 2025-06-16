import torch
import torch.nn as nn
from .H_01_Linear_cla import H_01_Linear_cla
from .H_02_distance_cla import H_02_distance_cla
from .H_03_Linear_pred import H_03_Linear_pred

class H_09_multiple_task(nn.Module):
    def __init__(self, args_m, num_classes_dict):
        super().__init__()
        self.args_m = args_m
        self.task_heads = nn.ModuleDict()

        # 根据配置初始化不同的任务头
        # 这里的 'classification' 和 'prediction' 是示例 Task_id
        # 您需要根据实际的 args_m.tasks 或类似配置来决定实例化哪些 head
        
        # 示例：如果配置中定义了分类任务
        if hasattr(args_m, 'classification_head') and args_m.classification_head == 'H_02_distance_cla':
            self.task_heads['classification'] = H_02_distance_cla(args_m, num_classes_dict)
        elif hasattr(args_m, 'classification_head') and args_m.classification_head == 'H_01_Linear_cla':
            self.task_heads['classification'] = H_01_Linear_cla(args_m, num_classes_dict)
        # else:
            # 可以选择抛出错误或使用默认的分类头
            # print("[H_09_multiple_task] Warning: No specific classification head configured, or H_02_distance_cla not specified.")


        # 示例：如果配置中定义了预测任务
        if hasattr(args_m, 'prediction_head') and args_m.prediction_head == 'H_03_Linear_pred':
            # H_03_Linear_pred 可能需要不同的参数，例如 P, in_dim 等，这些需要从 args_m 中获取
            # 例如: P = args_m.P, in_dim = args_m.output_dim
            # 这里我们假设 H_03_Linear_pred 的构造函数与分类头类似或可以从 args_m 获取所需参数
            # FlattenProjectHead 的参数: P, in_dim, hidden, max_len, max_out
            # 这些参数应该在配置文件中定义，并通过 args_m 传递
            P = getattr(args_m, 'P_for_pred_head', 128*256) # 示例值，需要配置
            in_dim = getattr(args_m, 'in_dim_for_pred_head', args_m.output_dim) # 示例值
            hidden = getattr(args_m, 'hidden_for_pred_head', 512)
            max_len = getattr(args_m, 'max_len_for_pred_head', 2048)
            max_out = getattr(args_m, 'max_out_for_pred_head', 512)
            self.task_heads['prediction'] = H_03_Linear_pred(P, in_dim, hidden, max_len, max_out)
        # else:
            # print("[H_09_multiple_task] Warning: No specific prediction head configured, or H_03_Linear_pred not specified.")

        if not self.task_heads:
            raise ValueError("[H_09_multiple_task] No task heads were initialized. Check your model configuration.")

    def forward(self, x, System_id=None, Task_id=None, return_feature=False, **kwargs):
        if Task_id is None:
            # 如果没有提供 Task_id，可能需要一个默认行为或抛出错误
            # 例如，默认执行第一个可用的 head 或在配置中指定默认任务
            if 'classification' in self.task_heads:
                Task_id = 'classification'
            elif 'prediction' in self.task_heads:
                Task_id = 'prediction'
            else:
                raise ValueError("Task_id is None and no default task head could be determined.")
            # print(f"[H_09_multiple_task] Warning: Task_id is None, defaulting to '{Task_id}'.")


        if Task_id not in self.task_heads:
            raise ValueError(f"Task_id '{Task_id}' not found in configured task_heads: {list(self.task_heads.keys())}")

        selected_head = self.task_heads[Task_id]

        if Task_id == 'classification':
            # H_01_Linear_cla 和 H_02_distance_cla 都期望 System_id
            return selected_head(x, System_id=System_id, Task_id=Task_id, return_feature=return_feature)
        elif Task_id == 'prediction':
            # H_03_Linear_pred (FlattenProjectHead) 期望 pred_len 和 out_dim
            # 这些参数应该从 kwargs 或 batch 中获取
            pred_len = kwargs.get('pred_len', self.args_m.pred_len_for_pred_head if hasattr(self.args_m, 'pred_len_for_pred_head') else x.size(1)) # 默认为输入序列长度
            out_dim = kwargs.get('out_dim', self.args_m.out_dim_for_pred_head if hasattr(self.args_m, 'out_dim_for_pred_head') else x.size(2))    # 默认为输入通道数
            
            if return_feature:
                # FlattenProjectHead 通常不直接返回 "feature" 像分类头那样，它直接输出预测结果
                # 如果需要中间特征，可能需要修改 H_03_Linear_pred
                print("[H_09_multiple_task] Warning: return_feature=True for prediction task might not behave as expected with H_03_Linear_pred.")
                # 暂时返回其 MLP 处理前的特征或其输出
                # 这里需要根据 H_03_Linear_pred 的具体实现来决定返回什么作为 "feature"
                # 假设 H_03_Linear_pred 内部有 .fc1 和 .act
                h_flat = x.reshape(x.size(0), -1)
                h_hidden = selected_head.act(selected_head.fc1(h_flat))
                return h_hidden # 返回 MLP 后的隐层特征

            return selected_head(x, pred_len=pred_len, out_dim=out_dim)
        else:
            # 对于其他自定义任务头，可能需要不同的参数传递方式
            return selected_head(x, System_id=System_id, Task_id=Task_id, return_feature=return_feature, **kwargs)

if __name__ == '__main__':
    from argparse import Namespace
    import pandas as pd

    # --- Mock Metadata ---
    class MockMetadata:
        def __init__(self):
            self.df = pd.DataFrame({
                'Dataset_id': ['dataset0', 'dataset0', 'dataset1', 'dataset1'],
                'Label': [0, 1, 0, 2]
            })
        
        def __getitem__(self, key): # key is File_id
            # 假设 File_id 0 和 1 属于 dataset0, File_id 2 和 3 属于 dataset1
            if key in [0,1]:
                return {'Sample_rate': 1000, 'Dataset_id': 'dataset0'}
            elif key in [2,3]:
                 return {'Sample_rate': 2000, 'Dataset_id': 'dataset1'}
            return {'Sample_rate': 1000, 'Dataset_id': 'unknown_dataset'}


    # --- Configuration ---
    args_m_cla = Namespace(
        output_dim=128,
        # 配置 H_09 使用哪个分类头
        classification_head='H_02_distance_cla', 
        # H_02_distance_cla 的参数 (如果它需要除了 output_dim 和 num_classes 之外的参数)
        # ... 
    )
    
    args_m_pred = Namespace(
        output_dim=128, # 这是 backbone 的输出维度，也是 H_03_Linear_pred 的 in_dim
        # 配置 H_09 使用哪个预测头
        prediction_head='H_03_Linear_pred',
        # H_03_Linear_pred (FlattenProjectHead) 的参数
        P_for_pred_head = 16 * 32, # 示例: num_patches * patch_len (对于 backbone 输出的序列长度)
        in_dim_for_pred_head = 128, # 应该等于 backbone 的 output_dim
        hidden_for_pred_head = 256,
        max_len_for_pred_head = 100,
        max_out_for_pred_head = 64,
        pred_len_for_pred_head = 50, # 运行时指定的 pred_len
        out_dim_for_pred_head = 3,   # 运行时指定的 out_dim
    )
    
    args_m_combined = Namespace(
        output_dim=128,
        classification_head='H_01_Linear_cla',
        prediction_head='H_03_Linear_pred',
        P_for_pred_head = 16 * 32, 
        in_dim_for_pred_head = 128,
        hidden_for_pred_head = 256,
        max_len_for_pred_head = 100,
        max_out_for_pred_head = 64,
        pred_len_for_pred_head = 50,
        out_dim_for_pred_head = 3,
    )


    mock_metadata = MockMetadata()
    num_classes_dict = {'dataset0': 2, 'dataset1': 3} # 从真实 metadata 获取

    # --- Test Classification Task Head ---
    print("--- Testing Classification Head (H_02_distance_cla) via H_09_multiple_task ---")
    multi_task_head_cla = H_09_multiple_task(args_m_cla, num_classes_dict)
    
    B, P_backbone, D_backbone = 4, 16, 128  # Backbone 输出: Batch, SeqLen, FeatureDim
    x_cla = torch.randn(B, P_backbone, D_backbone)
    
    # 模拟分类任务
    output_cla = multi_task_head_cla(x_cla, System_id='dataset0', Task_id='classification')
    print("Classification output shape:", output_cla.shape) # 期望 (B, num_classes_for_dataset0)
    assert output_cla.shape == (B, num_classes_dict['dataset0'])

    output_cla_feat = multi_task_head_cla(x_cla, System_id='dataset1', Task_id='classification', return_feature=True)
    print("Classification feature shape:", output_cla_feat.shape) # 期望 (B, 1, D_backbone) for H_02
    assert output_cla_feat.shape == (B, 1, D_backbone)


    # --- Test Prediction Task Head ---
    print("\n--- Testing Prediction Head (H_03_Linear_pred) via H_09_multiple_task ---")
    multi_task_head_pred = H_09_multiple_task(args_m_pred, num_classes_dict) # num_classes_dict 对预测头可能无用

    # H_03_Linear_pred 期望的输入是 (B, P, C) 其中 P 是扁平化前的序列长度
    # 假设 backbone 输出 (B, P_backbone, D_backbone) 
    # P_for_pred_head 应该是 P_backbone, in_dim_for_pred_head 应该是 D_backbone
    # 更新 args_m_pred 以匹配
    args_m_pred.P_for_pred_head = P_backbone 
    args_m_pred.in_dim_for_pred_head = D_backbone
    multi_task_head_pred_updated = H_09_multiple_task(args_m_pred, num_classes_dict)


    x_pred = torch.randn(B, P_backbone, D_backbone) # 输入到 H_03 的特征
    
    # 模拟预测任务
    # H_03_Linear_pred 的 forward 需要 pred_len 和 out_dim
    output_pred = multi_task_head_pred_updated(x_pred, Task_id='prediction', pred_len=50, out_dim=3)
    print("Prediction output shape:", output_pred.shape) # 期望 (B, pred_len, out_dim) -> (B, 50, 3)
    assert output_pred.shape == (B, 50, 3)

    output_pred_feat = multi_task_head_pred_updated(x_pred, Task_id='prediction', return_feature=True, pred_len=50, out_dim=3)
    print("Prediction feature shape (intermediate from H_03 via H_09):", output_pred_feat.shape) 
    assert output_pred_feat.shape == (B, args_m_pred.hidden_for_pred_head)


    # --- Test Combined Task Head ---
    print("\n--- Testing Combined Heads via H_09_multiple_task ---")
    args_m_combined.P_for_pred_head = P_backbone
    args_m_combined.in_dim_for_pred_head = D_backbone
    multi_task_head_combined = H_09_multiple_task(args_m_combined, num_classes_dict)

    # Classification
    output_c = multi_task_head_combined(x_cla, System_id='dataset0', Task_id='classification')
    print("Combined - Classification output shape:", output_c.shape)
    assert output_c.shape == (B, num_classes_dict['dataset0'])

    # Prediction
    output_p = multi_task_head_combined(x_pred, Task_id='prediction', pred_len=20, out_dim=5)
    print("Combined - Prediction output shape:", output_p.shape)
    assert output_p.shape == (B, 20, 5)
    
    print("\nAll tests passed for H_09_multiple_task!")

