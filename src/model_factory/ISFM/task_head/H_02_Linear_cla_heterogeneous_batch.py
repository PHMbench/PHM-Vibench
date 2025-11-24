import torch.nn as nn

from src.model_factory.ISFM.system_utils import group_forward_by_system


class H_02_Linear_cla_heterogeneous_batch(nn.Module):
    """
    H_02: 适用于“异构 batch”（一个 batch 内混合多个系统）的多头线性分类器。

    约束：
    - 所有系统的 num_classes 必须一致（共用统一标签空间），否则输出维度无法对齐。

    使用场景：
    - M_02_heterogeneous_batch 等支持多系统 batch 的模型；
    - CDDG / 对比预训练中按样本传入 system_id，由本 head 自动按系统分组 forward。
    """

    def __init__(self, args):
        super().__init__()
        self.heads = nn.ModuleDict()
        num_classes = args.num_classes

        # 构建 per-system 线性 head
        out_dims = set()
        for data_name, n_class in num_classes.items():
            key = str(data_name)
            head = nn.Linear(args.output_dim, n_class)
            self.heads[key] = head
            out_dims.add(int(n_class))

        if len(out_dims) != 1:
            raise ValueError(
                f"H_02_Linear_cla_heterogeneous_batch requires unified num_classes across systems, "
                f"but got: {out_dims}"
            )

    def forward(self, x, system_id=None, return_feature=False, **kwargs):
        """
        x: [B, T, D] or [B, D]
        system_id: 标量 / list / tensor，表示每个样本的 Dataset_id。

        内部通过 group_forward_by_system:
        - 对每个 system_id 分组子 batch；
        - 调用对应 head；
        - 将结果写回 logits 的对应位置。
        """
        return group_forward_by_system(x, system_id, self.heads)


if __name__ == "__main__":
    # Minimal self-test for heterogeneous batch head
    import torch
    from argparse import Namespace

    args = Namespace(
        output_dim=8,
        num_classes={"1": 3, "6": 3},  # two systems, same num_classes
    )

    head = H_02_Linear_cla_heterogeneous_batch(args)
    x = torch.randn(4, 5, 8)  # [B=4, T=5, D=8]
    system_ids = [1, 1, 6, 6]

    logits = head(x, system_id=system_ids)
    assert logits.shape == (4, 3)
    print("✓ H_02_Linear_cla_heterogeneous_batch self-test passed:", logits.shape)
