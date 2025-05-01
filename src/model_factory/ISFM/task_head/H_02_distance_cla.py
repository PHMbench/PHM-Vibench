import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        """
        CrossAttention模块实现了一个基础的交叉注意力机制。
        
        参数:
            dim: 输入的特征维度。
            num_heads: 注意力头的数量。
            qkv_bias: 是否为Q、K、V设置偏置。
            qk_norm: 是否对Q和K进行归一化。
            attn_drop: 注意力机制的dropout概率。
            proj_drop: 输出的dropout概率。
            norm_layer: 归一化层，默认是LayerNorm。
            var_num: 变量数量，用于模板初始化。
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        # 配置参数
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子

        # 定义各个线性变换
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 归一化层
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 模板初始化（如果var_num不为None）
        if var_num is not None:
            self.template = nn.Parameter(torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)

        self.var_num = var_num

    def forward(self, x, query=None):
        """
        前向传播。

        参数:
            x: 输入张量，形状为 (B, N, C)，B是批量大小，N是序列长度，C是特征维度。
            query: 可选的查询张量，形状为 (B, M, C)，其中M是查询的序列长度。
        
        返回:
            输出张量，形状为 (B, var_num, C)。
        """
        B, N, C = x.shape
        
        # 如果提供了query，则使用它；否则，使用模板初始化
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num

        # 对x进行线性变换，得到K和V
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        # 计算注意力
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        # 重塑输出，返回最终的结果
        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class H_02_distance_cla(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        """
        CLSHead模块，主要用于计算分类任务中的分类头。

        参数:
            d_model: 输入的特征维度。
            head_dropout: 分类头的dropout概率。
        """
        super().__init__()
        d_mid = d_model

        # 输入映射和交叉注意力层
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        # 分类头的MLP层
        self.mlp = nn.Linear(d_mid, d_model)

    def forward(self, x, category_token=None, return_feature=False):
        """
        前向传播。

        参数:
            x: 输入张量，形状为 (B, P, D)，B是批量大小，P是序列长度，D是特征维度。
            category_token: 类别token，形状为 (B, C, D_mid)，C是类别数，D_mid是映射后的特征维度。
            return_feature: 是否返回处理后的特征。

        返回:
            输出张量，形状为 (B, C)。
        """
        # 输入通过线性映射
        x = self.proj_in(x)  # B, P, D -> B, P, D_mid
        B, P, D_mid = x.shape  # 形状为 B, P, D_mid
        x = x.view(-1, P, D_mid)  # 展平 B, P, D_mid

        # 提取cls_token（取最后一个作为cls_token）
        cls_token = x.mean(dim = 1, keepdim = True)  # 取最后一个作为 cls_token: B, 1, D_mid

        # 使用交叉注意力计算cls_token
        cls_token = self.cross_att(x, query=cls_token)  # 交叉注意力计算
        cls_token = cls_token.reshape(B, 1, D_mid)  # 还原形状

        # 通过MLP层进行处理
        cls_token = self.mlp(cls_token)  # MLP 处理

        if return_feature:
            return cls_token

        # 计算cls_token与category_token之间的距离
        C = category_token.shape[1]  # 类别数 C
        distance = torch.einsum('bkc,bmc->bm', cls_token, category_token)  # 计算距离

        # 求均值，得到最终结果
        # distance = distance.mean(dim=1)  
        return distance


if __name__ == '__main__':
    # 测试用例
    B, P, D = 4, 10, 128  # B: 批量大小, P: 序列长度, D: 特征维度
    C = 40  # 类别数

    # 创建输入数据
    x = torch.randn(B, P, D)  # B, P, D 维度的输入信号
    category_token = torch.randn(B, C, D)  # B, C, D 维度的类别 token

    # 初始化 CLSHead
    cls_head = H_02_distance_cla(d_model=D)

    # 进行前向推理
    output = cls_head(x, category_token=category_token)

    print("Output:", output.shape)  # 输出形状
