import torch
import torch.nn as nn
import math


class FlowLoss(nn.Module):
    """
    Flow Loss 模块，用于训练一个基于流的生成模型。
    这个模型学习从噪声中恢复目标数据，条件是某个隐变量 z。
    """

    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps):
        """
        初始化 FlowLoss 模块。
        :param target_channels: 目标数据的通道数（或特征维度）。
        :param z_channels: 条件隐变量 z 的通道数。
        :param depth: 神经网络的深度（残差块的数量）。
        :param width: 神经网络的宽度（模型通道数）。
        :param num_sampling_steps: 采样时的步数。
        """
        super(FlowLoss, self).__init__()
        self.in_channels = target_channels
        # 初始化核心网络 SimpleMLPAdaLN
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth
        )
        self.num_sampling_steps = num_sampling_steps

    def forward(self, target, z, mask=None, mask_y=None):
        """
        前向传播，计算损失。
        :param target: 目标数据，形状为 [batch_size, target_channels]。
        :param z: 条件隐变量，形状为 [batch_size, z_channels]。
        :param mask: 样本掩码，用于忽略某些样本的损失。
        :param mask_y: 特征掩码，用于忽略某些特征的损失。
        :return: 标量损失值。
        """
        # 1. 生成与目标数据同形的标准正态分布噪声
        noise = torch.randn_like(target)
        # 2. 为每个样本生成一个随机时间步 t，范围在 [0, 1)
        t = torch.rand(target.shape[0], device=target.device)

        # 3. 构造加噪数据，通过 t 在目标和噪声之间进行线性插值
        noised_target = t[:, None] * target + (1 - t[:, None]) * noise

        # 4. 将加噪数据、时间步 t 和条件 z 输入网络，预测速度 v (v = target - noise)
        #    时间步 t 被乘以 1000 以匹配 DiT 等模型中的常见做法
        predict_v = self.net(noised_target, t * 1000, z)

        # 5. 计算损失
        #    为不同通道/特征设置权重，通道索引越大权重越小
        weights = 1.0 / \
            torch.arange(1, self.in_channels + 1, dtype=torch.float32, device=target.device)
        
        #    计算加权均方误差
        if mask_y is not None:
            # 如果有特征掩码，则只计算未被掩码的特征的损失
            loss = (mask_y * weights * (predict_v - target) ** 2).sum(dim=-1)
        else:
            loss = (weights * (predict_v - target) ** 2).sum(dim=-1)

        # 6. 如果有样本掩码，则应用掩码并计算有效样本的平均损失
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, num_samples=1):
        """
        从模型中采样生成数据。
        使用类似于 DDIM 的采样方法，从纯噪声开始，逐步去噪。
        :param z: 条件隐变量。
        :param num_samples: 每个条件 z 要生成的样本数量。
        :return: 生成的样本。
        """
        # 将条件 z 重复 num_samples 次，为每个样本生成提供条件
        z = z.repeat(num_samples, 1)
        # 从标准正态分布中采样初始噪声
        noise = torch.randn(z.shape[0], self.in_channels).to(z.device)
        x = noise
        # 计算每一步的时间步长
        dt = 1.0 / self.num_sampling_steps
        # 迭代去噪
        for i in range(self.num_sampling_steps):
            # 当前时间步 t
            t = (torch.ones((x.shape[0])) * i /
                 self.num_sampling_steps).to(x.device)
            # 使用网络预测速度 v
            pred = self.net(x, t * 1000, z)
            # 更新 x，向预测的目标方向移动一小步
            # x_t = x_{t-1} + (v - noise) * dt
            x = x + (pred - noise) * dt
        # 调整输出形状
        x = x.reshape(num_samples, -1, self.in_channels).transpose(0, 1)
        return x


def modulate(x, shift, scale):
    """
    仿射变换函数，用于 AdaLN (Adaptive Layer Normalization)。
    对输入 x 进行缩放 (scale) 和平移 (shift)。
    :param x: 输入张量。
    :param shift: 平移向量。
    :param scale: 缩放向量。
    :return: 变换后的张量。
    """
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    将标量时间步 t 嵌入为向量表示。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        :param hidden_size: 输出嵌入向量的维度。
        :param frequency_embedding_size: 频率嵌入的维度（正弦嵌入的维度）。
        """
        super().__init__()
        # 一个简单的多层感知机 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入，类似于 Transformer 中的位置编码。
        :param t: 一维张量，包含 N 个时间步索引。
        :param dim: 输出嵌入的维度。
        :param max_period: 控制嵌入的最小频率。
        :return: (N, D) 形状的位置嵌入张量。
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # 计算不同频率
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # 计算参数
        args = t[:, None].float() * freqs[None]
        # 使用 sin 和 cos 函数创建嵌入
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果维度是奇数，则填充一个零
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        前向传播，将时间步 t 转换为嵌入向量。
        """
        # 首先创建正弦嵌入
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 然后通过 MLP 进行变换
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    一个残差块，使用了自适应层归一化 (adaLN)。
    :param channels: 输入和输出通道数。
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        # 输入的层归一化
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        # 核心的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        # adaLN 的调制网络，从条件 y 生成 shift, scale, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True) # 输出3倍通道数，分别用于 shift, scale, gate
        )

    def forward(self, x, y):
        """
        :param x: 输入数据。
        :param y: 条件向量 (通常是时间嵌入和条件嵌入的和)。
        """
        # 从条件 y 生成调制参数
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y).chunk(3, dim=-1)
        # 对归一化后的 x 进行仿射变换 (modulate)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        # 通过 MLP
        h = self.mlp(h)
        # 应用残差连接，并用 gate 进行缩放
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    模型的最后一层，借鉴自 DiT (Diffusion Transformer)。
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        # 最终的层归一化，不带可学习的仿射参数
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        # 线性层，将特征映射到输出通道数
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        # adaLN 的调制网络，生成 shift 和 scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        """
        :param x: 输入特征。
        :param c: 条件向量。
        """
        # 从条件 c 生成调制参数
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # 对归一化后的 x 进行仿射变换
        x = modulate(self.norm_final(x), shift, scale)
        # 通过最后的线性层
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    使用 AdaLN 的简单 MLP 网络，是 FlowLoss 的核心。
    :param in_channels: 输入张量的通道数。
    :param model_channels: 模型的基础通道数。
    :param out_channels: 输出张量的通道数。
    :param z_channels: 条件 z 的通道数。
    :param num_res_blocks: 残差块的数量。
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # 时间嵌入模块
        self.time_embed = TimestepEmbedder(model_channels)
        # 条件 z 的嵌入层
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # 输入投影层，将输入 x 映射到模型通道数
        self.input_proj = nn.Linear(in_channels, model_channels)

        # 创建一系列残差块
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        # 最终输出层
        self.final_layer = FinalLayer(model_channels, out_channels)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """
        初始化模型权重。这对于训练的稳定性很重要。
        """
        # 基本初始化：对线性层使用 Xavier 均匀分布初始化，偏置为0
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 特殊初始化：
        # 初始化时间嵌入 MLP 的权重为正态分布
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # 将 adaLN 调制层的输出权重和偏置初始化为 0
        # 这样在训练初期，调制作用很小，模型接近一个标准的残差网络
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 将最终层的调制和线性输出权重和偏置初始化为 0
        # 这样在训练初期，模型的输出接近于0，有助于稳定训练
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        模型的前向传播。
        :param x: 输入张量 [N x C_in]。
        :param t: 一维时间步张量 [N]。
        :param c: 条件张量 [N x C_z]。
        :return: 输出张量 [N x C_out]。
        """
        # 1. 将输入 x 投影到模型维度
        x = self.input_proj(x)
        # 2. 将时间步 t 嵌入为向量
        t = self.time_embed(t)
        # 3. 将条件 c 嵌入为向量
        c = self.cond_embed(c)
        # 4. 将时间和条件嵌入相加，作为残差块的条件输入 y
        y = t + c

        # 5. 依次通过所有残差块
        for block in self.res_blocks:
            x = block(x, y)

        # 6. 通过最终层得到输出
        return self.final_layer(x, y)


if __name__ == "__main__":
    # 1. 设置模型参数
    target_channels = 64  # 目标数据的特征维度
    z_channels = 128      # 条件 z 的特征维度
    depth = 4             # 网络的深度 (残差块数量)
    width = 256           # 网络的宽度 (模型通道数)
    num_sampling_steps = 10 # 采样步数

    # 2. 实例化 FlowLoss 模型
    model = FlowLoss(
        target_channels=target_channels,
        z_channels=z_channels,
        depth=depth,
        width=width,
        num_sampling_steps=num_sampling_steps
    )
    print("模型实例化成功。")
    # print(model) # 取消注释以查看模型结构

    # 3. 准备虚拟输入数据
    batch_size = 4
    # 目标数据 (例如，时间序列的某个窗口)
    target = torch.randn(batch_size, target_channels)
    # 条件向量 (例如，从自回归模型获得的上下文表示)
    z = torch.randn(batch_size, z_channels)
    
    print(f"\n准备虚拟数据:")
    print(f"  - 目标数据 (target) 形状: {target.shape}")
    print(f"  - 条件向量 (z) 形状: {z.shape}")

    # 4. 测试前向传播 (计算损失)
    print("\n--- 测试训练步骤 (前向传播) ---")
    # 将模型置于训练模式
    model.train()
    loss = model(target, z)
    print(f"计算出的损失值: {loss.item():.4f}")
    # 在实际训练中，接下来会执行 loss.backward() 和 optimizer.step()

    # 5. 测试采样 (生成新数据)
    print("\n--- 测试推理步骤 (采样) ---")
    # 假设我们有一个新的条件向量来生成样本
    new_z = torch.randn(1, z_channels) # 为1个上下文生成
    num_samples = 5 # 生成5个样本
    
    # 将模型置于评估模式
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(new_z, num_samples=num_samples)
    
    print(f"为1个条件向量生成了 {num_samples} 个样本。")
    # model.sample 的输出形状是 [context_len, num_samples, channels]
    # 因为 new_z 的第一维是 1 (context_len=1)，所以输出形状是 [1, 5, 64]
    print(f"生成样本的形状: {generated_samples.shape}")
    print("演示完成。")
