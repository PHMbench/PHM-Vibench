import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

class E_com_00_PE(nn.Module):
    def __init__(self, args,args_d = False):
        super().__init__()
        self.embed_type = args.embed_type
        self.d_model = args.output_dim
        
        # 从args获取参数
        self.patch_len = args.patch_size_L if hasattr(args, 'patch_len') else 256
        self.stride = args.stride if hasattr(args, 'stride') else 128
        self.padding = args.padding if hasattr(args, 'padding') else 0
        self.dropout_rate = args.dropout
        self.freq = args.freq if hasattr(args, 'freq') else 'h'
        self.embed_type_sub = args.sub_embed_type if hasattr(args, 'sub_embed_type') else 'fixed'
        self.c_in = args.patch_size_C

        # 组件初始化
        if self.embed_type == 'positional':
            self.module = PositionalEmbedding(self.d_model, max_len=5000)
            
        elif self.embed_type == 'token':
            self.module = TokenEmbedding(self.c_in, self.d_model)
        # elif self.embed_type == 'temporal':
        #     self.module = TemporalEmbedding(self.d_model, self.embed_type_sub, self.freq)
            
        elif self.embed_type == 'time_feature':
            self.module = TimeFeatureEmbedding(self.d_model, 'timeF', self.freq)
            
        elif self.embed_type == 'data':
            self.module = DataEmbedding(self.c_in, self.d_model, self.embed_type_sub, self.freq, self.dropout_rate)
            
        elif self.embed_type == 'data_inverted':
            self.module = DataEmbedding_inverted(self.c_in, self.d_model, self.embed_type_sub, self.freq, self.dropout_rate)
            
        elif self.embed_type == 'data_wo_pos':
            self.module = DataEmbedding_wo_pos(self.c_in, self.d_model, self.embed_type_sub, self.freq, self.dropout_rate)
            
        elif self.embed_type == 'patch':
            assert self.patch_len and self.stride and self.padding, "Require patch parameters"
            self.module = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.padding, self.dropout_rate)
        
        elif self.embed_type == 'spatio_temporal':
            self.module = SpatioTemporalEmbedding(
                in_steps= 4096,# args.patch_size_L,
                # steps_per_day=args.steps_per_day,
                # input_dim=args.input_dim,
                input_embedding_dim=args.input_embedding_dim,
                # tod_embedding_dim=args.tod_embedding_dim,
                # dow_embedding_dim=args.dow_embedding_dim,
                spatial_embedding_dim=args.spatial_embedding_dim,
                adaptive_embedding_dim=args.adaptive_embedding_dim
            )
        
        else:
            raise ValueError(f"Unsupported embedding type: {self.embed_type}")

        # 公共组件
        self.final_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, x_mark=None):
        B, L, C = x.shape
        x = x[:,:,0:1]
        
        # 分支处理
        if self.embed_type == 'patch':
            # 维度转换 (B, L, C) -> (B, C, L)
            x_patched = x.permute(0, 2, 1)
            embedded, n_vars = self.module(x_patched)
            
            # 重塑维度 (B*C, num_patches, d_model) -> (B, C*num_patches, d_model)
            output = embedded.view(B, -1, self.d_model)
            
        elif self.embed_type in ['data', 'data_inverted', 'data_wo_pos']:
            output = self.module(x, x_mark)
            
        elif self.embed_type == 'time_feature':
            output = self.module(x_mark)
            
        elif self.embed_type == 'temporal':
            output = self.module(x_mark)
            
            
        elif self.embed_type == 'token':
            output = self.module(x)
            
        elif self.embed_type == 'positional':
            output = self.module(x).expand(B, -1, -1)
            
        elif self.embed_type == 'spatio_temporal':
            output = self.module(x)
        else:
            raise RuntimeError(f"Unhandled embedding type: {self.embed_type}")

        # 统一后处理
        output = self.final_proj(output)
        return self.dropout(output)



# Original embedding classes below (copied from provided code)
class PositionalEmbedding(nn.Module):  
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='fixed', freq='h'):
#         super(TemporalEmbedding, self).__init__()
#         minute_size = 4; hour_size = 24; weekday_size = 7; day_size = 32; month_size = 13
#         Embed = nn.Embedding
#         if freq == 't':
#             self.minute_embed = Embed(minute_size, d_model)
#         self.hour_embed = Embed(hour_size, d_model)
#         self.weekday_embed = Embed(weekday_size, d_model)
#         self.day_embed = Embed(day_size, d_model)
#         self.month_embed = Embed(month_size, d_model)
#     def forward(self, x):
#         x = x.long()
#         minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
#         hour_x = self.hour_embed(x[:, :, 3])
#         weekday_x = self.weekday_embed(x[:, :, 2])
#         day_x = self.day_embed(x[:, :, 1])
#         month_x = self.month_embed(x[:, :, 0])
#         return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark):
        
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) # + self.temporal_embedding(x_mark)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len # 256
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class SpatioTemporalEmbedding(nn.Module):
    """时空嵌入模块，适配 (B, L, C) 输入格式"""
    def __init__(self,
                 in_steps,
                 steps_per_day=288,
                 input_dim=1,
                 input_embedding_dim=24,
                #  tod_embedding_dim=24,
                #  dow_embedding_dim=24,
                 spatial_embedding_dim=0,
                 adaptive_embedding_dim=80):
        super().__init__()
        
        self.in_steps = in_steps
        self.steps_per_day = steps_per_day
        
        # 输入特征投影
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        
        # 时间嵌入
        # self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim) if tod_embedding_dim > 0 else None
        # self.dow_embedding = nn.Embedding(7, dow_embedding_dim) if dow_embedding_dim > 0 else None
        
        # 空间嵌入（单节点）
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(1, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None
            
        # 自适应嵌入（时间步相关）
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(torch.empty(in_steps, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None
            
        # 计算总维度
        self.total_embed_dim = sum([
            input_embedding_dim,
            # tod_embedding_dim,
            # dow_embedding_dim,
            spatial_embedding_dim,
            adaptive_embedding_dim
        ])

    def forward(self, x):
        """
        输入形状: (batch_size, seq_len, input_dim + 2)
        (假设最后两个通道分别是 tod 和 dow 信息)
        输出形状: (batch_size, seq_len, total_embed_dim)
        """
        batch_size, seq_len = x.shape[:2]
        
        # # 分割输入特征
        # input_feature = x[..., :self.input_proj.in_features]  # (B, L, input_dim)
        # if self.tod_embedding is not None:
        #     tod = x[..., self.input_proj.in_features]         # (B, L)
        # if self.dow_embedding is not None:
        #     dow = x[..., self.input_proj.in_features + 1]     # (B, L)

        # 输入特征投影
        x = self.input_proj(x)
        features = [x]
        
        # # 添加时间嵌入
        # if self.tod_embedding is not None:
        #     tod_idx = (tod * self.steps_per_day).long() % self.steps_per_day
        #     features.append(self.tod_embedding(tod_idx))
            
        # if self.dow_embedding is not None:
        #     dow_idx = dow.long() % 7
        #     features.append(self.dow_embedding(dow_idx))
            
        # 添加空间嵌入
        if self.node_emb is not None:
            spatial_emb = self.node_emb.expand(batch_size, seq_len, -1)  # (B, L, D)
            features.append(spatial_emb)
            
        # 添加自适应嵌入
        if self.adaptive_embedding is not None:
            assert seq_len == self.in_steps, f"Input seq_len {seq_len} != initialized in_steps {self.in_steps}"
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, D)
            features.append(adp_emb)
            
        return torch.cat(features, dim=-1)


# 测试用例
if __name__ == '__main__':

    # 定义通用参数
    class Args:
        def __init__(self):
            self.output_dim = 64      # d_model
            self.dropout = 0.1
            self.freq = 'h'
            self.sub_embed_type = 'fixed'
            self.patch_size_C = 8     # c_in
            self.patch_size_L = 128    # patch_len
            self.stride = 8
            self.padding = 1
            self.in_steps = 128        # 时空嵌入需要
            self.steps_per_day = 288
            self.input_dim = 1        # 时空嵌入输入维度
            self.patch_len = 128

    # 支持的embedding类型列表
    embed_types = [
        # 'positional',
        # 'token',
        # 'time_feature',
        # 'data',
        # 'data_inverted',
        # 'data_wo_pos',
        # 'patch',
        'spatio_temporal'
    ]

    # 测试配置
    batch_size = 32
    seq_len = 128

    for embed_type in embed_types:
        print(f"\n=== Testing {embed_type} embedding ===")
        
        # 初始化参数
        args = Args()
        args.embed_type = embed_type
        
        # 特殊参数设置
        if embed_type == 'spatio_temporal':
            args.input_embedding_dim = 32
            args.tod_embedding_dim = 0
            args.dow_embedding_dim = 0
            args.spatial_embedding_dim = 0
            args.adaptive_embedding_dim = 32
        
        # 初始化模块
        model = E_com_00_PE(args)
        
        # 生成测试数据
        x = torch.randn(batch_size, seq_len, 1)
        x_mark = None
        
        # 特殊输入处理
        if embed_type == 'time_feature':
            x_mark = torch.randn(batch_size, seq_len, 4)  # freq='h'对应4维
            
        elif embed_type == 'spatio_temporal':
            x = torch.randn(batch_size, seq_len, 1)  # (input_dim + tod + dow)
            
        elif embed_type in ['data', 'data_inverted', 'data_wo_pos']:
            x_mark = torch.randn(batch_size, seq_len, 4) if embed_type != 'data_inverted' else None
        
        # 执行前向传播
        try:
            output = model(x, x_mark)
            print(f"Input shape: {x.shape}")
            if x_mark is not None: 
                print(f"Mark shape: {x_mark.shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error: {str(e)}")