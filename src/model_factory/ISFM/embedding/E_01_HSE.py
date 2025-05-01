import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class E_01_HSE(nn.Module):
    """
    Randomly divides the input tensor along L (length) and C (channel) dimensions into patches,
    then mixes these patches using linear layers. After the patches are selected, a time embedding
    is added based on the sampling period, ensuring temporal information is preserved without
    including the time axis in patch selection.

    Args:
        patch_size_L (int): Patch size along the L dimension.
        patch_size_C (int): Patch size along the C dimension.
        num_patches (int): Number of random patches to sample.
        output_dim (int): Output feature dimension after linear mixing.
        f_s (int): Sampling frequency, used to compute sampling period (T = 1/f_s).
    """
    def __init__(self, args,args_d):
        super(E_01_HSE, self).__init__()
        self.patch_size_L = args.patch_size_L  # Patch size along L dimension
        self.patch_size_C = args.patch_size_C  # Patch size along C dimension
        self.num_patches = args.n_patches    # Number of patches to sample
        self.output_dim =  args.output_dim
        self.args_d = args_d   
        # self.f_s =  args_d.f_s  # Sampling frequency
        # self.T = 1.0 /  args_d.f_s  # Sampling period


        # Two linear layers for flatten + mixing
        self.linear1 = nn.Linear(self.patch_size_L * (self.patch_size_C * 2), self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x: torch.Tensor,data_name) -> torch.Tensor:
        """
        Forward pass of RandomPatchMixer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C),
                              where B is batch size, L is length, C is channels.

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, output_dim).
        """
        B, L, C = x.size()
        device = x.device
        fs = self.args_d.task[data_name]['f_s']
        T = 1.0 / fs

        # Generate time axis 't' for each sample, shape: (B, L)
        t = torch.arange(L, device=device, dtype=torch.float32) * T
        t = t.unsqueeze(0).expand(B, -1)

        # If input is smaller than required patch size, repeat along L or C as needed
        if self.patch_size_L > L:
            repeats_L = (self.patch_size_L + L - 1) // L  # Ceiling division
            x = repeat(x, 'b l c -> b (l r) c', r=repeats_L)
            t = repeat(t, 'b l -> b (l r)', r=repeats_L)
            L = x.size(1)

        if self.patch_size_C > C:
            repeats_C = (self.patch_size_C + C - 1) // C
            x = repeat(x, 'b l c -> b l (c r)', r=repeats_C)
            C = x.size(2)

        # Randomly sample starting positions for patches
        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C
        start_indices_L = torch.randint(0, max_start_L + 1, (B, self.num_patches), device=device)
        start_indices_C = torch.randint(0, max_start_C + 1, (B, self.num_patches), device=device)

        # Create offsets for patch sizes
        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)

        # Compute actual indices
        idx_L = (start_indices_L.unsqueeze(-1) + offsets_L) % L  # (B, num_patches, patch_size_L)
        idx_C = (start_indices_C.unsqueeze(-1) + offsets_C) % C  # (B, num_patches, patch_size_C)

        # Expand for advanced indexing
        idx_L = idx_L.unsqueeze(-1)  # (B, num_patches, patch_size_L, 1)
        idx_C = idx_C.unsqueeze(-2)  # (B, num_patches, 1, patch_size_C)

        # Gather patches
        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))
        patches = patches.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))

        # Gather corresponding time embeddings
        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)  # (B, num_patches, L)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))           # (B, num_patches, patch_size_L)
        t_patches = t_patches.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_C)

        # Concatenate time embedding to the end along channel dimension
        patches = torch.cat([patches, t_patches], dim=-1)  # shape: (B, num_patches, patch_size_L, patch_size_C + 1)

        # Flatten each patch and apply linear layers
        patches = rearrange(patches, 'b p l c -> b p (l c)')
        out = self.linear1(patches)
        out = F.silu(out)
        out = self.linear2(out)
        return out
    
#%% 
    
class E_01_HSE_abalation(nn.Module):
    """
    Hierarchical Signal Embedding (HSE) module.
    
    支持多种消融实验参数:
    - sampling_mode: 'random'或'sequential'采样模式
    - apply_mixing: 是否对patch进行mixing处理
    - linear_config: Linear层深度配置
    - patch_scale: Patch参数放大倍数
    - activation_type: 激活函数类型
    
    Args:
        patch_size_L (int): Patch size along the L dimension.
        patch_size_C (int): Patch size along the C dimension.
        num_patches (int): Number of random patches to sample.
        output_dim (int): Output feature dimension after linear mixing.
    """
    def __init__(self, args, args_d):
        super(E_01_HSE_abalation, self).__init__()
        # 基本参数
        self.patch_size_L = args.patch_size_L  # Patch size along L dimension
        self.patch_size_C = args.patch_size_C  # Patch size along C dimension
        self.num_patches = args.n_patches      # Number of patches to sample
        self.output_dim = args.output_dim
        self.args_d = args_d
        
        # 消融实验参数
        self.sampling_mode = getattr(args, 'sampling_mode', 'random')  # 'random'或'sequential'
        self.apply_mixing = getattr(args, 'apply_mixing', True)       # 是否混合patch
        
        # 获取线性层配置，默认为(1,1)
        if hasattr(args, 'linear_config'):
            if isinstance(args.linear_config, (list, tuple)) and len(args.linear_config) == 2:
                self.linear_config = tuple(args.linear_config) 
            else:
                self.linear_config = (1, 1)
        else:
            self.linear_config = (1, 1)
        
        # 获取patch缩放参数
        if hasattr(args, 'patch_scale'):
            if isinstance(args.patch_scale, (list, tuple)) and len(args.patch_scale) == 3:
                self.patch_scale = tuple(args.patch_scale)
            else:
                self.patch_scale = (1, 1, 1)
        else:
            self.patch_scale = (1, 1, 1)
            
        # 应用patch缩放
        self.patch_size_L *= self.patch_scale[0]
        self.patch_size_C *= self.patch_scale[1]
        self.num_patches *= self.patch_scale[2]
        
        # 激活函数类型
        self.activation_type = getattr(args, 'activation_type', 'silu')
        self.activation_fn = self._get_activation_fn(self.activation_type)
        
        # 创建线性层
        self._setup_linear_layers()
        
    def _get_activation_fn(self, activation_type):
        """获取激活函数"""
        activation_map = {
            'silu': F.silu,
            'relu': F.relu,
            'gelu': F.gelu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        return activation_map.get(activation_type.lower(), F.silu)
        
    def _setup_linear_layers(self):
        """设置线性层"""
        layer1_depth, layer2_depth = self.linear_config
        
        # 创建第一个线性变换
        if layer1_depth == 1:
            self.linear1 = nn.Linear(self.patch_size_L * (self.patch_size_C * 2), self.output_dim)
        else:
            layers = [nn.Linear(self.patch_size_L * (self.patch_size_C * 2), self.output_dim)]
            for _ in range(layer1_depth - 1):
                layers.extend([
                    nn.LayerNorm(self.output_dim),
                    # self.activation_fn,
                    nn.Linear(self.output_dim, self.output_dim)
                ])
            self.linear1 = nn.Sequential(*layers)
        
        # 创建第二个线性变换
        if not self.apply_mixing:
            self.linear2 = nn.Identity()  # 如果不应用mixing，使用恒等映射
        elif layer2_depth == 1:
            self.linear2 = nn.Linear(self.output_dim, self.output_dim)
        else:
            layers = [nn.Linear(self.output_dim, self.output_dim)]
            for _ in range(layer2_depth - 1):
                layers.extend([
                    nn.LayerNorm(self.output_dim),
                    # self.activation_fn,
                    nn.Linear(self.output_dim, self.output_dim)
                ])
            self.linear2 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, data_name) -> torch.Tensor:
        """
        Forward pass of HSE.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C),
                             where B is batch size, L is length, C is channels.
            data_name (str): Dataset name for sampling frequency.

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, output_dim).
        """
        B, L, C = x.size()
        device = x.device
        fs = self.args_d.task[data_name]['f_s']
        T = 1.0 / fs

        # 生成时间轴
        t = torch.arange(L, device=device, dtype=torch.float32) * T
        t = t.unsqueeze(0).expand(B, -1)

        # 处理尺寸不匹配
        if self.patch_size_L > L:
            repeats_L = (self.patch_size_L + L - 1) // L
            x = repeat(x, 'b l c -> b (l r) c', r=repeats_L)
            t = repeat(t, 'b l -> b (l r)', r=repeats_L)
            L = x.size(1)

        if self.patch_size_C > C:
            repeats_C = (self.patch_size_C + C - 1) // C
            x = repeat(x, 'b l c -> b l (c r)', r=repeats_C)
            C = x.size(2)

        # 根据采样模式选择处理方式
        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C
        
        if self.sampling_mode == 'random':
            # 随机采样
            start_indices_L = torch.randint(0, max(1, max_start_L + 1), (B, self.num_patches), device=device)
            start_indices_C = torch.randint(0, max(1, max_start_C + 1), (B, self.num_patches), device=device)
        else:
            # 顺序采样
            step_L = max(1, max_start_L // (self.num_patches - 1) if self.num_patches > 1 else 1)
            step_C = max(1, max_start_C // (self.num_patches - 1) if self.num_patches > 1 else 1)
            
            # 生成等距起始点
            start_L_seq = torch.arange(0, min(max_start_L+1, self.num_patches * step_L), step_L, device=device)
            start_C_seq = torch.arange(0, min(max_start_C+1, self.num_patches * step_C), step_C, device=device)
            
            # 确保有足够的起始点
            if len(start_L_seq) < self.num_patches:
                start_L_seq = start_L_seq.repeat((self.num_patches + len(start_L_seq) - 1) // len(start_L_seq))
                start_L_seq = start_L_seq[:self.num_patches]
            
            if len(start_C_seq) < self.num_patches:
                start_C_seq = start_C_seq.repeat((self.num_patches + len(start_C_seq) - 1) // len(start_C_seq))
                start_C_seq = start_C_seq[:self.num_patches]
            
            # 扩展到批次维度
            start_indices_L = start_L_seq.unsqueeze(0).expand(B, -1)
            start_indices_C = start_C_seq.unsqueeze(0).expand(B, -1)

        # 创建偏移量
        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)

        # 计算实际索引
        idx_L = (start_indices_L.unsqueeze(-1) + offsets_L) % L  # (B, num_patches, patch_size_L)
        idx_C = (start_indices_C.unsqueeze(-1) + offsets_C) % C  # (B, num_patches, patch_size_C)

        # 扩展用于高级索引
        idx_L = idx_L.unsqueeze(-1)  # (B, num_patches, patch_size_L, 1)
        idx_C = idx_C.unsqueeze(-2)  # (B, num_patches, 1, patch_size_C)

        # 收集patches
        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))
        patches = patches.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))

        # 收集对应的时间嵌入
        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))
        t_patches = t_patches.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_C)

        # 沿通道维度连接时间嵌入
        patches = torch.cat([patches, t_patches], dim=-1)

        # 展平每个patch并应用线性层
        patches = rearrange(patches, 'b p l c -> b p (l c)')
        out = self.linear1(patches)
        
        if self.apply_mixing:
            out = self.activation_fn(out)
            out = self.linear2(out)
            
        return out

if __name__ == '__main__':
    # Testing the RandomPatchMixer class
    def test_random_patch_mixer():
        B = 2  # Batch size
        L_list = [1024, 2048]  # Variable sequence lengths
        C_list = [8, 3]   # Variable channel dimensions

        patch_size_L = 128   # Patch size along L dimension
        patch_size_C = 5   # Patch size along C dimension
        num_patches = 100   # Number of patches to sample
        output_dim = 16    # Output dimension after mixing
        f_s = 100  # Sampling frequency

        model = E_01_HSE(patch_size_L, patch_size_C, num_patches, output_dim, f_s)

        for C in C_list:
            for L in L_list:
                x = torch.randn(B, L, C)
                y = model(x)
                print(f'Input shape: {x.shape}, Output shape: {y.shape}')

    # Run the test
    test_random_patch_mixer()
