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
    def __init__(self, args):
        super(E_01_HSE, self).__init__()
        self.patch_size_L = args.patch_size_L  # Patch size along L dimension
        self.patch_size_C = args.patch_size_C  # Patch size along C dimension
        self.num_patches = args.num_patches    # Number of patches to sample
        self.output_dim =  args.output_dim
        # self.args_d = args_d   
        # self.f_s =  args_d.f_s  # Sampling frequency
        # self.T = 1.0 /  args_d.f_s  # Sampling period


        # Two linear layers for flatten + mixing
        self.linear1 = nn.Linear(self.patch_size_L * (self.patch_size_C * 2), self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x: torch.Tensor, fs, **kwargs) -> torch.Tensor:
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
        # Handle sampling frequency - support both tensor and scalar inputs
        if torch.is_tensor(fs):
            T = 1.0 / fs  # [B] tensor
        else:
            T = 1.0 / fs  # scalar
            T = torch.full((B,), T, device=device)  # broadcast to [B]

        # Generate time axis 't' for each sample, shape: (B, L)
        t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1, L)
        t = t * T.unsqueeze(1)  # (B, 1) * (1, L) -> (B, L)

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
        self.num_patches = args.num_patches      # Number of patches to sample
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

# Note: Updated SystemPromptEncoder and PromptFusion are now imported from ISFM_Prompt components
# The classes below are kept for backward compatibility but use the new implementation

# Import the updated components from our ISFM_Prompt module
import sys
import os

# Try to import from ISFM_Prompt components if available, otherwise use fallback
try:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ISFM_Prompt.components.SystemPromptEncoder import SystemPromptEncoder as NewSystemPromptEncoder
    from ISFM_Prompt.components.PromptFusion import PromptFusion as NewPromptFusion
    _USE_NEW_COMPONENTS = True
except ImportError:
    _USE_NEW_COMPONENTS = False
    print("Warning: Could not import new Prompt components, using legacy implementations")

# Wrapper classes for backward compatibility
class SystemPromptEncoder(nn.Module):
    """
    Backward-compatible wrapper for the updated SystemPromptEncoder.
    
    CRITICAL UPDATE: Now uses two-level prompts only (NO fault-level prompts)
    - System level: Dataset_id + Domain_id  
    - Sample level: Sample_rate
    
    Label is NOT included as it's the prediction target!
    """
    def __init__(self, prompt_dim=128, max_ids=50):
        super().__init__()
        
        if _USE_NEW_COMPONENTS:
            # Use the new implementation
            self.encoder = NewSystemPromptEncoder(
                prompt_dim=prompt_dim,
                max_dataset_ids=max_ids,
                max_domain_ids=max_ids
            )
        else:
            # Fallback implementation (two-level prompts, no fault-level)
            self.prompt_dim = prompt_dim
            self.max_ids = max_ids
            
            # Calculate embedding dimensions to ensure proper concatenation  
            self.dataset_dim = prompt_dim // 3
            self.domain_dim = prompt_dim // 3
            self.sample_dim = prompt_dim - self.dataset_dim - self.domain_dim
            
            # Embedding tables for categorical features (NO Label embedding)
            self.dataset_embedding = nn.Embedding(max_ids, self.dataset_dim)
            self.domain_embedding = nn.Embedding(max_ids, self.domain_dim)
            
            # Linear layer for numerical features
            self.sample_rate_proj = nn.Linear(1, self.sample_dim)
            
            # Two-level fusion layers (system + sample, NO fault level)
            self.system_fusion = nn.Linear(self.dataset_dim + self.domain_dim, prompt_dim)
            self.sample_fusion = nn.Linear(self.sample_dim, prompt_dim)
            
            # Final prompt aggregation  
            self.prompt_aggregator = nn.MultiheadAttention(prompt_dim, 4, batch_first=True)
            self.final_proj = nn.Linear(prompt_dim, prompt_dim)
        
    def forward(self, metadata_dict):
        """
        Args:
            metadata_dict: Dictionary with system metadata
                - 'Dataset_id': tensor of shape (B,)
                - 'Domain_id': tensor of shape (B,)  
                - 'Sample_rate': tensor of shape (B,)
                
                CRITICAL: Does NOT contain 'Label' - fault type is prediction target!
        
        Returns:
            prompt_embedding: tensor of shape (B, prompt_dim)
        """
        if _USE_NEW_COMPONENTS:
            return self.encoder(metadata_dict)
        else:
            # Fallback implementation
            batch_size = metadata_dict['Dataset_id'].size(0)
            
            # System-level prompt (Dataset_id + Domain_id)
            dataset_emb = self.dataset_embedding(metadata_dict['Dataset_id'])
            domain_emb = self.domain_embedding(metadata_dict['Domain_id'])
            system_prompt = self.system_fusion(torch.cat([dataset_emb, domain_emb], dim=-1))
            
            # Sample-level prompt (Sample_rate)
            sample_rate_normalized = metadata_dict['Sample_rate'].unsqueeze(-1) / 10000.0
            sample_emb = self.sample_rate_proj(sample_rate_normalized)
            sample_prompt = self.sample_fusion(sample_emb)
            
            # Two-level prompt fusion (NO fault-level prompts)
            prompts = torch.stack([system_prompt, sample_prompt], dim=1)  # (B, 2, prompt_dim)
            
            # Self-attention to fuse system and sample levels
            fused_prompt, _ = self.prompt_aggregator(prompts, prompts, prompts)  # (B, 2, prompt_dim)
            
            # Aggregate to single prompt vector
            final_prompt = fused_prompt.mean(dim=1)  # (B, prompt_dim)
            final_prompt = self.final_proj(final_prompt)
            
            return final_prompt


class PromptFusion(nn.Module):
    """
    Backward-compatible wrapper for the updated PromptFusion.
    """
    def __init__(self, signal_dim, prompt_dim, fusion_type='attention'):
        super().__init__()
        
        if _USE_NEW_COMPONENTS:
            # Use the new implementation
            self.fusion = NewPromptFusion(signal_dim, prompt_dim, fusion_type)
        else:
            # Fallback implementation
            self.fusion_type = fusion_type
            self.signal_dim = signal_dim
            self.prompt_dim = prompt_dim
            
            if fusion_type == 'concat':
                self.fusion_proj = nn.Linear(signal_dim + prompt_dim, signal_dim)
                
            elif fusion_type == 'attention':
                # Cross-attention: signal attends to prompt
                self.cross_attention = nn.MultiheadAttention(signal_dim, 4, batch_first=True)
                self.prompt_proj = nn.Linear(prompt_dim, signal_dim)
                
            elif fusion_type == 'gating':
                # Adaptive gating mechanism
                self.gate_proj = nn.Linear(prompt_dim, signal_dim)
                self.transform_proj = nn.Linear(prompt_dim, signal_dim)
                
            self.layer_norm = nn.LayerNorm(signal_dim)
        
    def forward(self, signal_emb, prompt_emb):
        """
        Args:
            signal_emb: (B, num_patches, signal_dim)
            prompt_emb: (B, prompt_dim)
            
        Returns:
            fused_emb: (B, num_patches, signal_dim)
        """
        if _USE_NEW_COMPONENTS:
            return self.fusion(signal_emb, prompt_emb)
        else:
            # Fallback implementation
            if self.fusion_type == 'concat':
                # Expand prompt to match signal patches
                expanded_prompt = prompt_emb.unsqueeze(1).expand(-1, signal_emb.size(1), -1)
                concatenated = torch.cat([signal_emb, expanded_prompt], dim=-1)
                fused = self.fusion_proj(concatenated)
                
            elif self.fusion_type == 'attention':
                # Project prompt to signal dimension
                prompt_projected = self.prompt_proj(prompt_emb).unsqueeze(1)  # (B, 1, signal_dim)
                
                # Cross-attention: signal queries attend to prompt keys/values
                attended_signal, _ = self.cross_attention(
                    signal_emb, prompt_projected, prompt_projected
                )
                fused = signal_emb + attended_signal  # Residual connection
                
            elif self.fusion_type == 'gating':
                # Adaptive gating
                gate = torch.sigmoid(self.gate_proj(prompt_emb)).unsqueeze(1)
                transform = self.transform_proj(prompt_emb).unsqueeze(1)
                fused = gate * signal_emb + (1 - gate) * transform
                
            return self.layer_norm(fused)


class E_01_HSE_Prompt(nn.Module):
    """
    Prompt-guided Hierarchical Signal Embedding for heterogeneous system processing.
    
    CRITICAL UPDATE: Uses two-level prompts only (NO fault-level prompts):
    - System level: Dataset_id + Domain_id (identifies system and operating conditions)
    - Sample level: Sample_rate (captures signal acquisition parameters)
    - NO fault level: Label is the prediction target, not prompt input!
    
    Supports two-stage training:
    - Stage 1 (pretrain): Learn both signal and prompt features with contrastive learning
    - Stage 2 (finetune): Freeze prompt, finetune signal path for downstream tasks
    
    Args:
        patch_size_L (int): Patch size along L dimension
        patch_size_C (int): Patch size along C dimension  
        num_patches (int): Number of patches to sample
        output_dim (int): Output feature dimension
        prompt_dim (int): Prompt embedding dimension
        fusion_type (str): Fusion strategy ('concat', 'attention', 'gating')
        use_system_prompt (bool): Whether to use system-level prompts (Dataset_id + Domain_id)
        use_sample_prompt (bool): Whether to use sample-level prompts (Sample_rate)
    """
    def __init__(self, args):
        super().__init__()
        # Basic HSE parameters
        self.patch_size_L = getattr(args, 'patch_size_L', 256)
        self.patch_size_C = getattr(args, 'patch_size_C', 1) 
        self.num_patches = getattr(args, 'num_patches', 128)
        self.output_dim = getattr(args, 'output_dim', 1024)
        
        # Prompt parameters
        self.prompt_dim = getattr(args, 'prompt_dim', 128)
        self.fusion_type = getattr(args, 'fusion_type', 'attention')
        self.use_system_prompt = getattr(args, 'use_system_prompt', True)
        self.use_sample_prompt = getattr(args, 'use_sample_prompt', True) 
        # REMOVED: use_fault_prompt - Label is prediction target, not prompt input!
        
        # Training stage control
        self.training_stage = getattr(args, 'training_stage', 'pretrain')  # 'pretrain' or 'finetune'
        self.freeze_prompt = getattr(args, 'freeze_prompt', False)
        
        # Signal embedding path (based on original E_01_HSE)
        self.signal_linear1 = nn.Linear(self.patch_size_L * (self.patch_size_C * 2), self.output_dim)
        self.signal_linear2 = nn.Linear(self.output_dim, self.output_dim)
        
        # Prompt encoding path (only system + sample, NO fault prompts)
        if any([self.use_system_prompt, self.use_sample_prompt]):
            self.prompt_encoder = SystemPromptEncoder(self.prompt_dim)
            
            # Fusion module
            self.prompt_fusion = PromptFusion(
                signal_dim=self.output_dim,
                prompt_dim=self.prompt_dim, 
                fusion_type=self.fusion_type
            )
        
        # Final projection
        self.final_proj = nn.Linear(self.output_dim, self.output_dim)
        
    def set_training_stage(self, stage):
        """Set training stage: 'pretrain' or 'finetune'"""
        self.training_stage = stage
        if stage == 'finetune':
            self.freeze_prompt = True
        
    def encode_system_metadata(self, metadata_batch):
        """
        Extract and encode system information from metadata batch.
        
        CRITICAL UPDATE: Only extracts system and sample level information.
        Label is NOT included as it's the prediction target!
        
        Args:
            metadata_batch: List of metadata dictionaries or single dict
            
        Returns:
            metadata_dict: Dictionary with tensors for batch processing (NO Label!)
        """
        if isinstance(metadata_batch, dict):
            metadata_batch = [metadata_batch]
            
        batch_size = len(metadata_batch)
        
        # Extract system information (NO Label - it's the prediction target!)
        dataset_ids = []
        domain_ids = []
        sample_rates = []
        
        for meta in metadata_batch:
            dataset_ids.append(int(meta.get('Dataset_id', 0)))
            domain_ids.append(int(meta.get('Domain_id', 0)))
            sample_rates.append(float(meta.get('Sample_rate', 1000.0)))
            # REMOVED: labels.append(int(meta.get('Label', 0))) - Label is prediction target!
        
        device = next(self.parameters()).device
        
        return {
            'Dataset_id': torch.tensor(dataset_ids, device=device),
            'Domain_id': torch.tensor(domain_ids, device=device),
            'Sample_rate': torch.tensor(sample_rates, device=device)
            # REMOVED: 'Label' - fault type is prediction target, not prompt input!
        }
    
    def forward(self, x, fs=None, metadata=None, **kwargs):
        """
        Forward pass with prompt guidance.
        
        Args:
            x: Input tensor (B, L, C)
            fs: Sampling frequency (B,) or scalar
            metadata: System metadata (list of dicts or single dict)
            
        Returns:
            output: Signal embedding (B, num_patches, output_dim)
            prompt: System prompt embedding (B, prompt_dim) - for contrastive learning
        """
        B, L, C = x.size()
        device = x.device
        
        # Handle sampling frequency
        if fs is None:
            fs = 1000.0  # Default
        if torch.is_tensor(fs):
            T = 1.0 / fs
        else:
            T = 1.0 / fs
            T = torch.full((B,), T, device=device)
        
        # Generate time axis 
        t = torch.arange(L, device=device, dtype=torch.float32)
        if T.dim() > 0:
            t = t.unsqueeze(0) * T.unsqueeze(1)  # (B, L)
        else:
            t = t * T
            t = t.unsqueeze(0).expand(B, -1)
        
        # Handle size mismatches (same as original E_01_HSE)
        if self.patch_size_L > L:
            repeats_L = (self.patch_size_L + L - 1) // L
            x = repeat(x, 'b l c -> b (l r) c', r=repeats_L)
            t = repeat(t, 'b l -> b (l r)', r=repeats_L)
            L = x.size(1)
            
        if self.patch_size_C > C:
            repeats_C = (self.patch_size_C + C - 1) // C
            x = repeat(x, 'b l c -> b l (c r)', r=repeats_C)
            C = x.size(2)
        
        # Random patch sampling (same as original E_01_HSE)
        max_start_L = L - self.patch_size_L
        max_start_C = C - self.patch_size_C
        start_indices_L = torch.randint(0, max_start_L + 1, (B, self.num_patches), device=device)
        start_indices_C = torch.randint(0, max_start_C + 1, (B, self.num_patches), device=device)
        
        # Create offsets and gather patches
        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)
        
        idx_L = (start_indices_L.unsqueeze(-1) + offsets_L) % L
        idx_C = (start_indices_C.unsqueeze(-1) + offsets_C) % C
        
        idx_L = idx_L.unsqueeze(-1)
        idx_C = idx_C.unsqueeze(-2)
        
        patches = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))
        patches = patches.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))
        
        # Add time embeddings
        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))
        t_patches = t_patches.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_C)
        
        patches = torch.cat([patches, t_patches], dim=-1)
        
        # Signal embedding path
        patches = rearrange(patches, 'b p l c -> b p (l c)')
        signal_emb = self.signal_linear1(patches)
        signal_emb = F.silu(signal_emb)
        signal_emb = self.signal_linear2(signal_emb)
        
        # Prompt guidance
        prompt_emb = None
        if hasattr(self, 'prompt_encoder') and metadata is not None:
            # Encode system metadata
            metadata_dict = self.encode_system_metadata(metadata)
            prompt_emb = self.prompt_encoder(metadata_dict)
            
            # Freeze prompt during finetune stage
            if self.freeze_prompt or self.training_stage == 'finetune':
                prompt_emb = prompt_emb.detach()
            
            # Fuse prompt with signal embeddings
            signal_emb = self.prompt_fusion(signal_emb, prompt_emb)
        
        # Final projection
        output = self.final_proj(signal_emb)
        
        return output if prompt_emb is None else (output, prompt_emb)


if __name__ == '__main__':
    print("=== E_01_HSE Prompt-guided Testing ===")
    
    # Test original HSE
    def test_original_hse():
        print("\n1. Testing Original E_01_HSE:")
        class Args:
            patch_size_L = 128
            patch_size_C = 1
            num_patches = 64
            output_dim = 256
            
        model = E_01_HSE(Args())
        
        B, L, C = 2, 1024, 2
        x = torch.randn(B, L, C)
        fs = torch.tensor([1000.0, 2000.0])
        
        output = model(x, fs)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
        
    # Test prompt-guided HSE
    def test_prompt_hse():
        print("\n2. Testing Prompt-guided E_01_HSE:")
        
        class Args:
            patch_size_L = 128
            patch_size_C = 1
            num_patches = 64
            output_dim = 256
            prompt_dim = 128
            fusion_type = 'attention'
            use_system_prompt = True
            use_sample_prompt = True
            # REMOVED: use_fault_prompt - Label is prediction target, not prompt input!
            training_stage = 'pretrain'
            freeze_prompt = False
            
        model = E_01_HSE_Prompt(Args())
        
        B, L, C = 2, 1024, 2
        x = torch.randn(B, L, C)
        fs = 1000.0
        
        # Mock metadata (NO Label - it's the prediction target!)
        metadata = [
            {'Dataset_id': 1, 'Domain_id': 5, 'Sample_rate': 1000.0},
            {'Dataset_id': 2, 'Domain_id': 3, 'Sample_rate': 2000.0}
        ]
        
        output, prompt = model(x, fs, metadata)
        print(f"  Input: {x.shape} -> Output: {output.shape}, Prompt: {prompt.shape}")
        
        # Test stage switching
        print("\n  Testing stage switching:")
        model.set_training_stage('finetune')
        output2, prompt2 = model(x, fs, metadata)
        print(f"  Finetune stage - Output: {output2.shape}, Prompt: {prompt2.shape}")
        
        # Test without metadata (fallback)
        print("\n  Testing without metadata:")
        output3 = model(x, fs, metadata=None)
        print(f"  No metadata - Output: {output3.shape if isinstance(output3, torch.Tensor) else output3[0].shape}")
    
    # Test system prompt encoder
    def test_prompt_encoder():
        print("\n3. Testing SystemPromptEncoder:")
        
        encoder = SystemPromptEncoder(prompt_dim=128)
        
        metadata_dict = {
            'Dataset_id': torch.tensor([1, 2, 1]),
            'Domain_id': torch.tensor([5, 3, 7]),
            'Sample_rate': torch.tensor([1000.0, 2000.0, 1500.0])
            # REMOVED: 'Label' - fault type is prediction target, not prompt input!
        }
        
        prompt = encoder(metadata_dict)
        print(f"  Metadata batch size 3 -> Prompt: {prompt.shape}")
        
    # Test fusion strategies
    def test_fusion_strategies():
        print("\n4. Testing Fusion Strategies:")
        
        signal_dim, prompt_dim = 256, 128
        B, num_patches = 2, 64
        
        signal_emb = torch.randn(B, num_patches, signal_dim)
        prompt_emb = torch.randn(B, prompt_dim)
        
        for fusion_type in ['concat', 'attention', 'gating']:
            fusion = PromptFusion(signal_dim, prompt_dim, fusion_type)
            fused = fusion(signal_emb, prompt_emb)
            print(f"  {fusion_type}: Signal {signal_emb.shape} + Prompt {prompt_emb.shape} -> {fused.shape}")
    
    # Run all tests
    test_original_hse()
    test_prompt_hse()
    test_prompt_encoder()
    test_fusion_strategies()
    
    print("\n=== All tests completed successfully! ===")
    print("\nKey features implemented:")
    print("✓ UPDATED: System metadata encoding (Dataset_id, Domain_id, Sample_rate)")
    print("✓ CRITICAL: NO Label in prompts - it's the prediction target!")
    print("✓ Two-level prompt fusion (system + sample, NO fault level)")
    print("✓ Three fusion strategies (concat, attention, gating)")
    print("✓ Two-stage training support (pretrain/finetune)")
    print("✓ Prompt freezing mechanism")
    print("✓ Backward compatibility with original E_01_HSE")
    print("✓ Updated for ICML/NeurIPS 2025 publication requirements")
