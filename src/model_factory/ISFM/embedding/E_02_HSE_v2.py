import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class E_02_HSE_v2(nn.Module):
    """
    Hierarchical Signal Embedding (HSE) with channel attribute embedding and system-specific prompts.
    
    Randomly divides the input tensor along L (length) and C (channel) dimensions into patches,
    then mixes these patches using linear layers. Incorporates time embedding, channel attribute 
    embedding, and system-specific prompts for heterogeneous signal processing.

    Args:
        args: Configuration object containing:
            patch_size_L (int): Patch size along the L dimension.
            patch_size_C (int): Patch size along the C dimension.
            num_patches (int): Number of random patches to sample.
            output_dim (int): Output feature dimension after linear mixing.
            num_channel_types (int): Number of distinct channel types.
            channel_embed_dim (int): Embedding dimension for channel attributes.
            num_systems (int): Number of different systems for system prompts.
            system_embed_dim (int): Embedding dimension for system prompts.
    """
    def __init__(self, args):
        super(E_02_HSE_v2, self).__init__()
        self.patch_size_L = args.patch_size_L
        self.patch_size_C = args.patch_size_C
        self.num_patches = args.num_patches
        self.output_dim = args.output_dim
        
        # Channel embedding parameters
        self.num_channel_types = getattr(args, 'num_channel_types', 10)
        self.channel_embed_dim = getattr(args, 'channel_embed_dim', 8)
        self.channel_embed = nn.Embedding(self.num_channel_types, self.channel_embed_dim)
        
        # System embedding parameters for system-specific prompts
        self.num_systems = getattr(args, 'num_systems', 5)
        self.system_embed_dim = getattr(args, 'system_embed_dim', 16)
        self.system_embed = nn.Embedding(self.num_systems, self.system_embed_dim)
        
        # Learnable channel type mapping for different systems
        self.use_adaptive_channel_mapping = getattr(args, 'use_adaptive_channel_mapping', True)
        if self.use_adaptive_channel_mapping:
            self.channel_type_mapping = nn.Parameter(
                torch.randint(0, self.num_channel_types, (self.num_systems, 20))  # Max 20 channels per system
            )
        
        # Input dimension calculation
        # data + time + channel_embed + system_embed (broadcasted)
        patch_data_dim = self.patch_size_L * self.patch_size_C
        patch_time_dim = self.patch_size_L * self.patch_size_C
        patch_channel_dim = self.patch_size_L * self.patch_size_C * self.channel_embed_dim
        patch_system_dim = self.system_embed_dim  # System embedding is added per patch
        
        input_linear_dim = patch_data_dim + patch_time_dim + patch_channel_dim + patch_system_dim
        
        # Linear layers with improved architecture
        self.norm1 = nn.LayerNorm(input_linear_dim)
        self.linear1 = nn.Linear(input_linear_dim, self.output_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)
        self.dropout2 = nn.Dropout(0.1)

    def get_channel_types(self, system_id: int, num_channels: int) -> torch.Tensor:
        """
        Get channel types for a given system and number of channels.
        
        Args:
            system_id (int): System identifier
            num_channels (int): Number of channels in the input
            
        Returns:
            torch.Tensor: Channel type IDs of shape (num_channels,)
        """
        if self.use_adaptive_channel_mapping:
            # Use learnable mapping
            system_channel_types = self.channel_type_mapping[system_id]
            if num_channels <= len(system_channel_types):
                return system_channel_types[:num_channels]
            else:
                # Repeat pattern if more channels than available
                repeats = (num_channels + len(system_channel_types) - 1) // len(system_channel_types)
                extended = system_channel_types.repeat(repeats)
                return extended[:num_channels]
        else:
            # Simple sequential mapping
            return torch.arange(num_channels) % self.num_channel_types

    def forward(self, x: torch.Tensor, fs: torch.Tensor, system_id: int) -> torch.Tensor:
        """
        Forward pass with system-aware channel attribute embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C)
            fs (torch.Tensor): Sampling frequency (scalar or per-batch)
            system_id (int): System identifier for system-specific processing

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, output_dim)
        """
        B, L_orig, C_orig = x.size()
        device = x.device
        
        # Get channel types for this system
        channel_types = self.get_channel_types(system_id, C_orig).to(device)
        
        # Handle sampling frequency
        if isinstance(fs, (int, float)):
            T = 1.0 / fs
        else:
            T = 1.0 / fs
            if T.ndim > 0:
                T = T.unsqueeze(-1) if len(T.shape) == 1 else T

        # Generate time axis
        time_axis_base = torch.arange(L_orig, device=device, dtype=torch.float32)
        if isinstance(T, torch.Tensor) and T.ndim > 0:
            t = time_axis_base.unsqueeze(0) * T
        else:
            t = (time_axis_base * T).unsqueeze(0).expand(B, -1)

        # Handle size mismatches with padding/repeating
        current_channel_types = channel_types
        if self.patch_size_L > L_orig:
            repeats_L = (self.patch_size_L + L_orig - 1) // L_orig
            x = repeat(x, 'b l c -> b (l r) c', r=repeats_L)
            t = repeat(t, 'b l -> b (l r)', r=repeats_L)
            L = x.size(1)
        else:
            L = L_orig

        if self.patch_size_C > C_orig:
            repeats_C = (self.patch_size_C + C_orig - 1) // C_orig
            x = repeat(x, 'b l c -> b l (c r)', r=repeats_C)
            current_channel_types = current_channel_types.repeat(repeats_C)[:x.size(2)]
            C = x.size(2)
        else:
            C = C_orig

        # Sample patch positions
        max_start_L = max(0, L - self.patch_size_L)
        max_start_C = max(0, C - self.patch_size_C)
        
        start_indices_L = torch.randint(0, max(1, max_start_L + 1), (B, self.num_patches), device=device)
        start_indices_C = torch.randint(0, max(1, max_start_C + 1), (B, self.num_patches), device=device)

        # Create offsets
        offsets_L = torch.arange(self.patch_size_L, device=device)
        offsets_C = torch.arange(self.patch_size_C, device=device)

        # Compute indices with proper boundary handling
        idx_L = (start_indices_L.unsqueeze(-1) + offsets_L) % L
        idx_C = (start_indices_C.unsqueeze(-1) + offsets_C) % C

        # Expand for gathering
        idx_L_expanded = idx_L.unsqueeze(-1)
        idx_C_expanded = idx_C.unsqueeze(-2)

        # Gather patches
        patches_data = x.unsqueeze(1).expand(-1, self.num_patches, -1, -1)
        patches_data = patches_data.gather(2, idx_L_expanded.expand(-1, -1, -1, C))
        patches_data = patches_data.gather(3, idx_C_expanded.expand(-1, -1, self.patch_size_L, -1))

        # Gather time embeddings
        t_expanded = t.unsqueeze(1).expand(-1, self.num_patches, -1)
        t_patches = t_expanded.gather(2, idx_L)
        t_patches = t_patches.unsqueeze(-1).expand(-1, -1, -1, self.patch_size_C)

        # Gather channel embeddings
        patch_channel_ids = current_channel_types[idx_C.long()]
        patch_channel_embeds = self.channel_embed(patch_channel_ids)
        patch_channel_embeds = patch_channel_embeds.unsqueeze(2).expand(-1, -1, self.patch_size_L, -1, -1)
        patch_channel_embeds = rearrange(patch_channel_embeds, 'b p l pc ced -> b p l (pc ced)')

        # System embedding (broadcasted to all patches)
        system_embed = self.system_embed(torch.tensor(system_id, device=device))
        system_embed = system_embed.unsqueeze(0).unsqueeze(0).expand(B, self.num_patches, -1)

        # Flatten and concatenate all features
        patches_data_flat = rearrange(patches_data, 'b p l c -> b p (l c)')
        t_patches_flat = rearrange(t_patches, 'b p l c -> b p (l c)')
        
        # Concatenate all features
        patches = torch.cat([
            patches_data_flat, 
            t_patches_flat, 
            patch_channel_embeds, 
            system_embed
        ], dim=-1)

        # Apply improved linear layers with normalization and dropout
        patches = self.norm1(patches)
        out = self.linear1(patches)
        out = F.silu(out)
        out = self.dropout1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        
        return out



if __name__ == '__main__':
    # 测试代码
    class MockArgs:
        def __init__(self):
            self.patch_size_L = 64
            self.patch_size_C = 2
            self.num_patches = 32
            self.output_dim = 128
            self.num_channel_types = 8
            self.channel_embed_dim = 16
            self.num_systems = 3
            self.system_embed_dim = 32
            self.use_adaptive_channel_mapping = True

    def test_hse():
        args = MockArgs()
        model = E_02_HSE_v2(args)
        model.eval()

        B, L, C = 4, 512, 6
        x = torch.randn(B, L, C)
        fs = torch.tensor(1000.0)
        system_id = 1

        print("Testing E_01_HSE:")
        y = model(x, fs, system_id)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Expected: ({B}, {args.num_patches}, {args.output_dim})")
        
        assert y.shape == (B, args.num_patches, args.output_dim)
        print("✓ Test passed!")

    test_hse()
