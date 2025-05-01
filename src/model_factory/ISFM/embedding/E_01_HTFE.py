
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

class E_01_HTFE(nn.Module):
    """
    Randomly divides the input tensor along L and C dimensions into patches and mixes them using linear layers.
    Time embedding is added to the patches after they have been selected, preserving the temporal information
    without including the time axis in the patch selection process.
    """
    def __init__(self, args,args_d):
        super(E_01_HTFE, self).__init__()
        self.patch_size_L = args.patch_size_L  # Patch size along L dimension
        self.patch_size_C = args.patch_size_C  # Patch size along C dimension
        self.num_patches = args.n_patches    # Number of patches to sample
        self.output_dim =  args.output_dim
        self.f_s =  args.f_s  # Sampling frequency
        self.T = 1.0 /  args_d.f_s  # Sampling period
        

        # Linear layers for alignment and mixing
        # Input dimension remains the same as before, time embedding will be added after patch selection
        
        # [B, num_patches, patch_size_L, patch_size_C * 3] -> [B, num_patches, patch_size_L, output_dim]
        self.proj_in = nn.Sequential(
            # nn.LayerNorm(patch_size_C + 3),  # +3 for t, c, f
            nn.Linear(self.patch_size_C * 2 + 1,self.output_dim),  # 
            nn.SiLU(),
        )
        # [B, num_patches, patch_size_L, output_dim + 1] -> [B, num_patches, patch_size_L, output_dim]
        self.fft_proj = nn.Sequential(
            nn.Linear(self.output_dim * 2 + 1, self.output_dim),
            nn.SiLU(),
        )
        self.mixer = nn.Sequential(
            nn.Linear(self.output_dim * self.patch_size_L, self.output_dim),
            nn.SiLU(),
        )
        
        # self.fft_linear = nn.Linear(output_dim + 1, output_dim)
        # # [B, num_patches, patch_size_L, output_dim] -> [B, num_patches, output_dim]
        # self.linear2 = nn.Linear(output_dim * self.patch_size_L, output_dim)

    def forward(self, x):
        B, L, C = x.size()
        device = x.device

        # Generate time axis t, starting from 0
        t = torch.arange(L, device=device).float() * self.T  # Shape: (L,)
        t = t.unsqueeze(0).expand(B, -1)  # Shape: (B, L)
        f = torch.linspace(0, self.f_s / 2, self.patch_size_L // 2 + 1, device=device)  # Frequency axis Shape: (L // 2 + 1,)
        c = torch.arange(C, device=device)  # Channel axis Shape: (C,)
        
        # Adjust x to fit the patch sizes if necessary by repeating elements
        if self.patch_size_L > L:
            repeats_L = (self.patch_size_L + L - 1) // L  # Ceiling division
            x = repeat(x, 'b l c -> b (l r) c', r=repeats_L)
            t = repeat(t, 'b l -> b (l r)', r=repeats_L)
            L = x.size(1)  # Update L

        if self.patch_size_C > C:
            repeats_C = (self.patch_size_C + C - 1) // C  # Ceiling division
            x = repeat(x, 'b l c -> b l (c r)', r=repeats_C)
            c = repeat(c, 'c -> (c r)', r=repeats_C)
            C = x.size(2)  # Update C

        # Randomly select starting positions for patches along L and C dimensions
        max_start_L = L - self.patch_size_L
        start_indices_L = torch.randint(0, max_start_L + 1, (B, self.num_patches), device=device)  # (B, num_patches)
        # Create offsets for patch sizes
        offsets_L = torch.arange(self.patch_size_L, device=device)  # (patch_size_L,)


        # Compute indices for patches
        idx_L = start_indices_L.unsqueeze(-1) + offsets_L  # (B, num_patches, patch_size_L)
        # Randomly select indices for patches along C dimension
        idx_C = torch.randint(0, C, (B, self.num_patches, self.patch_size_C), device=device)  # (B, num_patches, patch_size_C)

        # Use modulo operation to handle cases where indices exceed dimensions
        idx_L = idx_L % L  # (B, num_patches, patch_size_L)

        # Expand batch dimension for indexing
        idx_L = rearrange(idx_L, 'b p l -> b p l 1')  # (B, num_patches, patch_size_L, 1)
        idx_C = rearrange(idx_C, 'b p c -> b p 1 c')  # (B, num_patches, 1, patch_size_C)

        # Extract patches using advanced indexing
        patches = rearrange(x,'b l c -> b 1 l c').expand(-1, self.num_patches, -1, -1)  # (B, num_patches, L, C)
        patches = patches.gather(2, idx_L.expand(-1, -1, -1, C))  # (B, num_patches, patch_size_L, C)
        patches = patches.gather(3, idx_C.expand(-1, -1, self.patch_size_L, -1))  # (B, num_patches, patch_size_L, patch_size_C)

        # After patches are selected, add time embedding
        t_expanded = rearrange(t,'b l -> b 1 l').expand(-1, self.num_patches, -1)  # (B, num_patches, L)
        t_patches = t_expanded.gather(2, idx_L.squeeze(-1))  # Extract corresponding time values, shape: (B, num_patches, patch_size_L)

        # 
        c_expanded = rearrange(c,'c -> 1 1 c').expand(B, self.num_patches, -1)  # (B, num_patches, C)
        c_patches = c_expanded.gather(2, idx_C.squeeze(-2))  # Extract corresponding channel values, shape: (B, num_patches, patch_size_C)
        
        # Reshape t_patches to match patches
        t_patches = rearrange(t_patches,'b n l -> b n l 1')  # (B, num_patches, patch_size_L, patch_size_C)
        c_patches = rearrange(c_patches,'b n c -> b n 1 c').expand(-1, -1, self.patch_size_L, -1)  # (B, num_patches, patch_size_L, patch_size_C)
        f_patches = rearrange(f,'f -> 1 1 f 1').expand(B, self.num_patches, -1, 1)  # (B, num_patches, patch_size_L, patch_size_C)
        # Concatenate time embedding to the patches along the channel dimension
        patches = torch.cat([patches, t_patches, c_patches], dim=-1)  # New patches shape: (B, num_patches, patch_size_L, patch_size_C + 1)
        patches =self.proj_in(patches)  # (B, num_patches, patch_size_L, output_dim)

        # Compute FFT along the patch dimension
        fft = torch.fft.rfft(patches, dim = -2, norm = 'ortho')  # (B, num_patches, patch_size_L, patch_size_C + 1)
        fft = torch.cat([fft.real, fft.imag,f_patches], dim=-1)  # (B, num_patches, patch_size_L, 2 * (patch_size_C + 1))
        fft = self.fft_proj(fft)  # (B, num_patches, patch_size_L, output_dim)
        # Flatten patches and apply linear layers
        patches = rearrange(patches, 'b p l c -> b p (l c)')  # (B * num_patches, patch_size_L * (patch_size_C + 1))
        out = self.mixer(patches)
        # out = rearrange(out, 'b d n -> b n d')

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

        model = HTFE(patch_size_L, patch_size_C, num_patches, output_dim, f_s)

        for C in C_list:
            for L in L_list:
                x = torch.randn(B, L, C)
                y = model(x)
                print(f'Input shape: {x.shape}, Output shape: {y.shape}')

    # Run the test
    test_random_patch_mixer()
