import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Final, Any
import argparse

# PyTorch 2.0+ specific import for fused attention
from packaging import version
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    def use_fused_attn():
        return True
else:
    def use_fused_attn():
        return False

def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, cos, sin):
    """
    Applies rotary positional embedding to the input tensor.
    
    Args:
        t (torch.Tensor): Input tensor.
        cos (torch.Tensor): Cosine component of RoPE.
        sin (torch.Tensor): Sine component of RoPE.
    
    Returns:
        torch.Tensor: Tensor with applied RoPE.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (t * cos) + (rotate_half(t) * sin)

class RotaryEmbedding(torch.nn.Module):
    """
    旋转位置嵌入 (Rotary Positional Embedding, RoPE)。
    一种将位置信息注入到Transformer的Attention层中的高效方法。
    """
    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim,
                          2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class Attention(nn.Module):
    """
    args:
        args.hidden_dim: int
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            use_fused_attn: bool = use_fused_attn(),
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()


        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.args.RoPE:
            self.rope = RotaryEmbedding(self.head_dim, max_position_embeddings=args.max_position_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.args.RoPE:
            cos, sin = self.rope(q, seq_len=N)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    # 1. 创建一个参数对象 (argparse.Namespace)
    args = argparse.Namespace()

    # 2. 设置 Attention 模块所需的参数
    args.hidden_dim = 128  # D: 特征维度
    args.num_heads = 8
    args.qkv_bias = True
    args.qk_norm = False
    args.proj_bias = True
    args.attn_drop = 0.1
    args.proj_drop = 0.1
    args.max_position_embeddings = 512
    args.RoPE = True

    # 3. 创建 Attention 模块的实例
    attention_layer = Attention(args)

    # 4. 打印模型结构
    print(attention_layer)

    # 5. 创建一个随机输入张量
    input_tensor = torch.rand(2, 60, 128)  # 示例输入，形状为 (batch_size, seq_length, hidden_dim)

    # 6. 前向传播
    output_tensor = attention_layer(input_tensor)

    # 7. 打印输出张量的形状
    print("Output tensor shape:", output_tensor.shape)  # 应该是 (2, 60, 128)