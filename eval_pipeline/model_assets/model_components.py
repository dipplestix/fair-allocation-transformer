import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GLU(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, activation=nn.SiLU):
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = activation()

        self.gate_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, output_dim, bias=False)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.activation(gate)

        proj = self.up_proj(x)
        
        fuse = proj*gate
        out = self.down_proj(fuse)

        return out


class MHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.proj_drop = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # (B, T, C) -> (B, H, T, D)
        return x.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self, 
        x_q: torch.Tensor, 
        x_kv: Optional[torch.Tensor] = None
    ):
        
        B, L, C = x_q.shape
        if x_kv is None:
            x_kv = x_q
        _, S, _ = x_kv.shape

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)
        
        q = self._shape(q, B, L)  # (B, H, L, D)
        k = self._shape(k, B, S)  # (B, H, S, D)
        v = self._shape(v, B, S)  # (B, H, S, D)

        y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
        )  # (B, H, L, D)

        
        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.o_proj(y)
        y = self.proj_drop(y)

        return y
