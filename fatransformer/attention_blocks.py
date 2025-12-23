import torch.nn as nn
import torch
from .model_components import GLU, MHA


class FASelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.attn = MHA(d_model, num_heads, dropout)
        self.glu = GLU(d_model, int(8/3*d_model), d_model)
        
        self.attn_norm = nn.RMSNorm(d_model)
        self.glu_norm = nn.RMSNorm(d_model)

    
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.glu(self.glu_norm(x))
        return x
    
class FACrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.attn = MHA(d_model, num_heads, dropout)
        self.glu = GLU(d_model, int(8/3*d_model), d_model)
        
        self.attn_q_norm = nn.RMSNorm(d_model)
        self.attn_kv_norm = nn.RMSNorm(d_model) 
        self.glu_norm = nn.RMSNorm(d_model)
    
    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor):
        x = x_q + self.attn(self.attn_q_norm(x_q), self.attn_kv_norm(x_kv))
        x = x + self.glu(self.glu_norm(x))
        return x

