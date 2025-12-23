import torch.nn as nn
import torch
from fatransformer.model_components import GLU, MHA
from typing import Optional

class MAB(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, cross_attn: bool = False):
        super().__init__()
        self.cross_attn = cross_attn

        self.attn = MHA(d_model, num_heads, dropout)
        self.glu = GLU(d_model, int(8/3*d_model), d_model)

        self.attn_q_norm = nn.RMSNorm(d_model)
        if cross_attn:
            self.attn_kv_norm = nn.RMSNorm(d_model)
        else:
            self.attn_kv_norm = None
        self.glu_norm = nn.RMSNorm(d_model)

    def forward(
        self, 
        x_q: torch.Tensor, 
        x_kv: Optional[torch.Tensor] = None
    ):
        if self.cross_attn:
            h = x_q + self.attn(self.attn_q_norm(x_q), self.attn_kv_norm(x_kv))
        else:
            h = x_q + self.attn(self.attn_q_norm(x_q))
        
        x_q = x_q + self.glu(self.glu_norm(h))
        return x_q
        

class SAB(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(d_model, num_heads, dropout)

    def forward(self, x: torch.Tensor):
        return self.mab(x)
    

class ISAB(nn.Module):
    def __init__(self, d_model: int, num_heads: int, m: int, dropout: float = 0.0):
        super().__init__()
        self.induced = MAB(d_model, num_heads, dropout, True)
        self.mab = MAB(d_model, num_heads, dropout, True)

        self.I = nn.Parameter(torch.empty(1, m, d_model))
        nn.init.xavier_uniform_(self.I)

    def forward(self, x: torch.Tensor):
        I = self.I.expand(x.size(0), -1, -1)

        h = self.induced(I, x)
        out = self.mab(x, h)
        return out
    

class PMA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, k: int, dropout: float = 0.0):
        super().__init__()
        self.pma = MAB(d_model, num_heads, dropout, True)
        self.rff = GLU(d_model, int(8/3*d_model), d_model)
        self.rff_norm = nn.RMSNorm(d_model)

        self.S = nn.Parameter(torch.empty(k, d_model))
        nn.init.xavier_uniform_(self.S)


    def forward(self, z: torch.Tensor):
        S = self.S.expand(z.size(0), -1, -1)

        z = z + self.rff(self.rff_norm(z))
        out = self.pma(S, z)
        return out
