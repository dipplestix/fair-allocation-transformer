import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components import GLU, MHA


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


class FATransformer(nn.Module):
    def __init__(self, n, m, d_model: int, num_heads: int, num_output_layers: int = 1, dropout: float = 0.0, initial_temperature: float = 1.0, final_temperature: float = 0.01):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_output_layers = num_output_layers
        self.dropout = dropout
        self.n = n
        self.m = m
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature
        
        self.agent_proj = nn.Linear(m, d_model, bias=True)
        self.item_proj = nn.Linear(n, d_model, bias=True)
        self.output_proj = nn.Linear(d_model, n, bias=True)

        self.agent_transformer = FASelfAttentionBlock(d_model, num_heads, dropout)
        self.item_transformer = FASelfAttentionBlock(d_model, num_heads, dropout)
        self.item_agent_transformer = FACrossAttentionBlock(d_model, num_heads, dropout)
        self.output_transformer = nn.ModuleList(
            [FASelfAttentionBlock(d_model, num_heads, dropout) for _ in range(num_output_layers)]
        )

        self.o_norm = nn.RMSNorm(d_model)
        
    def update_temperature(self, temperature: float):
        """Update the temperature parameter for softmax scaling"""
        self.temperature = temperature
        
    def eval(self):
        """Set model to evaluation mode and use final temperature"""
        super().eval()
        self.temperature = self.final_temperature
        return self
                
    def forward(self, x: torch.Tensor):
        B, n, m = x.shape
        

        if m < self.m:
            pad_size = self.m - m
            x_agent = torch.cat([x, torch.zeros(B, n, pad_size, device=x.device, dtype=x.dtype)], dim=2)
            m = self.m 
        else:
            x_agent = x

        x_agent = self.agent_proj(x_agent)  # expects input of shape (B, n, m)

        x_item = x.permute(0, 2, 1)
        x_item = self.item_proj(x_item)  # (B, m, n) -> (B, m, d_model)

        x_agent = self.agent_transformer(x_agent)
        x_item = self.item_transformer(x_item)
        x_output = self.item_agent_transformer(x_item, x_agent)
        for layer in self.output_transformer:
            x_output = layer(x_output)

        x_output = self.o_norm(x_output)
        x_output = self.output_proj(x_output)

        x_output = F.softmax(x_output / self.temperature, dim=-1)
        return x_output
