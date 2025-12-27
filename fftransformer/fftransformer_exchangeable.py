import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components import GLU, MHA
from .exchangeable_layer import ExchangeableLayer
from .attention_blocks import FFSelfAttentionBlock, FFCrossAttentionBlock


class BilinearLayer(nn.Module):
    """
    Bilinear layer that computes allocation scores from item and agent embeddings.
    
    For each item-agent pair (i, j), computes: score[i,j] = item[i]^T agent[j]
    
    Args:
        d_model: Dimension of embeddings (same for item and agent)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_item: torch.Tensor, x_agent: torch.Tensor):
        """
        Computes scores by taking the dot product of x_item and x_agent along the last dimension.

        Args:
            x_item: (B, m, d_model)
            x_agent: (B, n, d_model)

        Returns:
            (B, m, n)
        """
        # (B, m, d_model) x (B, d_model, n) -> (B, m, n)
        return torch.matmul(x_item, x_agent.transpose(1, 2))


class FFTransformerExchangeable(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_output_layers: int = 1, dropout: float = 0.0, initial_temperature: float = 1.0, final_temperature: float = 0.01):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_output_layers = num_output_layers
        self.dropout = dropout
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature
        
        self.agent_proj = ExchangeableLayer(1, d_model, pool_config={'row': ['mean', 'max', 'min'], 'column': [], 'global': []})
        self.item_proj = ExchangeableLayer(1, d_model, pool_config={'row': ['mean', 'max', 'min'], 'column': [], 'global': []})
        self.output_proj = BilinearLayer()

        self.agent_transformer = FFSelfAttentionBlock(d_model, num_heads, dropout)
        self.item_transformer = FFSelfAttentionBlock(d_model, num_heads, dropout)
        self.item_agent_transformer = FFCrossAttentionBlock(d_model, num_heads, dropout)
        self.output_transformer = nn.ModuleList(
            [FFSelfAttentionBlock(d_model, num_heads, dropout) for _ in range(num_output_layers)]
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

        # ExchangeableLayer outputs (B, d_model, H, W)
        x_agent = self.agent_proj(x)  # (B, n, m) -> (B, d_model, n, m)
        x_agent = x_agent.mean(dim=3).permute(0, 2, 1)  # (B, d_model, n, m) -> (B, n, d_model)

        x_item = x.permute(0, 2, 1)  # (B, m, n)
        x_item = self.item_proj(x_item)  # (B, m, n) -> (B, d_model, m, n)
        x_item = x_item.mean(dim=3).permute(0, 2, 1)  # (B, d_model, m, n) -> (B, m, d_model)

        # Apply transformers
        x_agent = self.agent_transformer(x_agent)  # (B, n, d_model)
        x_item = self.item_transformer(x_item)  # (B, m, d_model)
        x_output = self.item_agent_transformer(x_item, x_agent)  # (B, m, d_model)
        for layer in self.output_transformer:
            x_output = layer(x_output)

        x_output = self.o_norm(x_output)  # (B, m, d_model)
        
        x_output = self.output_proj(x_output, x_agent)  # (B, m, d_model) x (B, n, d_model) -> (B, m, n)

        x_output = F.softmax(x_output / self.temperature, dim=-1) # (B, m, n) with each row summing to 1
        return x_output
