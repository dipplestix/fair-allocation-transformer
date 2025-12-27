import torch
import torch.nn as nn
import torch.nn.functional as F
from .exchangeable_layer import ExchangeableLayer
from .attention_blocks import FFSelfAttentionBlock, FFCrossAttentionBlock


class FFTransformerResidual(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_output_layers: int = 1,
                 num_encoder_layers: int = 1, dropout: float = 0.0,
                 initial_temperature: float = 1.0, final_temperature: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_output_layers = num_output_layers
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature

        pool_config = {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': []}
        self.agent_proj = ExchangeableLayer(1, d_model, pool_config=pool_config)
        self.item_proj = ExchangeableLayer(1, d_model, pool_config=pool_config)

        # Learnable residual scale - initialized small
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self.agent_transformer = nn.ModuleList(
            [FFSelfAttentionBlock(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.item_transformer = nn.ModuleList(
            [FFSelfAttentionBlock(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
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

        # Exchangeable projections
        x_agent = self.agent_proj(x)  # (B, d_model, n, m)
        x_agent = x_agent.mean(dim=3).permute(0, 2, 1)  # (B, n, d_model)

        x_item = self.item_proj(x.permute(0, 2, 1))  # (B, d_model, m, n)
        x_item = x_item.mean(dim=3).permute(0, 2, 1)  # (B, m, d_model)

        # Transformer processing
        for layer in self.agent_transformer:
            x_agent = layer(x_agent)
        for layer in self.item_transformer:
            x_item = layer(x_item)
        x_output = self.item_agent_transformer(x_item, x_agent)
        for layer in self.output_transformer:
            x_output = layer(x_output)
        x_output = self.o_norm(x_output)

        # Bilinear output + residual from input
        bilinear_out = torch.matmul(x_output, x_agent.transpose(1, 2))  # (B, m, n)
        residual = x.permute(0, 2, 1)  # (B, m, n)
        x_output = bilinear_out + self.residual_scale * residual

        x_output = F.softmax(x_output / self.temperature, dim=-1)
        return x_output
