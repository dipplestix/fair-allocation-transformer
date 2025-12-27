import torch.nn as nn
import torch
from .attention_blocks import FFSelfAttentionBlock


class PoolLayer(nn.Module):
    """
    Pool layer that applies different aggregation functions across rows, columns, or the whole matrix
    and adds the expanded results as new channels to the input.
    
    Args:
        pool_config: Dictionary mapping pooling modes to aggregation functions.
                    Keys: 'row', 'column', 'global'
                    Values: 'mean', 'min', 'max', or list of these (e.g., ['mean', 'max'])
                    Example: {'global': ['mean', 'max'], 'row': 'min', 'column': ['mean', 'min', 'max']}
    """
    def __init__(self, pool_config: dict = {'global': 'mean'}):
        super().__init__()
        self.pool_config = pool_config
        
        # Validate pool config
        valid_modes = ['row', 'column', 'global']
        valid_aggs = ['mean', 'min', 'max']
        
        for mode, agg in pool_config.items():
            if mode not in valid_modes:
                raise ValueError(f"pool_mode '{mode}' must be one of {valid_modes}")
            
            # Handle both single aggregation and list of aggregations
            if isinstance(agg, str):
                if agg not in valid_aggs:
                    raise ValueError(f"aggregation '{agg}' must be one of {valid_aggs}")
            elif isinstance(agg, list):
                for single_agg in agg:
                    if single_agg not in valid_aggs:
                        raise ValueError(f"aggregation '{single_agg}' must be one of {valid_aggs}")
            else:
                raise ValueError(f"aggregation must be a string or list of strings, got {type(agg)}")
        
        # Store modes for easy access
        self.pool_modes = list(pool_config.keys())
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the pool layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width) or (batch_size, height, width)
        
        Returns:
            Tensor with original channels plus new pooled channels for each input channel. Output shape:
            - For 3D input: (batch_size, 1 + total_aggregations, height, width)
            - For 4D input: (batch_size, channels + channels * total_aggregations, height, width)
            where total_aggregations = sum of number of aggregation functions for each mode
        """
        original_shape = x.shape
        
        # Ensure input is 4D for consistent processing
        if len(original_shape) == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # Add channel dimension: (B, 1, H, W)
            input_channels = 1
        elif len(original_shape) == 4:  # (B, C, H, W)
            input_channels = original_shape[1]
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        batch_size, channels, height, width = x.shape
        
        # Collect all pooled channels
        pooled_channels = []
        
        for mode, agg_funcs in self.pool_config.items():
            # Handle both single aggregation and list of aggregations
            if isinstance(agg_funcs, str):
                agg_funcs = [agg_funcs]
            
            for agg_func in agg_funcs:
                if mode == 'global':
                    # Aggregate across the entire matrix for each channel separately
                    # Aggregate over spatial dimensions (H, W) for each channel
                    if agg_func == 'mean':
                        agg_values = x.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
                    elif agg_func == 'min':
                        # Reduce one dimension at a time for min
                        agg_values = x.min(dim=2, keepdim=True)[0]  # Shape: (B, C, 1, W)
                        agg_values = agg_values.min(dim=3, keepdim=True)[0]  # Shape: (B, C, 1, 1)
                    elif agg_func == 'max':
                        # Reduce one dimension at a time for max
                        agg_values = x.max(dim=2, keepdim=True)[0]  # Shape: (B, C, 1, W)
                        agg_values = agg_values.max(dim=3, keepdim=True)[0]  # Shape: (B, C, 1, 1)
                    # Expand to original spatial shape for each channel
                    pooled_channel = agg_values.expand(batch_size, channels, height, width)
                    
                elif mode == 'row':
                    # Aggregate across rows (width dimension) for each channel separately
                    if agg_func == 'mean':
                        agg_values = x.mean(dim=3, keepdim=True)  # Shape: (B, C, H, 1)
                    elif agg_func == 'min':
                        agg_values = x.min(dim=3, keepdim=True)[0]  # Shape: (B, C, H, 1)
                    elif agg_func == 'max':
                        agg_values = x.max(dim=3, keepdim=True)[0]  # Shape: (B, C, H, 1)
                    # Expand to original spatial shape for each channel
                    pooled_channel = agg_values.expand(batch_size, channels, height, width)
                    
                elif mode == 'column':
                    # Aggregate across columns (height dimension) for each channel separately
                    if agg_func == 'mean':
                        agg_values = x.mean(dim=2, keepdim=True)  # Shape: (B, C, 1, W)
                    elif agg_func == 'min':
                        agg_values = x.min(dim=2, keepdim=True)[0]  # Shape: (B, C, 1, W)
                    elif agg_func == 'max':
                        agg_values = x.max(dim=2, keepdim=True)[0]  # Shape: (B, C, 1, W)
                    # Expand to original spatial shape for each channel
                    pooled_channel = agg_values.expand(batch_size, channels, height, width)
                
                pooled_channels.append(pooled_channel)
        
        # Concatenate original channels with pooled channels
        if pooled_channels:
            pooled_tensor = torch.cat(pooled_channels, dim=1)  # Concatenate along channel dimension
            output = torch.cat([x, pooled_tensor], dim=1)
        else:
            output = x
        
        return output



class ExchangeableLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool_config: dict = {'row': 'mean', 'column': 'mean', 'global': 'mean'},
                 activation: nn.Module = nn.GELU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate total number of aggregation functions
        total_aggs = 0
        for mode, agg_funcs in pool_config.items():
            if isinstance(agg_funcs, str):
                total_aggs += 1
            elif isinstance(agg_funcs, list):
                total_aggs += len(agg_funcs)

        proj_in_channels = in_channels + total_aggs * in_channels

        self.pool_layer = PoolLayer(pool_config)
        self.proj = nn.Conv2d(proj_in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the exchangeable layer.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor of shape (batch_size, out_channels, height, width)
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.pool_layer(x)  # Shape: (B, C + C*total_aggs, n, m)
        x = self.activation(self.proj(x))  # Shape: (B, out_channels, n, m)
        return x
    

class AxisAttnPool1D(nn.Module):
    """
    Reduces the width axis (W) of a (B, D, H, W) tensor to (B, H, D) via learned attention.
    Permutation-equivariant over the reduced axis and length-agnostic.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm  = nn.RMSNorm(d_model)
        self.score = nn.Linear(d_model, 1, bias=False)  # per-element score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        s = self.score(x).squeeze(-1)             # (B, H, W)
        a = s.softmax(dim=2)                      # softmax over width axis
        pooled = (a.unsqueeze(-1) * x).sum(dim=2) # (B, H, D)
        return pooled


class AxisSelfAttention1D(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.block = FFSelfAttentionBlock(d_model, num_heads, dropout)

    def forward(self, x):                  # x: (B, D, H, W)
        B, D, H, W = x.shape
        X = x.permute(0, 2, 3, 1).reshape(B*H, W, D)   # (B*H, W, D)
        X = self.block(X)                              # set-level mixing along W
        X = X.view(B, H, W, D).permute(0, 3, 1, 2)     # (B, D, H, W)
        return X

class AxisAdditivePool1D(nn.Module):
    """
    Permutation-invariant, length-agnostic pooling with signed weights.
    Returns (B, H, D) from (B, D, H, W).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, 1, bias=False),   # no softmax → signed weights
        )

    def forward(self, x):
        # x: (B, D, H, W)
        B, D, H, W = x.shape
        z = self.norm(x.permute(0, 2, 3, 1))   # (B, H, W, D)
        w = self.scorer(z).squeeze(-1)         # (B, H, W), signed
        # variance-preserving normalization across set size
        pooled = (w.unsqueeze(-1) * z).sum(dim=2) / (W**0.5 + 1e-6)  # (B, H, D)
        return pooled

