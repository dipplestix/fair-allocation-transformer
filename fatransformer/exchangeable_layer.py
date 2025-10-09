import torch.nn as nn
import torch


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
            x: Input tensor of shape (batch_size, height, width, channels) or (batch_size, height, width)
        
        Returns:
            Tensor with original channels plus new pooled channels for each input channel. Output shape:
            - For 3D input: (batch_size, height, width, 1 + total_aggregations)
            - For 4D input: (batch_size, height, width, channels + channels * total_aggregations)
            where total_aggregations = sum of number of aggregation functions for each mode
        """
        original_shape = x.shape
        
        # Ensure input is 4D for consistent processing
        if len(original_shape) == 3:  # (B, H, W)
            x = x.unsqueeze(-1)  # Add channel dimension: (B, H, W, 1)
            input_channels = 1
        elif len(original_shape) == 4:  # (B, H, W, C)
            input_channels = original_shape[3]
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        batch_size, height, width, channels = x.shape
        
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
                        agg_values = x.mean(dim=(1, 2), keepdim=True)  # Shape: (B, 1, 1, C)
                    elif agg_func == 'min':
                        # Reduce one dimension at a time for min
                        agg_values = x.min(dim=1, keepdim=True)[0]  # Shape: (B, 1, W, C)
                        agg_values = agg_values.min(dim=2, keepdim=True)[0]  # Shape: (B, 1, 1, C)
                    elif agg_func == 'max':
                        # Reduce one dimension at a time for max
                        agg_values = x.max(dim=1, keepdim=True)[0]  # Shape: (B, 1, W, C)
                        agg_values = agg_values.max(dim=2, keepdim=True)[0]  # Shape: (B, 1, 1, C)
                    # Expand to original spatial shape for each channel
                    pooled_channel = agg_values.expand(batch_size, height, width, channels)
                    
                elif mode == 'row':
                    # Aggregate across rows (width dimension) for each channel separately
                    if agg_func == 'mean':
                        agg_values = x.mean(dim=2, keepdim=True)  # Shape: (B, H, 1, C)
                    elif agg_func == 'min':
                        agg_values = x.min(dim=2, keepdim=True)[0]  # Shape: (B, H, 1, C)
                    elif agg_func == 'max':
                        agg_values = x.max(dim=2, keepdim=True)[0]  # Shape: (B, H, 1, C)
                    # Expand to original spatial shape for each channel
                    pooled_channel = agg_values.expand(batch_size, height, width, channels)
                    
                elif mode == 'column':
                    # Aggregate across columns (height dimension) for each channel separately
                    if agg_func == 'mean':
                        agg_values = x.mean(dim=1, keepdim=True)  # Shape: (B, 1, W, C)
                    elif agg_func == 'min':
                        agg_values = x.min(dim=1, keepdim=True)[0]  # Shape: (B, 1, W, C)
                    elif agg_func == 'max':
                        agg_values = x.max(dim=1, keepdim=True)[0]  # Shape: (B, 1, W, C)
                    # Expand to original spatial shape for each channel
                    pooled_channel = agg_values.expand(batch_size, height, width, channels)
                
                pooled_channels.append(pooled_channel)
        
        # Concatenate original channels with pooled channels
        if pooled_channels:
            pooled_tensor = torch.cat(pooled_channels, dim=-1)  # Concatenate along channel dimension
            output = torch.cat([x, pooled_tensor], dim=-1)
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

        self.pool_layer = PoolLayer(pool_config)
        
        # Calculate total number of aggregation functions
        total_aggs = 0
        for mode, agg_funcs in pool_config.items():
            if isinstance(agg_funcs, str):
                total_aggs += 1
            elif isinstance(agg_funcs, list):
                total_aggs += len(agg_funcs)
        
        self.lin_in = in_channels + total_aggs*in_channels
        self.proj = nn.Linear(in_channels + total_aggs*in_channels, out_channels, bias=True)
        


    def forward(self, x: torch.Tensor):
        B, n, m, _ = x.shape
        x = self.pool_layer(x)
        x = x.view(B, n*m, self.lin_in)
        x = self.activation(self.proj(x))
        x = x.view(B, n, m, self.out_channels)
        return x
    
