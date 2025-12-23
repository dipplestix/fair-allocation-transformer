"""Fair Allocation Transformer package

This package implements transformer-based architectures for learning fair allocations
in discrete fair division problems.
"""

# Main models
from .fatransformer import FATransformer
from .fatransformer_exchangeable import FATransformer as FATransformerExchangeable

# Core components
from .attention_blocks import FASelfAttentionBlock, FACrossAttentionBlock
from .model_components import GLU, MHA
from .exchangeable_layer import (
    ExchangeableLayer,
    PoolLayer,
    AxisAttnPool1D,
    AxisSelfAttention1D,
    AxisAdditivePool1D,
)

# Utilities
from .helpers import get_nash_welfare

__all__ = [
    # Models
    "FATransformer",
    "FATransformerExchangeable",
    # Attention blocks
    "FASelfAttentionBlock",
    "FACrossAttentionBlock",
    # Components
    "GLU",
    "MHA",
    "ExchangeableLayer",
    "PoolLayer",
    "AxisAttnPool1D",
    "AxisSelfAttention1D",
    "AxisAdditivePool1D",
    # Helpers
    "get_nash_welfare",
]

__version__ = "0.1.0"
