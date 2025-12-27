#!/usr/bin/env python
"""Simple model loader for evaluation that uses standard Python imports."""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fatransformer.fatransformer import FATransformer

def load_model(checkpoint_path, n=10, m=20, d_model=768, num_heads=16, num_output_layers=5, dropout=0.008020981126192437):
    """Load FATransformer model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        n: Number of agents
        m: Number of items
        d_model: Model dimension
        num_heads: Number of attention heads
        num_output_layers: Number of output layers
        dropout: Dropout rate

    Returns:
        model: Loaded FATransformer model in eval mode
        model_name: Name for the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = FATransformer(
        n=n,
        m=m,
        d_model=d_model,
        num_heads=num_heads,
        num_output_layers=num_output_layers,
        dropout=dropout,
        initial_temperature=1.0,  # Not used in eval
        final_temperature=0.01    # Used in eval mode
    )

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    model_name = Path(checkpoint_path).stem

    return model, model_name

if __name__ == "__main__":
    # Test loading
    model, name = load_model("../checkpoints/best_model.pt")
    print(f"Model {name} loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
