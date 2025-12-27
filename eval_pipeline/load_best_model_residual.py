#!/usr/bin/env python
"""Simple model loader for evaluation of FATransformerResidual that uses standard Python imports."""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fatransformer.fatransformer_residual import FATransformer as FATransformerResidual


def load_model(
    checkpoint_path,
    n=10,
    m=20,
    d_model=256,
    num_heads=8,
    num_output_layers=2,
    dropout=0.0,
    initial_temperature=1.0,
    final_temperature=0.01,
):
    """Load FATransformerResidual model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        n: Number of agents
        m: Number of items
        d_model: Model dimension
        num_heads: Number of attention heads
        num_output_layers: Number of output layers
        dropout: Dropout rate
        initial_temperature: Initial temperature (not used in eval)
        final_temperature: Final temperature (used in eval mode)

    Returns:
        model: Loaded FATransformerResidual model in eval mode
        model_name: Name for the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = FATransformerResidual(
        n=n,
        m=m,
        d_model=d_model,
        num_heads=num_heads,
        num_output_layers=num_output_layers,
        dropout=dropout,
        initial_temperature=initial_temperature,
        final_temperature=final_temperature
    )

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    model_name = Path(checkpoint_path).stem

    return model, model_name


if __name__ == "__main__":
    # Test loading
    model, name = load_model("../checkpoints/residual/best_model.pt")
    print(f"Model {name} loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Residual scale: {model.residual_scale.item():.4f}")
