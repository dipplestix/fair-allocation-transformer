"""
Compare NeuralRR baseline against FairFormer model.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import torch
from fftransformer.fftransformer_residual import FFTransformerResidual
from baselines.neural_rr_adapter import NeuralRRAdapter
from eval_pipeline.utils.inference import get_model_allocations_batch


def compute_nash_welfare(valuations, allocations):
    """Compute Nash welfare for allocations."""
    utilities = (valuations * allocations).sum(axis=-1)  # (batch, n_agents)
    # Avoid log(0) by adding small epsilon
    log_utilities = np.log(utilities + 1e-10)
    nash = np.exp(log_utilities.mean(axis=-1))  # (batch,)
    return nash.mean()


def compute_utilitarian_welfare(valuations, allocations):
    """Compute utilitarian (sum) welfare for allocations."""
    utilities = (valuations * allocations).sum(axis=-1)  # (batch, n_agents)
    return utilities.sum(axis=-1).mean()


def load_fairformer_10_20(device):
    """Load the 10x20 FairFormer model."""
    model = FFTransformerResidual(
        d_model=768,
        num_heads=4,
        num_output_layers=4,
        num_encoder_layers=3,
        dropout=0.071,
        initial_temperature=1.77,
        final_temperature=0.054,
    )
    weights_path = project_root / "checkpoints" / "residual" / "from_sweep_pt1hesbo" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_agents, m_items = 10, 20
    n_samples = 1000

    # Generate test data
    print(f"\nGenerating {n_samples} test instances ({n_agents}x{m_items})...")
    np.random.seed(42)
    valuations = np.random.rand(n_samples, n_agents, m_items).astype(np.float32)

    # Load FairFormer model
    print("\nLoading FairFormer model...")
    fairformer = load_fairformer_10_20(device)

    # Load NeuralRR model
    print("Loading NeuralRR model...")
    checkpoint_path = project_root / "checkpoints" / "neural_rr" / "neural_rr_10x20.pt"
    neural_rr = NeuralRRAdapter(checkpoint_path=str(checkpoint_path), device=device)

    # Get FairFormer allocations
    print("\nRunning FairFormer inference...")
    fairformer_allocs = get_model_allocations_batch(fairformer, valuations)

    # Get NeuralRR allocations
    print("Running NeuralRR inference...")
    neural_rr_allocs = neural_rr.get_allocations_batch(valuations, batch_size=100)

    # Compute metrics
    print("\n" + "="*60)
    print("RESULTS: FairFormer vs NeuralRR (10x20)")
    print("="*60)

    ff_nash = compute_nash_welfare(valuations, fairformer_allocs)
    ff_util = compute_utilitarian_welfare(valuations, fairformer_allocs)

    nrr_nash = compute_nash_welfare(valuations, neural_rr_allocs)
    nrr_util = compute_utilitarian_welfare(valuations, neural_rr_allocs)

    print(f"\nFairFormer (77M params):")
    print(f"  Nash Welfare:       {ff_nash:.4f}")
    print(f"  Utilitarian Welfare: {ff_util:.4f}")

    print(f"\nNeuralRR (141 params):")
    print(f"  Nash Welfare:       {nrr_nash:.4f}")
    print(f"  Utilitarian Welfare: {nrr_util:.4f}")

    print(f"\nRelative difference (FairFormer vs NeuralRR):")
    nash_diff = (ff_nash - nrr_nash) / nrr_nash * 100
    util_diff = (ff_util - nrr_util) / nrr_util * 100
    print(f"  Nash:       {nash_diff:+.2f}%")
    print(f"  Utilitarian: {util_diff:+.2f}%")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
