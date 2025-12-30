"""
Adapter for NeuralRR baseline model.

NeuralRR is a neural network approach to round-robin allocation that learns
agent orderings using SVD + MLP + SoftSort + SoftRR.

Reference: https://github.com/MandR1215/neural_rr
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add neural_rr directory to path for imports
_neural_rr_path = Path(__file__).parent / "neural_rr"
sys.path.insert(0, str(_neural_rr_path))

# Now import from neural_rr
from neural_rr import NeuralRR
from softsort import SoftSort
from layers import SoftRR


class NeuralRRAdapter:
    """Adapter to use NeuralRR models with our evaluation pipeline."""

    def __init__(self, checkpoint_path=None, device=None):
        """
        Initialize NeuralRR adapter.

        Args:
            checkpoint_path: Path to trained model checkpoint (optional)
            device: Device to run on (defaults to CUDA if available)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Default hyperparameters from hp_nrr.json
        self.model = NeuralRR(
            softsort_tau=0.1,
            softsort_pow=2.0,
            srr_tau=0.01
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.to(device)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def get_allocations(self, valuations):
        """
        Get allocations for a batch of valuation matrices.

        Args:
            valuations: numpy array of shape (batch, n_agents, m_items)
                       or (n_agents, m_items) for single instance

        Returns:
            allocations: numpy array of shape (batch, n_agents, m_items)
                        Binary allocation matrix
        """
        single_instance = valuations.ndim == 2
        if single_instance:
            valuations = valuations[np.newaxis, ...]

        batch_size, n_agents, m_items = valuations.shape

        # Convert to tensor
        V = torch.tensor(valuations, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # NeuralRR outputs pi_hat of shape (batch, n, m)
            # This is a soft allocation where pi_hat[i,j] = probability agent i gets item j
            pi_hat = self.model(V)

            # Convert soft allocation to hard allocation using argmax
            # For each item (column), assign to agent with highest probability
            allocations = torch.zeros_like(pi_hat)
            max_indices = torch.argmax(pi_hat, dim=1)  # (batch, m)

            for b in range(batch_size):
                for j in range(m_items):
                    allocations[b, max_indices[b, j], j] = 1.0

        result = allocations.cpu().numpy()

        if single_instance:
            result = result[0]

        return result

    def get_allocations_batch(self, valuations, batch_size=100):
        """
        Get allocations for a large batch, processing in smaller chunks.

        Args:
            valuations: numpy array of shape (N, n_agents, m_items)
            batch_size: Size of chunks to process

        Returns:
            allocations: numpy array of shape (N, n_agents, m_items)
        """
        n_samples = len(valuations)
        all_allocations = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_valuations = valuations[i:batch_end]
            batch_allocations = self.get_allocations(batch_valuations)
            all_allocations.append(batch_allocations)

        return np.concatenate(all_allocations, axis=0)


def get_neural_rr_allocations_batch(valuations, checkpoint_path=None, batch_size=100):
    """
    Convenience function to get NeuralRR allocations.

    Args:
        valuations: numpy array of shape (N, n_agents, m_items)
        checkpoint_path: Path to trained model checkpoint
        batch_size: Batch size for processing

    Returns:
        allocations: numpy array of shape (N, n_agents, m_items)
    """
    adapter = NeuralRRAdapter(checkpoint_path=checkpoint_path)
    return adapter.get_allocations_batch(valuations, batch_size=batch_size)


def train_neural_rr(n_agents, m_items, num_samples=1000, epochs=20,
                    save_dir="checkpoints/neural_rr", device=None):
    """
    Train a NeuralRR model for a specific problem size.

    Args:
        n_agents: Number of agents
        m_items: Number of items
        num_samples: Number of training samples per epoch
        epochs: Number of training epochs
        save_dir: Directory to save checkpoints
        device: Device to train on

    Returns:
        model: Trained NeuralRR model
        checkpoint_path: Path to saved checkpoint
    """
    import sys
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training NeuralRR on {device}", flush=True)

    # Create model
    model = NeuralRR(
        softsort_tau=0.1,
        softsort_pow=2.0,
        srr_tau=0.01
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}", flush=True)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    batch_size = 16  # Smaller batch for faster iteration

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = num_samples // batch_size

        for batch_idx in range(num_batches):
            # Generate random valuations
            V = torch.rand(batch_size, n_agents, m_items, device=device)

            # Get target allocation (MaxUtil - each item to highest-valuing agent)
            with torch.no_grad():
                target = torch.zeros_like(V)
                max_agents = torch.argmax(V, dim=1)  # (batch, m)
                for b in range(batch_size):
                    for j in range(m_items):
                        target[b, max_agents[b, j], j] = 1.0

            # Forward pass
            optimizer.zero_grad()
            pi_hat = model(V)

            # Item cross-entropy loss
            loss = -torch.sum(target * torch.log(pi_hat + 1e-9), dim=1).mean()

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)
        sys.stdout.flush()

    # Save checkpoint
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"neural_rr_{n_agents}x{m_items}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}", flush=True)

    model.eval()
    return model, checkpoint_path


if __name__ == "__main__":
    # Test the adapter
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--n", type=int, default=10, help="Number of agents")
    parser.add_argument("--m", type=int, default=20, help="Number of items")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--test", action="store_true", help="Test the adapter")
    args = parser.parse_args()

    if args.train:
        print(f"Training NeuralRR for {args.n} agents x {args.m} items...")
        model, checkpoint = train_neural_rr(args.n, args.m, epochs=args.epochs)
        print(f"Training complete. Checkpoint: {checkpoint}")

    if args.test:
        print("Testing NeuralRR adapter...")
        adapter = NeuralRRAdapter()

        # Generate random test data
        valuations = np.random.rand(10, args.n, args.m).astype(np.float32)
        allocations = adapter.get_allocations(valuations)

        print(f"Input shape: {valuations.shape}")
        print(f"Output shape: {allocations.shape}")
        print(f"Each item allocated to exactly one agent: {np.allclose(allocations.sum(axis=1), 1)}")
