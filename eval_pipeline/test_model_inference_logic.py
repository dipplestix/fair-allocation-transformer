#!/usr/bin/env python3
"""Test model inference logic to ensure allocations are computed correctly."""

import numpy as np
import torch
import sys
sys.path.insert(0, 'eval_pipeline')
from utils.load_model import load_model

def test_inference_logic():
    """Verify the inference logic is correct"""
    print("Testing model inference logic")
    print("="*60)

    # Load model
    model, model_name = load_model('eval_pipeline/best_model_config.json')
    print(f"Loaded model: {model_name}\n")

    # Create a simple test case (use correct dimensions for the model)
    np.random.seed(42)
    n_agents, m_items = 10, 20  # Model trained for this size

    valuation_matrix = np.random.uniform(0, 1, (1, n_agents, m_items))

    print("Test valuation matrix shape:", valuation_matrix.shape)
    print("Valuation matrix:")
    print(valuation_matrix[0])

    # Get model output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valuation_tensor = torch.tensor(valuation_matrix, dtype=torch.float32).to(device)

    with torch.no_grad():
        model.eval()
        output = model(valuation_tensor)

    print(f"\nModel output shape: {output.shape}")
    print(f"Expected shape: (batch=1, m_items={m_items}, n_agents={n_agents})")

    # The model should output (batch, m_items, n_agents)
    # For each item, we want to know which agent gets it
    # So we take argmax over the agent dimension (dim=2)

    print("\nModel output (item assignments):")
    print(output[0].cpu().numpy())

    # Take argmax over agents dimension (dim=2)
    max_indices = torch.argmax(output, dim=2)
    print(f"\nArgmax over agents (dim=2): {max_indices}")
    print(f"Shape: {max_indices.shape}")

    # Convert to allocation matrix (n_agents, m_items)
    allocation_matrix = torch.zeros((1, n_agents, m_items))
    allocation_matrix.scatter_(1, max_indices.unsqueeze(1), 1)

    print(f"\nAfter scatter (should be n_agents x m_items):")
    print(f"Shape: {allocation_matrix.shape}")
    print("Allocation matrix:")
    print(allocation_matrix[0].cpu().numpy())

    # Transpose to (n_agents, m_items)
    allocation_matrix = allocation_matrix.transpose(1, 2)
    print(f"\nAfter transpose to (n_agents, m_items):")
    print(f"Shape: {allocation_matrix.shape}")
    print("Final allocation matrix:")
    final_alloc = allocation_matrix[0].cpu().numpy()
    print(final_alloc)

    # Verify each item assigned to exactly one agent
    col_sums = np.sum(final_alloc, axis=0)
    row_sums = np.sum(final_alloc, axis=1)

    print(f"\nColumn sums (should all be 1): {col_sums}")
    print(f"Row sums (items per agent): {row_sums}")

    if np.all(col_sums == 1):
        print("✓ Each item assigned to exactly one agent")
    else:
        print("⚠ ERROR: Invalid allocation!")

    # Now let me trace through what the actual inference code does
    print("\n" + "="*60)
    print("Tracing through actual inference.py code:")
    print("="*60)

    from utils.inference import get_model_allocations_batch

    allocation_from_function = get_model_allocations_batch(model, valuation_matrix)
    print(f"\nAllocation from get_model_allocations_batch:")
    print(f"Shape: {allocation_from_function.shape}")
    print(allocation_from_function[0])

    col_sums_fn = np.sum(allocation_from_function[0], axis=0)
    print(f"\nColumn sums: {col_sums_fn}")

    if np.all(col_sums_fn == 1):
        print("✓ Function produces valid allocation")
    else:
        print("⚠ ERROR: Function produces invalid allocation!")

    print("\n" + "="*60)

if __name__ == '__main__':
    test_inference_logic()
