"""
PrefLib Data Conversion Module

This module provides functionality to parse PrefLib .soi (Strict Order - Incomplete)
files and convert ordinal rankings to cardinal valuations for fair allocation problems.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import re


def parse_soi_file(file_path: str) -> Tuple[int, int, List[Tuple[int, List[int]]]]:
    """
    Parse a PrefLib .soi file and extract preference data.

    Args:
        file_path: Path to the .soi file

    Returns:
        Tuple of (num_items, num_voters, rankings) where:
        - num_items: Number of alternatives/items in the election
        - num_voters: Total number of voters/agents
        - rankings: List of (count, ranking) tuples where:
            - count: Number of agents with this ranking
            - ranking: List of item IDs in preference order (1-indexed)
    """
    rankings = []
    num_items = 0
    num_voters = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                # Parse header information from comments
                if '# NUMBER ALTERNATIVES:' in line:
                    num_items = int(line.split(':')[1].strip())
                elif '# NUMBER VOTERS:' in line:
                    num_voters = int(line.split(':')[1].strip())
                continue

            # Parse ranking lines (format: "count: item1, item2, item3, ...")
            if ':' in line:
                parts = line.split(':', 1)
                count = int(parts[0].strip())

                # Parse item rankings
                items_str = parts[1].strip()
                if items_str:
                    # Handle both comma-separated and space-separated formats
                    items = [int(x.strip()) for x in re.split(r'[,\s]+', items_str) if x.strip()]
                    rankings.append((count, items))

    return num_items, num_voters, rankings


def ranking_to_valuation(ranking: List[int], n_items: int, method: str = 'borda') -> np.ndarray:
    """
    Convert an ordinal ranking to cardinal valuation using specified method.
    All methods are normalized to [0, 1] range.

    Args:
        ranking: List of item IDs in preference order (1-indexed, most preferred first)
        n_items: Total number of items
        method: Conversion method ('borda' or 'linear')
            - 'borda': Normalized Borda count ((n_items - rank) / n_items)
            - 'linear': Linear decay (1.0 - rank/n_items)

    Returns:
        Valuation array of shape (n_items,) with values in [0, 1] (0-indexed)
    """
    valuation = np.zeros(n_items)

    if method == 'borda':
        for rank, item_id in enumerate(ranking):
            if 1 <= item_id <= n_items:
                # Normalized Borda: most preferred gets 1.0, least ranked gets 1/n_items
                valuation[item_id - 1] = (n_items - rank) / n_items

    elif method == 'linear':
        base = 20
        for rank, item_id in enumerate(ranking):
            if 1 <= item_id <= n_items:
                # Linear decay: most preferred gets 1.0, linearly decreasing
                valuation[item_id - 1] = 1.0 - (rank / base)

    else:
        raise ValueError(f"Unknown conversion method: {method}. Use 'borda' or 'linear'.")

    return valuation


def extract_complete_rankings(
    rankings: List[Tuple[int, List[int]]],
    n_items: int,
    required_items: int
) -> List[List[int]]:
    """
    Extract all agent rankings that include at least the required number of items.

    Args:
        rankings: List of (count, ranking) tuples from parse_soi_file
        n_items: Total number of items in the dataset
        required_items: Minimum number of items that must be ranked

    Returns:
        List of complete rankings (each is a list of item IDs)
    """
    complete_rankings = []

    for count, ranking in rankings:
        # Only include rankings with at least required_items
        if len(ranking) >= required_items:
            # Replicate the ranking 'count' times (since multiple agents have same ranking)
            for _ in range(count):
                complete_rankings.append(ranking[:required_items])  # Take first required_items

    return complete_rankings


def create_valuation_matrices(
    complete_rankings: List[List[int]],
    n_agents: int,
    n_items: int,
    num_matrices: int,
    method: str = 'borda',
    seed: int = 42
) -> List[np.ndarray]:
    """
    Create multiple UNIQUE valuation matrices using deterministic combinations.
    Uses itertools.combinations to generate unique agent sets, then shuffles
    the order based on seed for randomization while maintaining uniqueness.

    Args:
        complete_rankings: List of complete rankings
        n_agents: Number of agents per matrix
        n_items: Number of items
        num_matrices: Number of matrices to generate
        method: Valuation conversion method ('borda' or 'linear')
        seed: Random seed for reproducibility (shuffles combination order)

    Returns:
        List of valuation matrices, each of shape (n_agents, n_items)

    Raises:
        ValueError: If not enough complete rankings or combinations available
    """
    from math import comb
    from itertools import combinations
    import random

    # Check minimum requirement
    if len(complete_rankings) < n_agents:
        raise ValueError(
            f"Not enough complete rankings ({len(complete_rankings)}) "
            f"to create matrices with {n_agents} agents"
        )

    # Calculate maximum unique combinations
    max_combinations = comb(len(complete_rankings), n_agents)

    if num_matrices > max_combinations:
        raise ValueError(
            f"Requested {num_matrices} matrices but only {max_combinations:,} "
            f"unique combinations possible with {len(complete_rankings)} rankings "
            f"and {n_agents} agents per matrix"
        )

    # Generate all possible combinations and select first num_matrices
    # Use combinations() which is deterministic and memory-efficient
    all_indices = range(len(complete_rankings))
    combo_generator = combinations(all_indices, n_agents)

    # Take first num_matrices combinations
    selected_combinations = []
    for i, combo in enumerate(combo_generator):
        if i >= num_matrices:
            break
        selected_combinations.append(combo)

    # Shuffle the selected combinations based on seed for randomization
    # while maintaining uniqueness and reproducibility
    random.seed(seed)
    random.shuffle(selected_combinations)

    # Create matrices from combinations
    matrices = []
    np.random.seed(seed)  # Also seed numpy for any internal operations

    for combination in selected_combinations:
        # Convert each ranking to valuation
        matrix = np.zeros((n_agents, n_items))
        for agent_idx, ranking_idx in enumerate(combination):
            ranking = complete_rankings[ranking_idx]
            matrix[agent_idx] = ranking_to_valuation(ranking, n_items, method)

        matrices.append(matrix)

    return matrices


def load_preflib_dataset(
    dataset_dir: str,
    n_agents: int,
    n_items: int,
    num_matrices: int,
    method: str = 'borda',
    seed: int = 42
) -> List[np.ndarray]:
    """
    Load and process a PrefLib dataset directory.

    Args:
        dataset_dir: Path to dataset directory containing .soi files
        n_agents: Number of agents per matrix
        n_items: Number of items
        num_matrices: Number of matrices to generate
        method: Valuation conversion method
        seed: Random seed

    Returns:
        List of valuation matrices
    """
    dataset_path = Path(dataset_dir)

    # Find all .soi files in the directory
    soi_files = list(dataset_path.glob("*.soi"))

    if not soi_files:
        raise ValueError(f"No .soi files found in {dataset_dir}")

    # For now, use the first .soi file
    # (In the future, this could aggregate across multiple files)
    soi_file = soi_files[0]

    print(f"Loading PrefLib data from: {soi_file}")

    # Parse the file
    total_items, total_voters, rankings = parse_soi_file(str(soi_file))
    print(f"Dataset info: {total_items} items, {total_voters} voters")

    # Extract complete rankings
    complete_rankings = extract_complete_rankings(rankings, total_items, n_items)
    print(f"Found {len(complete_rankings)} complete rankings with at least {n_items} items")

    if len(complete_rankings) < n_agents:
        raise ValueError(
            f"Insufficient complete rankings ({len(complete_rankings)}) "
            f"for {n_agents} agents. Try reducing n_agents or n_items."
        )

    # Calculate and report maximum unique combinations
    from math import comb
    max_unique_combinations = comb(len(complete_rankings), n_agents)
    print(f"Maximum unique matrices possible: {max_unique_combinations:,}")

    if num_matrices > max_unique_combinations:
        raise ValueError(
            f"Cannot generate {num_matrices:,} unique matrices. "
            f"Maximum possible: {max_unique_combinations:,} "
            f"(C({len(complete_rankings)}, {n_agents}))"
        )

    # Create valuation matrices
    matrices = create_valuation_matrices(
        complete_rankings, n_agents, n_items, num_matrices, method, seed
    )

    return matrices


def get_dataset_name(dataset_dir: str) -> str:
    """
    Extract dataset name from directory path.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        Dataset name (e.g., "french-irv-2007" from "00072_french-irv-2007")
    """
    dataset_path = Path(dataset_dir)
    dir_name = dataset_path.name

    # Remove numeric prefix if present (e.g., "00072_" from "00072_french-irv-2007")
    if '_' in dir_name:
        parts = dir_name.split('_', 1)
        if parts[0].isdigit():
            return parts[1]

    return dir_name
