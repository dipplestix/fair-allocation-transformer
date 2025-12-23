"""
MovieLens Data Conversion Module

This module provides functionality to parse MovieLens rating data
and convert ratings to valuations for fair allocation problems.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
import pandas as pd


def load_ratings_csv(file_path: str) -> pd.DataFrame:
    """
    Load MovieLens ratings CSV file.

    Args:
        file_path: Path to ratings.csv file

    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    df = pd.read_csv(file_path)
    required_columns = ['userId', 'movieId', 'rating']

    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    return df[required_columns]  # Drop timestamp if present


def find_common_movies_and_users(
    df: pd.DataFrame,
    n_agents: int,
    n_items: int
) -> Tuple[Set[int], Set[int]]:
    """
    Find sets of users and movies where all users rated all movies.

    Args:
        df: Ratings DataFrame
        n_agents: Number of users (agents) needed
        n_items: Number of movies (items) needed

    Returns:
        Tuple of (user_ids, movie_ids) sets that satisfy the constraints
    """
    # Start with movies that have been rated by at least n_agents users
    movie_counts = df.groupby('movieId')['userId'].nunique()
    candidate_movies = set(movie_counts[movie_counts >= n_agents].index)

    if len(candidate_movies) < n_items:
        raise ValueError(
            f"Only {len(candidate_movies)} movies have been rated by at least "
            f"{n_agents} users. Need at least {n_items} movies."
        )

    # Try to find a set of n_items movies and n_agents users with complete ratings
    # Strategy: Start with most-rated movies and find users who rated all of them
    top_movies = movie_counts.nlargest(n_items * 3).index.tolist()  # Get more candidates

    best_users = set()
    best_movies = set()

    # Try different combinations of movies
    for start_idx in range(0, len(top_movies) - n_items + 1):
        candidate_movie_set = set(top_movies[start_idx:start_idx + n_items])

        # Find users who rated ALL these movies
        movie_subset = df[df['movieId'].isin(candidate_movie_set)]
        user_rating_counts = movie_subset.groupby('userId')['movieId'].count()
        complete_users = set(user_rating_counts[user_rating_counts == n_items].index)

        if len(complete_users) >= n_agents:
            best_users = complete_users
            best_movies = candidate_movie_set
            break

    if len(best_users) < n_agents:
        raise ValueError(
            f"Could not find {n_agents} users who all rated a common set of "
            f"{n_items} movies. Maximum found: {len(best_users)} users."
        )

    return best_users, best_movies


def ratings_to_valuation(
    ratings: Dict[int, float],
    max_rating: float = 5.0
) -> Tuple[np.ndarray, List[int]]:
    """
    Convert ratings directly to normalized valuations.

    Args:
        ratings: Dict mapping movieId -> rating
        max_rating: Maximum possible rating (default: 5.0 for MovieLens)

    Returns:
        Tuple of (valuation array, list of movie IDs in same order)
    """
    # Sort movie IDs for consistent ordering
    movie_ids = sorted(ratings.keys())
    n_items = len(movie_ids)

    # Create valuation array
    valuation = np.zeros(n_items)

    for idx, movie_id in enumerate(movie_ids):
        # Normalize rating to [0, 1] range
        valuation[idx] = ratings[movie_id] / max_rating

    return valuation, movie_ids


def create_valuation_matrices(
    df: pd.DataFrame,
    user_ids: Set[int],
    movie_ids: Set[int],
    n_agents: int,
    n_items: int,
    num_matrices: int,
    max_rating: float = 5.0,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    Create multiple UNIQUE valuation matrices using deterministic combinations.
    Ratings are directly normalized to [0, 1] range.

    Args:
        df: Ratings DataFrame
        user_ids: Set of valid user IDs
        movie_ids: Set of valid movie IDs
        n_agents: Number of agents per matrix
        n_items: Number of items per matrix
        num_matrices: Number of matrices to generate
        max_rating: Maximum rating value for normalization (default: 5.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (matrices, selected_user_ids, selected_movie_ids)
    """
    from math import comb
    from itertools import combinations
    import random

    # Convert to sorted lists for deterministic indexing
    user_list = sorted(list(user_ids))
    movie_list = sorted(list(movie_ids))

    # Validate we have enough users and movies
    if len(user_list) < n_agents:
        raise ValueError(
            f"Not enough users ({len(user_list)}) to create matrices with {n_agents} agents"
        )

    if len(movie_list) < n_items:
        raise ValueError(
            f"Not enough movies ({len(movie_list)}) to create matrices with {n_items} items"
        )

    # Calculate maximum unique combinations
    max_user_combinations = comb(len(user_list), n_agents)
    max_movie_combinations = comb(len(movie_list), n_items)
    max_combinations = max_user_combinations * max_movie_combinations

    if num_matrices > max_combinations:
        raise ValueError(
            f"Requested {num_matrices} matrices but only {max_combinations:,} "
            f"unique combinations possible (C({len(user_list)}, {n_agents}) × "
            f"C({len(movie_list)}, {n_items}))"
        )

    # Generate combinations
    # For simplicity, we'll generate unique user combinations and reuse movie subset
    # This ensures uniqueness while keeping it manageable

    # Select n_items movies (use first n_items from sorted list for now)
    selected_movies = movie_list[:n_items]

    # Generate user combinations
    all_user_indices = range(len(user_list))
    user_combo_generator = combinations(all_user_indices, n_agents)

    # Take first num_matrices combinations
    selected_user_combinations = []
    for i, combo in enumerate(user_combo_generator):
        if i >= num_matrices:
            break
        selected_user_combinations.append(combo)

    # Shuffle for randomization
    random.seed(seed)
    random.shuffle(selected_user_combinations)

    # Create matrices
    matrices = []
    np.random.seed(seed)

    # Filter DataFrame to only selected movies
    df_subset = df[df['movieId'].isin(selected_movies)]

    for user_combo in selected_user_combinations:
        # Get actual user IDs
        selected_user_ids = [user_list[idx] for idx in user_combo]

        # Create matrix for these users and movies
        matrix = np.zeros((n_agents, n_items))

        for agent_idx, user_id in enumerate(selected_user_ids):
            # Get this user's ratings for selected movies
            user_ratings = df_subset[df_subset['userId'] == user_id]
            ratings_dict = dict(zip(user_ratings['movieId'], user_ratings['rating']))

            # Convert ratings directly to valuations (normalized)
            valuation, movie_id_order = ratings_to_valuation(ratings_dict, max_rating)
            matrix[agent_idx] = valuation

        matrices.append(matrix)

    return matrices, user_list, selected_movies


def load_movielens_dataset(
    csv_path: str,
    n_agents: int,
    n_items: int,
    num_matrices: int,
    max_rating: float = 5.0,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Load and process a MovieLens dataset.
    Ratings are directly normalized to [0, 1] range as valuations.

    Args:
        csv_path: Path to ratings.csv file
        n_agents: Number of agents per matrix
        n_items: Number of items per matrix
        num_matrices: Number of matrices to generate
        max_rating: Maximum rating value for normalization (default: 5.0)
        seed: Random seed

    Returns:
        List of valuation matrices
    """
    print(f"Loading MovieLens data from: {csv_path}")

    # Load ratings
    df = load_ratings_csv(csv_path)
    total_users = df['userId'].nunique()
    total_movies = df['movieId'].nunique()
    total_ratings = len(df)

    print(f"Dataset info: {total_users} users, {total_movies} movies, {total_ratings} ratings")

    # Find common movies and users
    print(f"Finding {n_agents} users who all rated a common set of {n_items} movies...")
    user_ids, movie_ids = find_common_movies_and_users(df, n_agents, n_items)

    print(f"Found {len(user_ids)} users who rated {len(movie_ids)} movies in common")

    # Calculate and report maximum unique combinations
    from math import comb
    max_user_combos = comb(len(user_ids), n_agents)
    max_movie_combos = comb(len(movie_ids), n_items)
    max_unique_combinations = max_user_combos * max_movie_combos

    print(f"Maximum unique matrices possible: {max_unique_combinations:,}")
    print(f"  = C({len(user_ids)}, {n_agents}) × C({len(movie_ids)}, {n_items})")

    if num_matrices > max_unique_combinations:
        raise ValueError(
            f"Cannot generate {num_matrices:,} unique matrices. "
            f"Maximum possible: {max_unique_combinations:,}"
        )

    # Create valuation matrices
    matrices, _, _ = create_valuation_matrices(
        df, user_ids, movie_ids, n_agents, n_items, num_matrices, max_rating, seed
    )

    return matrices
