import numpy as np
import torch
from typing import List, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB

def get_model_allocations(model, valuation_matrix):
    """Get model allocations for a given valuation matrix"""
    # Placeholder for model allocations

    # Convert valuation_matrix to tensor
    valuation_tensor = torch.tensor(valuation_matrix, dtype=torch.float32)

    allocation_matrix = np.zeros_like(valuation_matrix)
    return allocation_matrix

def get_model_allocations_batch(model, valuation_matrices, apply_ef1_repair=False,
                                ef1_repair_params=None):
    """Get model allocations for a batch of valuation matrices with optional EF1 repair.

    Args:
        model: Trained FATransformer model
        valuation_matrices: (N, n_agents, m_items) valuation matrices
        apply_ef1_repair: Whether to apply EF1 quick repair post-processing
        ef1_repair_params: Dict of parameters for EF1 repair (e.g., {'max_passes': 10})

    Returns:
        Allocation matrices of shape (N, n_agents, m_items)
    """
    N, n_agents, m_items = valuation_matrices.shape
    allocation_matrices = np.zeros((N, n_agents, m_items), dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert valuation_matrices to tensor
    valuation_tensors = torch.tensor(valuation_matrices, dtype=torch.float32).to(device)

    allocation_tensors = model(valuation_tensors)

    # max across rows
    max_indices = torch.argmax(allocation_tensors, dim=2)

    #convert to allocation matrices
    allocation_matrices = torch.zeros_like(allocation_tensors)
    allocation_matrices.scatter_(2, max_indices.unsqueeze(2), 1)

    # transpose to (N, n_agents, m_items)
    allocation_matrices = allocation_matrices.transpose(1, 2).cpu().numpy()

    # Apply EF1 quick repair if requested
    if apply_ef1_repair:
        from utils.ef1_repair import ef1_quick_repair_batch
        ef1_repair_params = ef1_repair_params or {}
        allocation_matrices = ef1_quick_repair_batch(
            allocation_matrices, valuation_matrices, **ef1_repair_params
        )

    return allocation_matrices

def get_random_allocations(valuation_matrix):
    """Generate random allocation matrix ensuring each item goes to exactly one agent"""
    
    n_agents, m_items = valuation_matrix.shape
    allocation_matrix = np.zeros((n_agents, m_items), dtype=int)
    
    for item in range(m_items):
        agent = np.random.randint(0, n_agents)
        allocation_matrix[agent][item] = 1

    return allocation_matrix

def get_random_allocations_batch(valuation_matrices):
    """Generate random allocation matrices for a batch of valuation matrices"""
    
    N, n_agents, m_items = valuation_matrices.shape
    # We want generate 5 random allocations for each matrix
    N = N * 5
    allocation_matrices = np.zeros((N, n_agents, m_items), dtype=int)
    
    # Generate random assignments and set in one go
    random_agents = np.random.randint(0, n_agents, size=(N, m_items))

    
    allocation_matrices[np.arange(N)[:, None], random_agents, np.arange(m_items)] = 1
    return allocation_matrices

def get_rr_allocations(valuation_matrix):
    """Generate round-robin allocation matrix"""
    
    n_agents, m_items = valuation_matrix.shape
    allocation_matrix = np.zeros((n_agents, m_items), dtype=int)
    
    for item in range(m_items):
        agent = item % n_agents
        allocation_matrix[agent][item] = 1

    return allocation_matrix

def get_rr_allocations_batch(valuation_matrices, n_permutations=5):
    """Generate round-robin allocations with random agent orders and vectorized operations"""
    N, n_agents, m_items = valuation_matrices.shape
    
    # Pre-sort item preferences for all agents (descending order)
    sorted_preferences = np.argsort(-valuation_matrices, axis=2)  # (N, n_agents, m_items)
    
    all_allocations = []
    
    for perm in range(n_permutations):
        # Generate random agent orders for each matrix
        agent_orders = np.array([np.random.permutation(n_agents) for _ in range(N)])  # (N, n_agents)
        
        # Create round-robin schedule (repeat agent order until all items assigned)
        schedule_length = m_items
        schedule = np.tile(agent_orders, (1, (m_items // n_agents) + 1))[:, :schedule_length]  # (N, m_items)
        
        # Initialize allocations and tracking
        allocation_matrices = np.zeros((N, n_agents, m_items), dtype=int)
        taken_items = np.zeros((N, m_items), dtype=bool)  # Track taken items per matrix
        agent_pref_idx = np.zeros((N, n_agents), dtype=int)  # Track preference index per agent
        
        # Round-robin picking
        for pick_round in range(schedule_length):
            # Get current picking agents for each matrix
            current_agents = schedule[:, pick_round]  # (N,)
            
            # Find next available item for each current agent
            valid_matrices = np.arange(N)
            
            for i in range(N):
                agent = current_agents[i]
                
                # Find next available item in this agent's preference list
                while agent_pref_idx[i, agent] < m_items:
                    preferred_item = sorted_preferences[i, agent, agent_pref_idx[i, agent]]
                    
                    if not taken_items[i, preferred_item]:
                        # Assign item to agent
                        allocation_matrices[i, agent, preferred_item] = 1
                        taken_items[i, preferred_item] = True
                        agent_pref_idx[i, agent] += 1
                        break
                    
                    agent_pref_idx[i, agent] += 1
        
        all_allocations.append(allocation_matrices)
    
    # Stack all permutations: (N*n_permutations, n_agents, m_items)
    return np.concatenate(all_allocations, axis=0)

def get_rr_allocations_batch_old(valuation_matrices, n_permutations=5):
    """Generate round-robin allocations with random agent orders"""
    N, n_agents, m_items = valuation_matrices.shape
    allocation_matrices = np.zeros((N, n_agents, m_items), dtype=int)
    
    for i in range(N):
        for _ in range(n_permutations):
            # for now don't randomize agent order
            agent_order = np.arange(n_agents)

            # assign each agent their most preferred available item in round-robin fashion, until all items are assigned
            taken_items = set()
            while len(taken_items) < m_items:
                for agent in agent_order:
                    # get agent's preferences sorted by valuation
                    preferences = np.argsort(-valuation_matrices[i][agent])
                    for item in preferences:
                        if item not in taken_items:
                            allocation_matrices[i][agent][item] = 1
                            taken_items.add(item)
                            break

    return allocation_matrices

import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict, Optional

# -------------------------
# UM oracle using Gurobi
# -------------------------
def _um_total_value_with_fixed_gurobi(
    V: np.ndarray,
    quotas: List[int],
    fixed_pairs: List[Tuple[int, int]],
    time_limit: float = 10.0
) -> Optional[float]:
    """
    Return the maximum total utility value achievable with fixed assignments.
    Uses your Gurobi pattern from best_nash_welfare.
    """
    n, m = V.shape
    quotas_rem = quotas[:]
    assigned_items = set()
    sum_fixed = 0.0

    # Apply fixed assignments
    for i, o in fixed_pairs:
        if o in assigned_items or quotas_rem[i] <= 0:
            return None
        quotas_rem[i] -= 1
        assigned_items.add(o)
        sum_fixed += float(V[i, o])

    remaining_items = [o for o in range(m) if o not in assigned_items]
    F = len(remaining_items)

    if F == 0:
        return sum_fixed if sum(quotas_rem) == 0 else None
    if any(q < 0 for q in quotas_rem) or sum(quotas_rem) != F:
        return None

    # Build Gurobi model (following your pattern)
    try:
        model = gp.Model("um_completion")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)

        # Binary allocation variables
        x = model.addVars(n, F, vtype=GRB.BINARY, name="x")

        # Each remaining item assigned to exactly one agent
        for j_idx in range(F):
            model.addConstr(gp.quicksum(x[i, j_idx] for i in range(n)) == 1)

        # Each agent gets exactly their remaining quota
        for i in range(n):
            model.addConstr(
                gp.quicksum(x[i, j_idx] for j_idx in range(F)) == quotas_rem[i]
            )

        # Objective: Maximize total utility
        obj = gp.quicksum(
            V[i, remaining_items[j_idx]] * x[i, j_idx]
            for i in range(n)
            for j_idx in range(F)
        )
        model.setObjective(obj, GRB.MAXIMIZE)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            return sum_fixed + model.objVal
        else:
            return None

    except Exception as e:
        print(f"Gurobi error: {e}")
        return None


# -------------------------
# Preference handling (equivalence classes)
# -------------------------
def _equivalence_classes_for_agent(V_row: np.ndarray, tol: float = 1e-12) -> List[List[int]]:
    """Turn a valuation row into equivalence classes in descending order."""
    m = V_row.shape[0]
    order = sorted(range(m), key=lambda o: (-float(V_row[o]), o))
    classes: List[List[int]] = []
    if m == 0:
        return classes
    current = [order[0]]
    for prev, cur in zip(order, order[1:]):
        if abs(float(V_row[prev]) - float(V_row[cur])) <= tol:
            current.append(cur)
        else:
            classes.append(current)
            current = [cur]
    classes.append(current)
    return classes


# -------------------------
# Single instance W-CRR
# -------------------------
def _constrained_round_robin_single(
    V: np.ndarray,
    welfare: str = "um",
    tol: float = 1e-8,
    gurobi_time_limit: float = 10.0
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Single instance of W-CRR algorithm."""
    assert V.ndim == 2
    n, m = V.shape
    assert n >= 1 and m >= 1

    # Balanced quotas
    base = m // n
    r = m % n
    quotas = [base + (1 if i < r else 0) for i in range(n)]

    # Precompute UM* value (w*) if needed
    if welfare == "um":
        w_star = _um_total_value_with_fixed_gurobi(V, quotas, [], gurobi_time_limit)
        if w_star is None:
            raise RuntimeError("UM oracle: baseline allocation infeasible with balanced quotas.")
    elif welfare == "none":
        w_star = None
    else:
        raise ValueError("welfare must be 'um' or 'none'.")

    # Build equivalence class lists (descending)
    pref_classes: List[List[List[int]]] = [
        _equivalence_classes_for_agent(V[i, :]) for i in range(n)
    ]
    top_idx = [0 for _ in range(n)]

    # State
    assigned_count = [0] * n
    assigned_items = set()
    picks: List[Tuple[int, int]] = []

    def active_agents() -> List[int]:
        out = []
        for i in range(n):
            if assigned_count[i] >= quotas[i]:
                continue
            k = top_idx[i]
            while k < len(pref_classes[i]) and all(o in assigned_items for o in pref_classes[i][k]):
                k += 1
            if k < len(pref_classes[i]):
                out.append(i)
        return out

    def current_top_available(i: int) -> List[int]:
        k = top_idx[i]
        while k < len(pref_classes[i]) and all(o in assigned_items for o in pref_classes[i][k]):
            k += 1
        if k >= len(pref_classes[i]):
            return []
        return [o for o in pref_classes[i][k] if o not in assigned_items]

    # Main loop
    while len(assigned_items) < m:
        N_active = active_agents()
        if not N_active:
            raise RuntimeError("CRR got stuck: no active agents although items remain.")

        min_count = min(assigned_count[i] for i in N_active)
        N_min = sorted([i for i in N_active if assigned_count[i] == min_count])

        chosen: Optional[Tuple[int, int]] = None

        for i in N_min:
            top_items = current_top_available(i)
            if not top_items:
                continue
            for o in top_items:
                if welfare == "none":
                    chosen = (i, o)
                    break
                else:
                    val = _um_total_value_with_fixed_gurobi(
                        V, quotas, picks + [(i, o)], gurobi_time_limit
                    )
                    if val is not None and val >= (w_star - tol * max(1.0, abs(w_star))):
                        chosen = (i, o)
                        break
            if chosen is not None:
                break

        if chosen is not None:
            i, o = chosen
            picks.append((i, o))
            assigned_items.add(o)
            assigned_count[i] += 1
            continue

        for i in N_min:
            if top_idx[i] < len(pref_classes[i]):
                top_idx[i] += 1

    # Build assignment matrix
    X = np.zeros_like(V, dtype=np.int64)
    for i, o in picks:
        X[i, o] = 1

    return picks, X


# -------------------------
# Batched W-CRR (following RR pattern)
# -------------------------
def get_crr_allocations_batch(
    valuation_matrices: np.ndarray,
    welfare: str = "um",
    tol: float = 1e-8,
    gurobi_time_limit: float = 10.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Generate W-CRR allocations for a batch of valuation matrices.
    Follows the pattern of your get_rr_allocations_batch function.

    Args:
        valuation_matrices: (N, n_agents, m_items) - batch of valuation matrices
        welfare: 'um' (utilitarian-maximal) or 'none' (no welfare constraint)
        tol: numerical tolerance for equality checks
        gurobi_time_limit: time limit for each Gurobi solve (seconds)
        verbose: print progress

    Returns:
        allocation_matrices: (N, n_agents, m_items) - batch of allocation matrices
    """
    assert valuation_matrices.ndim == 3, "Expected (N, n_agents, m_items) shape"
    N, n_agents, m_items = valuation_matrices.shape

    allocation_matrices = np.zeros((N, n_agents, m_items), dtype=np.int64)

    for batch_idx in range(N):
        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"Processing {batch_idx + 1}/{N}...")

        V = valuation_matrices[batch_idx]
        try:
            _, X = _constrained_round_robin_single(
                V, welfare=welfare, tol=tol,
                gurobi_time_limit=gurobi_time_limit
            )
            allocation_matrices[batch_idx] = X
        except Exception as e:
            if verbose:
                print(f"Error processing batch {batch_idx}: {e}")
            # Return zero allocation for failed instances
            allocation_matrices[batch_idx] = np.zeros((n_agents, m_items), dtype=np.int64)

    return allocation_matrices


def build_envy_graph(valuation_matrix, agent_bundles):
    """
    Build directed envy graph where edge i->j means agent i envies agent j.

    Args:
        valuation_matrix: (n_agents, m_items) numpy array
        agent_bundles: list of sets, agent_bundles[i] = set of items agent i has

    Returns:
        envy_graph: dict mapping agent -> list of agents they envy
    """
    n_agents = valuation_matrix.shape[0]

    # Compute bundle values
    bundle_values = {}
    for i in range(n_agents):
        bundle_values[i] = {}
        for j in range(n_agents):
            bundle_values[i][j] = sum(valuation_matrix[i][item]
                                     for item in agent_bundles[j])

    # Build envy graph
    envy_graph = {i: [] for i in range(n_agents)}
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j and bundle_values[i][j] > bundle_values[i][i]:
                envy_graph[i].append(j)

    return envy_graph


def detect_and_remove_cycle(envy_graph, agent_bundles):
    """
    Detect a cycle in the envy graph and remove it via cyclic bundle exchange.

    Uses DFS to find a cycle, then rotates bundles along the cycle.

    Args:
        envy_graph: dict mapping agent -> list of envied agents
        agent_bundles: list of sets (modified in place)

    Returns:
        bool: True if a cycle was found and removed, False if no cycle exists
    """
    n_agents = len(envy_graph)
    visited = set()
    rec_stack = set()

    def dfs(node, path):
        """DFS to find cycle. Returns cycle as list if found, None otherwise."""
        if node in rec_stack:
            # Found cycle - extract it
            cycle_start = path.index(node)
            return path[cycle_start:]

        if node in visited:
            return None

        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in envy_graph[node]:
            cycle = dfs(neighbor, path)
            if cycle:
                return cycle

        rec_stack.remove(node)
        path.pop()
        return None

    # Try to find a cycle starting from each node
    for start_node in range(n_agents):
        if start_node not in visited:
            cycle = dfs(start_node, [])
            if cycle:
                # Remove cycle by rotating bundles
                # If cycle is [a, b, c], then a gets b's bundle, b gets c's, c gets a's
                temp_bundle = agent_bundles[cycle[0]].copy()
                for i in range(len(cycle) - 1):
                    agent_bundles[cycle[i]] = agent_bundles[cycle[i + 1]].copy()
                agent_bundles[cycle[-1]] = temp_bundle

                return True

    return False


def get_ece_allocation(valuation_matrix):
    """
    Envy Cycle Elimination algorithm for single valuation matrix.

    Maintains EF1 property at every step by:
    1. Assigning items in order (0, 1, 2, ..., m-1)
    2. Detecting and removing envy cycles when needed
    3. Giving each item to an unenvied agent

    Args:
        valuation_matrix: (n_agents, m_items) numpy array

    Returns:
        allocation_matrix: (n_agents, m_items) binary allocation (EF1 guaranteed)
    """
    n_agents, m_items = valuation_matrix.shape
    allocation_matrix = np.zeros((n_agents, m_items), dtype=int)

    # Initialize agent bundles (list of sets)
    agent_bundles = [set() for _ in range(n_agents)]

    # Process items in order (arbitrary ordering: 0, 1, 2, ..., m-1)
    for item_idx in range(m_items):
        # Build envy graph based on current allocations
        envy_graph = build_envy_graph(valuation_matrix, agent_bundles)

        # Find unenvied agents (nodes with no incoming edges)
        envied_agents = set()
        for agent, envies_list in envy_graph.items():
            envied_agents.update(envies_list)

        unenvied_agents = [i for i in range(n_agents) if i not in envied_agents]

        # If no unenvied agent, remove cycles until one exists
        while len(unenvied_agents) == 0:
            cycle_removed = detect_and_remove_cycle(envy_graph, agent_bundles)
            if not cycle_removed:
                # Safety: shouldn't happen, but fallback to agent 0
                unenvied_agents = [0]
                break

            # Update allocation matrix to reflect swapped bundles
            allocation_matrix = np.zeros((n_agents, m_items), dtype=int)
            for agent_idx, bundle in enumerate(agent_bundles):
                for item in bundle:
                    allocation_matrix[agent_idx][item] = 1

            # Rebuild envy graph after cycle removal
            envy_graph = build_envy_graph(valuation_matrix, agent_bundles)

            # Recompute unenvied agents
            envied_agents = set()
            for agent, envies_list in envy_graph.items():
                envied_agents.update(envies_list)
            unenvied_agents = [i for i in range(n_agents) if i not in envied_agents]

        # Pick an unenvied agent (tie-break by smallest index for determinism)
        selected_agent = min(unenvied_agents)

        # Allocate item to selected agent
        allocation_matrix[selected_agent][item_idx] = 1
        agent_bundles[selected_agent].add(item_idx)

    return allocation_matrix


def get_ece_allocations_batch(valuation_matrices):
    """
    Generate ECE allocations for a batch of valuation matrices.

    Args:
        valuation_matrices: (N, n_agents, m_items) numpy array

    Returns:
        allocation_matrices: (N, n_agents, m_items) binary allocations (EF1 guaranteed)
    """
    N, n_agents, m_items = valuation_matrices.shape
    allocation_matrices = np.zeros((N, n_agents, m_items), dtype=int)

    # Loop over batch (ECE is sequential, hard to vectorize)
    for i in range(N):
        allocation_matrices[i] = get_ece_allocation(valuation_matrices[i])

    return allocation_matrices


def get_max_util_allocations(valuation_matrix):
    """Generate max utilitarian welfare allocation (greedy: each item to highest-valuing agent)"""
    n_agents, m_items = valuation_matrix.shape
    allocation_matrix = np.zeros((n_agents, m_items), dtype=int)

    # Assign each item to the agent that values it most
    best_agents = np.argmax(valuation_matrix, axis=0)  # shape: (m_items,)
    allocation_matrix[best_agents, np.arange(m_items)] = 1

    return allocation_matrix


def get_max_util_allocations_batch(valuation_matrices):
    """Batch version for max utilitarian welfare allocations"""
    N, n_agents, m_items = valuation_matrices.shape
    allocation_matrices = np.zeros((N, n_agents, m_items), dtype=int)

    # Vectorized: for each matrix, find argmax across agents for each item
    best_agents = np.argmax(valuation_matrices, axis=1)  # shape: (N, m_items)

    # Use advanced indexing to set the allocations
    batch_indices = np.arange(N)[:, None]
    item_indices = np.arange(m_items)
    allocation_matrices[batch_indices, best_agents, item_indices] = 1

    return allocation_matrices


def get_max_nash_allocation(valuation_matrix):
    """Generate max Nash welfare allocation using Gurobi LP with PWL approximation"""
    from utils.max_utility import best_nash_welfare
    _, allocation = best_nash_welfare(valuation_matrix, return_allocation=True)
    return allocation


def get_max_nash_allocations_batch(valuation_matrices):
    """Batch version for max Nash welfare allocations.

    Note: This is slow as it requires solving a MIP for each matrix.
    """
    N, n_agents, m_items = valuation_matrices.shape
    allocation_matrices = np.zeros((N, n_agents, m_items), dtype=int)

    for i in range(N):
        allocation_matrices[i] = get_max_nash_allocation(valuation_matrices[i])

    return allocation_matrices
