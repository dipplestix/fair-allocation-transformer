import numpy as np

def calculate_agent_bundle_values(valuation_matrix, allocation_matrix):
    """
    Calculate how much each agent values each bundle (including their own and others')
    Returns: matrix where agent_bundle_values[i][j] = how much agent i values agent j's bundle
    """
    m, n = valuation_matrix.shape
    agent_bundle_values = np.zeros((m, m))
    
    for agent_i in range(m):
        for agent_j in range(m):
            bundle_j = allocation_matrix[agent_j]
            agent_bundle_values[agent_i][agent_j] = np.sum(valuation_matrix[agent_i] * bundle_j)
    
    return agent_bundle_values

def calculate_agent_bundle_values_batch(valuation_matrices, allocation_matrices):
    """
    Calculate agent bundle values for a batch of N matrices
    Args:
        valuation_matrices: (N, m, n) - N valuation matrices
        allocation_matrices: (N, m, n) - N allocation matrices
    Returns: (N, m, m) - agent bundle values for each matrix pair
    """
    return valuation_matrices @ allocation_matrices.transpose(0, 2, 1)

def is_envy_free(agent_bundle_values):
    """Check if allocation is envy-free"""
    m = agent_bundle_values.shape[0]
    
    for agent_i in range(m):
        own_value = agent_bundle_values[agent_i][agent_i]
        for agent_j in range(m):
            if agent_i != agent_j:
                if agent_bundle_values[agent_i][agent_j] > own_value:
                    return False
    return True

def is_envy_free_batch(agent_bundle_values_batch):
    """Check envy-freeness for batch of matrices
    Args: agent_bundle_values_batch (N, m, m)
    Returns: (N,) boolean array
    """
    # Extract diagonals for each matrix: (N, m)
    diagonals = np.diagonal(agent_bundle_values_batch, axis1=1, axis2=2)
    
    # Get row maxes for each matrix: (N, m) 
    row_maxes = np.max(agent_bundle_values_batch, axis=2)
    
    # Check if diagonal >= row_max for each agent in each matrix
    return np.all(diagonals >= row_maxes, axis=1)

def is_ef1(valuation_matrix, allocation_matrix, agent_bundle_values):
    """Check if allocation is EF1"""
    m, n = valuation_matrix.shape
    
    for agent_i in range(m):
        own_value = agent_bundle_values[agent_i][agent_i]
        
        for agent_j in range(m):
            if agent_i != agent_j:
                other_bundle = allocation_matrix[agent_j]
                other_items = np.where(other_bundle == 1)[0]
                
                if len(other_items) == 0:
                    continue
                
                max_item_value = np.max(valuation_matrix[agent_i][other_items])
                other_value_minus_max = agent_bundle_values[agent_i][agent_j] - max_item_value
                
                if own_value < other_value_minus_max:
                    return False
    return True

def is_ef1_batch(valuation_matrices, allocation_matrices, agent_bundle_values_batch):
    """Check EF1 for batch of matrices
        Returns: (N,) boolean array
    """
    N, m, n = valuation_matrices.shape
    
    # Broadcasting: (N,m,1,n) * (N,1,m,n) = (N,m,m,n)
    valuations_expanded = valuation_matrices[:, :, np.newaxis, :]
    allocations_expanded = allocation_matrices[:, np.newaxis, :, :]
    
    masked_valuations = np.where(allocations_expanded, valuations_expanded, -np.inf)
    max_item_values = np.max(masked_valuations, axis=3)  # (N,m,m)
    
    # Handle empty bundles
    bundle_has_items = np.sum(allocation_matrices, axis=2) > 0  # (N,m)
    bundle_has_items_expanded = bundle_has_items[:, np.newaxis, :]  # (N, 1, m)
    max_item_values = np.where(bundle_has_items_expanded, max_item_values, 0)
    
    # Check EF1 conditions
    other_value_minus_max = agent_bundle_values_batch - max_item_values
    own_values = np.diagonal(agent_bundle_values_batch, axis1=1, axis2=2)  # (N,m)
    
    mask = ~np.eye(m, dtype=bool)
    ef1_conditions = own_values[..., np.newaxis] >= other_value_minus_max - 1e-9 # small tolerance
    
    return np.all(ef1_conditions[..., mask], axis=1)  # (N,)

def is_efx(valuation_matrix, allocation_matrix, agent_bundle_values):
    """Check if allocation is EFx"""
    m, n = valuation_matrix.shape
    
    for agent_i in range(m):
        own_value = agent_bundle_values[agent_i][agent_i]
        
        for agent_j in range(m):
            if agent_i != agent_j:
                other_bundle = allocation_matrix[agent_j]
                other_items = np.where(other_bundle == 1)[0]
                
                if len(other_items) == 0:
                    continue
                
                min_item_value = np.min(valuation_matrix[agent_i][other_items])
                other_value_minus_min = agent_bundle_values[agent_i][agent_j] - min_item_value
                
                if own_value < other_value_minus_min:
                    return False
    return True

def is_efx_batch(valuation_matrices, allocation_matrices, agent_bundle_values_batch):
    """Check EFx for batch of matrices
    Returns: (N,) boolean array
    """
    N, m, n = valuation_matrices.shape
    
    # Broadcasting: (N,m,1,n) * (N,1,m,n) = (N,m,m,n)
    valuations_expanded = valuation_matrices[:, :, np.newaxis, :]
    allocations_expanded = allocation_matrices[:, np.newaxis, :, :]
    
    masked_valuations = np.where(allocations_expanded, valuations_expanded, np.inf)
    min_item_values = np.min(masked_valuations, axis=3)  # (N,m,m)
    
    # Handle empty bundles
    bundle_has_items = np.sum(allocation_matrices, axis=2) > 0  # (N,m)
    bundle_has_items_expanded = bundle_has_items[:, np.newaxis, :]  # (N, 1, m)
    min_item_values = np.where(bundle_has_items_expanded, min_item_values, 0)

    # Check EFx conditions
    other_value_minus_min = agent_bundle_values_batch - min_item_values
    own_values = np.diagonal(agent_bundle_values_batch, axis1=1, axis2=2)  # (N,m)
    
    mask = ~np.eye(m, dtype=bool)
    efx_conditions = own_values[..., np.newaxis] >= other_value_minus_min
    
    return np.all(efx_conditions[..., mask], axis=1)  # (N,)

def utility_sum(agent_bundle_values):
    """Calculate sum of utilities (each agent's value of their own bundle)"""
    return np.sum(np.diag(agent_bundle_values))

def utility_sum_batch(agent_bundle_values_batch):
    """Calculate utility sums for batch of matrices
        Returns: (N,) array of sums
    """
    return np.sum(np.diagonal(agent_bundle_values_batch, axis1=1, axis2=2), axis=1)

def nash_welfare(agent_bundle_values):
    """Calculate Nash welfare (geometric mean of utilities)"""
    return np.prod(np.diag(agent_bundle_values)) ** (1 / agent_bundle_values.shape[0])

def nash_welfare_batch(agent_bundle_values_batch):
    """Calculate Nash welfares for batch of matrices
    Returns: (N,) array of Nash welfares
    """
    diagonals = np.diagonal(agent_bundle_values_batch, axis1=1, axis2=2)
    prod_diagonals = np.prod(diagonals, axis=1)
    m = agent_bundle_values_batch.shape[1]
    return prod_diagonals ** (1 / m)

