import numpy as np
import torch

def get_model_allocations(model, valuation_matrix):
    """Get model allocations for a given valuation matrix"""
    # Placeholder for model allocations

    # Convert valuation_matrix to tensor
    valuation_tensor = torch.tensor(valuation_matrix, dtype=torch.float32)

    allocation_matrix = np.zeros_like(valuation_matrix)
    return allocation_matrix

def get_model_allocations_batch(model, valuation_matrices):
    """Get model allocations for a batch of valuation matrices"""
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

    # Placeholder for batch model allocations
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