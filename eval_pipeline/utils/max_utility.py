from utils.calculations import calculate_agent_bundle_values, utility_sum
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linprog


def best_nash_welfare(valuation_matrix, num_segments=10):
    """
    Approximate log with piecewise linear function
    Runtime: Polynomial 
    """
    m, n = valuation_matrix.shape
    # scale valuation matrix by 10
    valuation_matrix = valuation_matrix * 10
    model = gp.Model("nash_welfare_pwl")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    
    # Binary allocation variables
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x")
    
    # Each item to exactly one agent
    for j in range(n):
        model.addConstr(gp.quicksum(x[i,j] for i in range(m)) == 1)
    
    lower_bound = 0.01  # To avoid log(0)
    upper_bound = 100  # Arbitrary upper bound for utilities
    # Agent utilities
    u = model.addVars(m, lb=lower_bound, ub=upper_bound, name="u")  # Set reasonable bounds
    for i in range(m):
        model.addConstr(u[i] == gp.quicksum(valuation_matrix[i,j] * x[i,j] 
                                           for j in range(n)) + 0.01)  # Small constant
    
    # Piecewise linear approximation of log
    log_u = model.addVars(m, name="log_u")
    for i in range(m):
        # Create breakpoints for piecewise linear function
        breakpoints = np.linspace(lower_bound, upper_bound, num_segments)
        log_values = np.log(breakpoints)
        
        model.addGenConstrPWL(u[i], log_u[i], breakpoints.tolist(), log_values.tolist())
    
    # Maximize sum of log utilities
    model.setObjective(gp.quicksum(log_u[i] for i in range(m)), GRB.MAXIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # allocation = np.array([[x[i,j].x for j in range(n)] for i in range(m)])
        sum_of_logs = model.objVal
        nash_welfare = np.exp(sum_of_logs / m)  # Geometric mean
        return nash_welfare / 10 # Scale back
    else:
        print("No optimal solution found, using bruteforce fallback")
        return best_nash_welfare_bruteforce(valuation_matrix / 10)  # Scale back and use bruteforce as fallback


def best_nash_welfare_bruteforce(valuation_matrix):
    """Calculate best possible Nash welfare using brute force (O(m^n))"""
    m, n = valuation_matrix.shape
    best_nash_welfare = 0
    
    def generate_all_allocations(items, agents):
        """Generate all possible allocations of items to agents"""
        if len(items) == 0:
            yield [[] for _ in range(agents)]
            return
        
        item = items[0]
        remaining_items = items[1:]
        
        for sub_allocation in generate_all_allocations(remaining_items, agents):
            for agent in range(agents):
                new_allocation = [bundle.copy() for bundle in sub_allocation]
                new_allocation[agent].append(item)
                yield new_allocation
    
    for allocation in generate_all_allocations(list(range(n)), m):
        allocation_matrix = np.zeros((m, n))
        for agent in range(m):
            for item in allocation[agent]:
                allocation_matrix[agent][item] = 1
        
        agent_bundle_values = calculate_agent_bundle_values(valuation_matrix, allocation_matrix)
        current_nash = np.prod(np.diag(agent_bundle_values)) ** (1 / m)  # Geometric mean
        
        if current_nash > best_nash_welfare:
            best_nash_welfare = current_nash
    
    return best_nash_welfare


def best_utilitarian_welfare(valuation_matrix):
    """Calculate best possible utilitarian welfare"""
    best_utilitarian_welfare = np.sum(np.max(valuation_matrix, axis=0))
    return best_utilitarian_welfare
