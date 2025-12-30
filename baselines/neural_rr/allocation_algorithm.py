import gc
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch import Tensor
import gurobipy as gp
from gurobipy import GRB

class AllocationAlgorithm(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def allocate(self, valuation: Tensor) -> Tensor:
        raise NotImplementedError

class MaximumUtilitarianWelfare(AllocationAlgorithm):
    def __init__(self):
        super().__init__()

    def __solve(self, valuation:np.ndarray):
        try:
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()

            model = gp.Model(name="assignment", env=env)

            num_agents = len(valuation)
            num_objects = len(valuation[0])

            # Add variables
            x = model.addVars(num_agents, num_objects, vtype=GRB.BINARY, name='x')

            # Set objective
            model.setObjective(gp.quicksum(valuation[i,j] * x[i,j] for i in range(num_agents) for j in range(num_objects)), GRB.MAXIMIZE)

            # Each item is assigned to one agent
            model.addConstrs((gp.quicksum(x[i,j] for i in range(num_agents)) == 1 for j in range(num_objects)), name='object')

            # Solve
            model.optimize()

            if model.status in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT]:
                solution = model.getAttr('x', x)
                assignment = [[0.0]*num_objects for _ in range(num_agents)]
                for i in range(num_agents):
                    for j in range(num_objects):
                        if solution[i, j] > 0.5:
                            assignment[i][j] = 1.0
                return torch.tensor(assignment)
            else:
                print("No feasible or optimal solution found.")
                return None
            
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
            return None

        finally:
            # Clean-up
            model.dispose()
            env.dispose()
            del model
            del env
            gc.collect()

    def allocate(self, valuation: Tensor) -> Tensor:
        valuation = valuation.detach().numpy()
        if valuation.ndim == 2:
            allocation = self.__solve(valuation=valuation)
        elif valuation.ndim == 3:
            allocation = torch.stack([self.__solve(valuation=valuation[i]) for i in range(len(valuation))])

        return allocation
