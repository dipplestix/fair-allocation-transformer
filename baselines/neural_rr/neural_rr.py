import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from softsort import SoftSort
from layers import SoftRR
from EEF1NN import bin_argmax

class NeuralRR(nn.Module):
    def __init__(self, 
                 softsort_tau:float=1.0, 
                 softsort_pow:float=2.0,
                 srr_tau:float=1.0,
                 *args, **kwargs) -> None:
        super().__init__()
        
        self.fcU1 = nn.Linear(5, 20)
        self.fcU2 = nn.Linear(20, 1)
        self.tie_break = lambda x: x + torch.argsort(torch.argsort(x))
        self.soft_sort = SoftSort(tau=softsort_tau, pow=softsort_pow)
        self.srr = SoftRR(tau=srr_tau)

    def __forward(self, V:Tensor, require_permutations:bool=False) -> Tensor:
        assert V.ndim == 2
        # Compute hard permutation in evaluation
        if not self.training:
            self.soft_sort.hard = True

        # Embedding
        n, m = V.shape
        U, _, _ = torch.svd(V)
        U = torch.cat([V.max(dim=1, keepdim=True).values, V.min(dim=1, keepdim=True).values, U[:, :3]], dim=1)
        score = self.fcU2(F.sigmoid(self.fcU1(U))).view(1,-1)
        score = self.tie_break(score)

        P_hat = self.soft_sort.forward(score)[0]
        V_hat = torch.matmul(P_hat, V)
        pi_hat = self.srr.forward(V=V_hat)
        pi_hat = torch.matmul(P_hat.T, pi_hat)
        pi_hat = pi_hat / pi_hat.sum(dim=0)

        if require_permutations:
            return pi_hat, P_hat
        else:
            return pi_hat

    def forward(self, V:Tensor, require_permutations:bool=False) -> Tensor:
        if V.ndim == 2: # Single input
            return self.__forward(V, require_permutations=require_permutations)
        elif V.ndim == 3: # Batched input
            outputs = [self.__forward(V[b], require_permutations=require_permutations) for b in range(len(V))]
            if require_permutations:
                pi_hats = [pi_hat for (pi_hat, _) in outputs]
                P_hats = [P_hat for (_, P_hat) in outputs]
                return torch.stack(pi_hats), torch.stack(P_hats)
            else:
                return torch.stack(outputs)
        else:
            assert False
    
    def predict(self, V:Tensor, require_permutations:bool=False) -> Tensor:
        outputs = None
        if require_permutations:
            pi_hats, P_hats = self.forward(V=V, require_permutations=require_permutations)
            pi_hats = bin_argmax(pi_hats).to(torch.int)
            outputs = (pi_hats, P_hats)
        else:
            pi_hats = self.forward(V=V, require_permutations=require_permutations)
            outputs = bin_argmax(pi_hats).to(torch.int)

        return outputs
