import torch
from torch import Tensor
import torch.nn as nn
import math

class RR(nn.Module):
    def __init__(self):
        super().__init__()

    def __mask_bin_argmax(self, a, b):
        masked_indices = torch.nonzero(b, as_tuple=True)[0]

        if len(masked_indices) == 0:
            return torch.zeros_like(a, dtype=torch.float)

        max_index = masked_indices[torch.argmax(a[masked_indices])]

        mask = torch.zeros_like(a, dtype=torch.float)
        mask[max_index] = 1.0

        return mask

    def round_robin(self, V: Tensor):
        assert V.ndim == 2, "V should be a 2D tensor"
        n, m = V.shape
        num_rounds = math.ceil(m / n)
        pi = torch.zeros(size=(n, m), dtype=torch.float, device=V.device)

        rest = torch.zeros(m, dtype=torch.bool, device=V.device)

        for _ in range(num_rounds):
            for i in range(n):
                mask = self.__mask_bin_argmax(V[i], ~rest)
                pi[i] += mask
                rest = rest | mask.bool()

        return pi

    def forward(self, X: Tensor):
        if X.ndim == 2:
            return self.round_robin(V=X)
        elif X.ndim == 3:
            return torch.stack([self.round_robin(V=X[i]) for i in range(len(X))])

    def predict(self, X: Tensor):
        y = self.forward(X)
        return y.to(torch.int)
