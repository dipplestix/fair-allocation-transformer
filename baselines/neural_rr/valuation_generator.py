from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
from typing import Literal

class ValuationGenerator(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(self, size) -> Tensor:
        raise NotImplementedError

class AverageNoise(ValuationGenerator):
    def __init__(self, low:float=1.0, high:float=2.0, eps=1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.low = low
        self.high = high
        assert low < high, "`low` is lower than `high`"

    def generate(self, size) -> Tensor:
        N, n, m = size
        mu = self.low + (self.high-self.low) * torch.rand(size=(N, n, 1))
        V = mu + torch.rand(size=(N, n, m)) * self.eps
        V = torch.clamp(V, self.low, self.high)
        assert V.shape == size

        return V
