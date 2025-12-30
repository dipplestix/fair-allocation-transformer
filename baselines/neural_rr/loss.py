from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from typing import Literal, List

class AllocationLoss(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
    
    def _assert_shape_equal(self, outputs: Tensor, labels: Tensor):
        assert outputs.ndim == labels.ndim, f"outputs and labels must have the same dimensions, got: {outputs.ndim=}, {labels.ndim=}"
        assert outputs.shape == labels.shape, f"outputs and labels must have the same shape, got: {outputs.shape=}, {labels.shape=}"

    @abstractmethod
    def forward(self, outputs: Tensor, labels: Tensor, valuations: Tensor=None) -> Tensor:
        raise NotImplementedError

class ItemCrossEntropy(AllocationLoss):
    def __init__(self, reduction: Literal['none', 'mean', 'sum']='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: Tensor, labels: Tensor, valuations: Tensor=None) -> Tensor:
        self._assert_shape_equal(outputs=outputs, labels=labels)

        # Calculate the col-wise cross-entropy loss
        loss = -torch.sum(labels * torch.log(outputs + 1e-9), dim=1)  # Add a small value to avoid log(0)

        # Apply the specified reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class EFViolation(AllocationLoss):
    def __init__(self):
        super().__init__()

    def forward(self, outputs: Tensor, labels: Tensor, valuations: Tensor=None) -> Tensor:
        assert valuations is not None, "valuations must not be empty"
        self._assert_shape_equal(outputs=outputs, labels=labels)

        if outputs.ndim == 2:
            return torch.relu((valuations * outputs.unsqueeze(1)).sum(dim=2, keepdim=True) - (valuations * outputs).sum(dim=1, keepdim=True)).sum()
        elif outputs.ndim == 3:
            return torch.relu(((valuations.unsqueeze(1) * outputs.unsqueeze(2)) - (valuations * outputs).unsqueeze(1))).sum((3,2,1)).mean()

class Regularized(AllocationLoss):
    def __init__(self, lams:List[float], losses:List[AllocationLoss]):
        super().__init__()
        self.lams = lams
        self.losses = nn.ModuleList(losses)

    def forward(self, outputs: Tensor, labels: Tensor, valuations: Tensor=None) -> Tensor:
        return sum([lam * loss(outputs=outputs, labels=labels, valuations=valuations) for (lam, loss) in zip(self.lams, self.losses)])
