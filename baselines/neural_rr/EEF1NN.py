import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from layers import ConvUpConvBlock
from evaluation_metrics import EF1Hard

def bin_argmax(X):
    if X.ndim == 2:
        X = X.unsqueeze(0)
    indecies = torch.argmax(X, dim=1)
    y = torch.zeros_like(X).scatter(1, indecies.unsqueeze(1), 1).int()
    return y

class EEF1NN(nn.Module):
    def __init__(self,
                 temperature:float=0.01,
                 *args, **kwargs) -> None:
        super().__init__()
        self.temperature = temperature

        self.blocks = nn.Sequential(
            ConvUpConvBlock(in_channels=6),
            ConvUpConvBlock(in_channels=1),
            ConvUpConvBlock(in_channels=1),
        )
        self.fractional_allocation = nn.Softmax(dim=2) # agent-wise softmax

    def transform_inputs(self, D: torch.Tensor) -> torch.Tensor:
        if D.ndim == 2:
            D = D.unsqueeze(0)
        
        B, n, m = D.shape
        I = torch.zeros(size=(B, 6, n, m), device=D.device, dtype=D.dtype)

        # Assign first channel
        I[:, 0] = D

        # Compute max along the first dimension
        max_data, ids = torch.max(D, dim=1, keepdim=True)

        # Scatter max_data to appropriate positions
        X = torch.zeros_like(D).scatter_(dim=1, index=ids, src=max_data)

        for c in range(1, 6):
            mask = torch.arange(m, device=D.device) % 5 == c - 1
            I[:, c] = X * mask.unsqueeze(0).unsqueeze(0)
        
        return I

    def forward(self, X:Tensor):
        X = self.transform_inputs(X)
        X = self.blocks(X)
        X = X / self.temperature
        X = self.fractional_allocation(X)
        X = X.squeeze(dim=1)
        return X
        
    def predict(self, X:Tensor):
        y = self.forward(X)
        return bin_argmax(y)
