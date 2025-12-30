import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn.init as init
import math

class SoftRR(nn.Module):
    def __init__(self, tau:float=1.) -> None:
        super().__init__()
        self.tau = tau

    def __one_round(self, M:Tensor):
        assert M.ndim == 2, "The tensor must be 2-dimensional"
        n, m = M.shape

        R = []
        c = torch.ones(m, device=M.device) if self.training else None
        c_set = torch.tensor([True for _ in range(m)], device=M.device) if not self.training else None
        for i in range(n):
            if self.training:
                y = F.softmax((M[i] - torch.min(M[i]) + 1) * c / self.tau, dim=0)
                c = (1 - y) * c
            else:
                if torch.any(c_set):
                    y = torch.argmax(torch.where(c_set, M[i], -float('inf')))
                    y = F.one_hot(y, num_classes=m)
                else:
                    y = torch.zeros_like(M[i])
                c_set = torch.logical_and(c_set, (1-y).to(torch.bool))
                y = y.float()
            R.append(y)
        return torch.vstack(R)

    def forward(self, V:Tensor) -> Tensor:
        assert V.ndim == 2, "V should be a 2D tensor"
        n, m = V.shape
        num_rounds = math.ceil(m / n)
        pi = torch.zeros(size=(n, m), dtype=torch.float, device=V.device)

        V = V.repeat(num_rounds, 1)
        pi = self.__one_round(V)
        pi = pi.view(num_rounds, n, m).sum(dim=0)
        return pi


class ConvUpConvBlock(nn.Module):
    def __init__(self, in_channels:int=6):
        super().__init__()
        # Convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Up-convolution
        self.upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

        # tanh activation
        self.tanh = nn.Tanh()

        # Xavier Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        # Apply conv
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.conv3(x))
        x = self.tanh(self.conv4(x))

        # Apply up-conv
        x = self.tanh(self.upconv1(x))
        x = self.tanh(self.upconv2(x))
        x = self.tanh(self.upconv3(x))
        x = self.tanh(self.upconv4(x))

        return x
