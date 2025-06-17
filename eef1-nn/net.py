import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUpConv(nn.Module):
    def __init__(self, in_channels, conv_activation=nn.Tanh, up_activation=nn.Tanh):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            conv_activation(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            conv_activation(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            conv_activation(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            conv_activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            up_activation(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            up_activation(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            up_activation(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, padding=1),
            up_activation(),
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class EEF1NN(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.eef1 = nn.Sequential(
            ConvUpConv(in_channels),
            ConvUpConv(1),
            ConvUpConv(1),
        )

    def forward(self, x, temperature=1.0):
        x = self.eef1(x)
        x = x.squeeze(1)
        x = torch.softmax(x / temperature, dim=2)
        return x