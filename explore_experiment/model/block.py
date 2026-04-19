import torch
import torch.nn as nn

# バイアスブロック
class Bias(nn.Module):
    def __init__(self, shape: int) -> None:
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.bias


# ニューラルネットワーク構築class
class ResNetBlock(nn.Module):
    def __init__(self, channels: int, activation_function: lambda x: torch.Tensor):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation_function(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return self.activation_function(out + x)
