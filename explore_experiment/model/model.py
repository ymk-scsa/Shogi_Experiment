import torch
import torch.nn as nn
import torch.nn.functional as F

from model.block import ResNetBlock, Bias
from shogi.feature import FEATURES_NUM, MOVE_PLANES_NUM, MOVE_LABELS_NUM


class PolicyValueResNetModel(nn.Module):
    def __init__(
        self,
        input_features: int = FEATURES_NUM,
        activation_function: nn.Module = F.relu,
        blocks: int = 10,
        channels: int = 192,
        policy_channels: int = MOVE_PLANES_NUM,
        value_step_channels: int = 256,
    ):
        super(PolicyValueResNetModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_features, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)

        # resnet blocks
        self.blocks = nn.Sequential(*[ResNetBlock(channels, activation_function) for _ in range(blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=policy_channels, kernel_size=1, bias=False)
        self.policy_bias = Bias(MOVE_LABELS_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=policy_channels, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(policy_channels)
        self.value_fc1 = nn.Linear(MOVE_LABELS_NUM, value_step_channels)
        self.value_fc2 = nn.Linear(value_step_channels, 1)

        # activation function
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.activation_function(self.norm1(x))

        # resnet blocks
        x = self.blocks(x)

        # policy head
        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        # value head
        value = self.activation_function(self.value_norm1(self.value_conv1(x)))
        value = self.activation_function(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)

        return policy, value
