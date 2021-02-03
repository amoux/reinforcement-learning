import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True) -> None:
        super(NoisyLinear, self).__init__()
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Parameter weights initialization.
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform(-std, std)
        self.bias.data.uniform(-std, std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        outputs = F.linear(
            input, self.weight + self.sigma_weight * self.epsilon_weight, bias
        )
        return outputs


class SimpleFFDQN(nn.Module):
    def __init__(self, obs_length: int, num_actions: int) -> None:
        super(SimpleFFDQN, self).__init__()
        self.fc_val = nn.Sequential(
            nn.Linear(obs_length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(obs_length, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_val = self.fc_val(input)
        x_adv = self.fc_adv(input)
        outputs = x_val + x_adv - x_adv.mean(dim=1, keepdim=True)
        return outputs


class DQNConv1dSM(nn.Module):
    def __init__(self, input_size: Tuple[int, ...], num_actions: int) -> None:
        super(DQNConv1dSM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size[0], out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5),
            nn.ReLU(),
        )
        output_size = self._get_conv_output(input_size)
        self.fc_val = nn.Sequential(
            nn.Linear(output_size, 512), nn.ReLU(), nn.Linear(512, 1),
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(output_size, 512), nn.ReLU(), nn.Linear(512, num_actions),
        )

    def _get_conv_output(self, input_size: Tuple[int, ...]) -> int:
        output = self.conv(torch.zeros(1, *input_size))
        return int(np.prod(output.size()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv(input).view(input.size()[0], -1)
        x_val = self.fc_val(x)
        x_adv = self.fc_adv(x)
        outputs = x_val + x_adv - x_adv.mean(dim=1, keepdims=True)
        return outputs


class DQNConv1dXL(nn.Module):
    def __init__(self, input_size: Tuple[int, ...], num_actions: int) -> None:
        super(DQNConv1dXL, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size[0], 32, kernel_size=3),
            nn.MaxPool1d(3, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, out_channels=32, kernel_size=3),
            nn.MaxPool1d(3, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, out_channels=32, kernel_size=3),
            nn.MaxPool1d(3, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, out_channels=32, kernel_size=3),
            nn.MaxPool1d(3, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, out_channels=32, kernel_size=3),
            nn.ReLU(),
        )
        output_size = self._get_conv_output(input_size)
        self.fc_val = nn.Sequential(
            nn.Linear(output_size, 512), nn.ReLU(), nn.Linear(512, 1),
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(output_size, 512), nn.ReLU(), nn.Linear(512, num_actions),
        )

    def _get_conv_output(self, input_size: Tuple[int, ...]) -> int:
        output = self.conv(torch.zeros(1, *input_size))
        return int(np.prod(output.size()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv(input).view(input.size()[0], -1)
        x_val = self.fc_val(x)
        x_adv = self.fc_adv(x)
        outputs = x_val + x_adv - x_adv.mean(dim=1, keepdims=True)
        return outputs
