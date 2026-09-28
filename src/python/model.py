"""Excerpt from the training pipeline (mlp_train.py) - the model definition.
Author: Hadi Sarhangi Fard

A plain fully connected network: N Linear layers with ReLU in between.
The hidden sizes (1024, 784, 484) are passed in, so this same class produced
every architecture experiment before the final 784-1024-784-484-10 network.
"""
from collections.abc import Sequence

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """Fully connected network for MNIST digits."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int = 10):
        super().__init__()
        dims = [input_dim, *hidden_dims, num_classes]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:  # no ReLU after the output layer (raw logits)
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten 28x28 images to 784
        return self.net(x)
