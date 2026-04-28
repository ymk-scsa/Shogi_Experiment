from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from shogi.halfkav2_feature import SPEC


class SCReLU(nn.Module):
    def __init__(self, clip: float = 1.0) -> None:
        super().__init__()
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(x, 0.0, self.clip)
        return y * y


class NnueLargeModel(nn.Module):
    """HalfKAv2 入力を受ける NNUE-Large 風 value-only ネットワーク。"""

    def __init__(
        self,
        feature_dim: int = SPEC.total_feature_dim,
        accum_dim: int = 1024,
        hidden1: int = 1024,
        hidden2: int = 512,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.accumulator = nn.EmbeddingBag(feature_dim, accum_dim, mode="sum")
        self.fc1 = nn.Linear(accum_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.act = SCReLU(1.0)

    def forward_sparse_batched(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """EmbeddingBag 用に平坦化済み indices と offsets（バッチ長 batch_size）で順伝播。"""
        a = self.accumulator(indices, offsets)
        h1 = self.act(self.fc1(a))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2)

    def forward_sparse(self, indices_list: Sequence[torch.Tensor]) -> torch.Tensor:
        device = self.fc1.weight.device
        flattened = []
        offsets = [0]
        for x in indices_list:
            xi = x.to(device=device, dtype=torch.long).view(-1)
            flattened.append(xi)
            offsets.append(offsets[-1] + int(xi.numel()))
        if not flattened:
            return torch.zeros((0, 1), device=device, dtype=torch.float32)

        values = torch.cat(flattened, dim=0)
        offs = torch.tensor(offsets[:-1], device=device, dtype=torch.long)
        a = self.accumulator(values, offs)
        h1 = self.act(self.fc1(a))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2)
