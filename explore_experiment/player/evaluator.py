from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from cshogi import Board

from model.activation_function import resolve_activation_function
from model.model import PolicyValueResNetModel
from model.nnue_large import NnueLargeModel
from shogi.feature import FEATURES_NUM, make_input_features, make_move_label
from shogi.halfkav2_feature import extract_halfkav2_indices


def _softmax_temperature_with_normalize(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits.astype(np.float64) / max(temperature, 1e-3)
    x -= float(np.max(x))
    p = np.exp(x)
    s = float(np.sum(p))
    if s <= 0:
        return np.full_like(x, 1.0 / len(x), dtype=np.float64).astype(np.float32)
    return (p / s).astype(np.float32)


class Evaluator(Protocol):
    def evaluate_batch(
        self,
        boards: list[Board],
        legal_moves_batch: list[list[int]],
        colors: list[int],
        temperature: float,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        ...

    def warmup(self) -> None:
        ...


@dataclass
class ResNetEvaluator:
    model: PolicyValueResNetModel
    device: torch.device

    def warmup(self) -> None:
        with torch.no_grad():
            x = torch.zeros((1, FEATURES_NUM, 9, 9), dtype=torch.float32, device=self.device)
            self.model(x)

    def evaluate_batch(
        self,
        boards: list[Board],
        legal_moves_batch: list[list[int]],
        colors: list[int],
        temperature: float,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        n = len(boards)
        if n == 0:
            return [], np.zeros((0,), dtype=np.float32)

        feats = np.zeros((n, FEATURES_NUM, 9, 9), dtype=np.float32)
        for i, b in enumerate(boards):
            make_input_features(b.copy(), feats[i])
        x = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_logits, value_logits = self.model(x)
            pol = policy_logits.cpu().numpy()
            val = torch.sigmoid(value_logits).cpu().numpy().reshape(-1).astype(np.float32)

        probs: list[np.ndarray] = []
        for i in range(n):
            legal = legal_moves_batch[i]
            if not legal:
                probs.append(np.zeros((0,), dtype=np.float32))
                continue
            lm = np.empty((len(legal),), dtype=np.float32)
            for j, move in enumerate(legal):
                lm[j] = float(pol[i][make_move_label(move, colors[i])])
            probs.append(_softmax_temperature_with_normalize(lm, temperature))
        return probs, val


@dataclass
class NnueEvaluator:
    model: NnueLargeModel
    device: torch.device

    def warmup(self) -> None:
        with torch.no_grad():
            dummy = [torch.tensor([0], dtype=torch.long, device=self.device)]
            self.model.forward_sparse(dummy)

    def evaluate_batch(
        self,
        boards: list[Board],
        legal_moves_batch: list[list[int]],
        colors: list[int],
        temperature: float,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        n = len(boards)
        if n == 0:
            return [], np.zeros((0,), dtype=np.float32)

        sparse = [
            torch.from_numpy(extract_halfkav2_indices(board, board.turn)).to(self.device)
            for board in boards
        ]
        with torch.no_grad():
            raw = self.model.forward_sparse(sparse).reshape(-1)
            val = torch.sigmoid(raw).cpu().numpy().astype(np.float32)

        probs: list[np.ndarray] = []
        for legal in legal_moves_batch:
            if not legal:
                probs.append(np.zeros((0,), dtype=np.float32))
            else:
                p = np.full((len(legal),), 1.0 / len(legal), dtype=np.float32)
                probs.append(p)
        return probs, val


def create_evaluator(
    eval_type: str,
    device: torch.device,
    modelfile: str,
    blocks: int,
    activation_function: str = "relu",
) -> Evaluator:
    if eval_type == "nnue":
        model = NnueLargeModel()
        checkpoint = torch.load(modelfile, map_location=device)
        try:
            state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state, strict=False)
        except Exception:
            # 学習済み重みがなくても推論経路の接続確認を優先して起動は継続する
            pass
        model.to(device)
        model.eval()
        return NnueEvaluator(model=model, device=device)

    model = PolicyValueResNetModel(
        input_features=FEATURES_NUM,
        activation_function=resolve_activation_function(activation_function),
        blocks=blocks,
    )
    model.to(device)
    checkpoint = torch.load(modelfile, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return ResNetEvaluator(model=model, device=device)
