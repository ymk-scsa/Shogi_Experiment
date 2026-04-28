from typing import Callable

import torch
import torch.nn.functional as F

ActivationFn = Callable[[torch.Tensor], torch.Tensor]


def soft_sin(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x / (1.0 + torch.abs(x))


def scaled_arc_tanh(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x / (1.0 + torch.abs(x))


def quadratic(x: torch.Tensor) -> torch.Tensor:
    return x * (2.0 - x)


def cube_root(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3.0)


def squash_two(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x / (1.0 + torch.abs(x))


ACTIVATION_FUNCTION_SETTING: dict[str, ActivationFn] = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "soft_sin": soft_sin,
    "scaled_arc_tanh": scaled_arc_tanh,
    "quadratic": quadratic,
    "cube_root": cube_root,
    "squash_two": squash_two,
    "gelu": F.gelu,
}

# 以前の番号指定にも対応（互換性維持）
_ACTIVATION_FUNCTION_INDEX = [
    "relu",
    "soft_sin",
    "scaled_arc_tanh",
    "quadratic",
    "cube_root",
    "squash_two",
    "leaky_relu",
    "gelu",
]


def available_activation_function_names() -> list[str]:
    return list(ACTIVATION_FUNCTION_SETTING.keys())


def resolve_activation_function(name_or_index: str) -> ActivationFn:
    key = str(name_or_index).strip().lower()
    if key in ACTIVATION_FUNCTION_SETTING:
        return ACTIVATION_FUNCTION_SETTING[key]
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(_ACTIVATION_FUNCTION_INDEX):
            return ACTIVATION_FUNCTION_SETTING[_ACTIVATION_FUNCTION_INDEX[idx]]
    names = ", ".join(available_activation_function_names())
    raise ValueError(f"unknown activation_function '{name_or_index}'. available: {names}")
