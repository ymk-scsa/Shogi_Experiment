import torch
import torch.nn.functional as F


def soft_sin(x: torch.Tensor) -> torch.Tensor:
    return 2 * x / (1 + abs(x))


def scaled_arc_tahn(x: torch.Tensor) -> torch.Tensor:
    return 2 * x / (1 + abs(x))


def quafratic(x: torch.Tensor) -> torch.Tensor:
    return x * (2 - x)


def cube_root(x: torch.Tensor) -> torch.Tensor:
    return x**1 / 3


def squash_two(x: torch.Tensor) -> torch.Tensor:
    return (2 * x) / (1 + abs(x))


ACTIVATION_FUNCTION_MODE = [
    ACTIVATION_FUNCTION_RELU,
    ACTIVATION_FUNCTION_SOFT_SIN,
    ACTIVATION_FUNCTION_SCALED_ARC_TAHN,
    ACTIVATION_FUNCTION_QUADRATIC,
    ACTIVATION_FUNCTION_CUBE_ROOT,
    ACTIVATION_FUNCTION_SQUASH_TWO,
    ACTIVATION_FUNCTION_LEAKY_RELU,
] = range(7)


ACTIVATION_FUNCTION_SETTING = {
    ACTIVATION_FUNCTION_RELU: F.relu,
    ACTIVATION_FUNCTION_SOFT_SIN: soft_sin,
    ACTIVATION_FUNCTION_SCALED_ARC_TAHN: scaled_arc_tahn,
    ACTIVATION_FUNCTION_QUADRATIC: quafratic,
    ACTIVATION_FUNCTION_CUBE_ROOT: cube_root,
    ACTIVATION_FUNCTION_SQUASH_TWO: squash_two,
    ACTIVATION_FUNCTION_LEAKY_RELU: F.leaky_relu,
}
