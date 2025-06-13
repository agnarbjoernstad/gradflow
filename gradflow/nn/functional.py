from gradflow import Tensor, zeros_like, ones_like
import numpy as np
from typing import Any, Optional, Union


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    r = input.__array_ufunc__(
        lambda x: np.where(x >= 0, x, zeros_like(x)),
        "__call__",
        input,
        derivative_functions=(
            lambda p_g, x: p_g * np.where(x >= 0, ones_like(x), zeros_like(x)),
        ),
    )
    if inplace:
        Tensor.swap(r, input)
        return input
    return r


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def softmax(input: Tensor, dim: int, dtype: Optional[Any] = None) -> Tensor:
    return input.softmax(dim, dtype)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    mat_mul = input @ weight
    if bias is not None:
        return mat_mul + bias
    return mat_mul


def norm(
    input: Tensor,
    p: Optional[Union[float, str]] = "fro",
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Tensor:
    return input.norm(p=p, dim=dim, keepdim=keepdim)


def mse_loss(
    input: Tensor,
    target: Tensor,
) -> Tensor:
    return (input - target).pow(2).mean()


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    return (-target * input.log()).mean()
