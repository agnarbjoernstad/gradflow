from gradflow.tensor import Tensor, zeros_like, ones_like
from typing import Any, Optional
import numpy as np


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
