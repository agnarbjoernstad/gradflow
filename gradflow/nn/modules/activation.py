from gradflow.tensor import Tensor
import numpy as np


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return np.maximum(x, 0)

    def backward(self, x):
        return (x > 0).astype(float)
