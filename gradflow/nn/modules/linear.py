from gradflow.nn.modules.module import Module
from gradflow.nn import init
import gradflow.nn.functional as F
from gradflow import Tensor
import math
from typing import Optional


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):

    in_features: int
    out_features: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, dtype=None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.zeros(self.in_features, self.out_features)
        if bias:
            self.bias = Tensor.zeros(self.out_features)
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
