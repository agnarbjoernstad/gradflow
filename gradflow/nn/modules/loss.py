from gradflow import Tensor
from gradflow.nn import functional as F
from .module import Module


class _Loss(Module):
    reduction: str

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction


class MSELoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)


class CrossEntropyLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target)
