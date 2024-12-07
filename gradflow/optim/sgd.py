from gradflow.autograd.grad_mode import no_grad
from gradflow.optim.optimizer import Optimizer
from gradflow.tensor import Tensor
from typing import Any, Dict, List


class SGD(Optimizer):
    def __init__(
        self,
        params: List[Tensor] | Dict[str, Any],
        lr: float = 0.01,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        maximize: bool = False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    @no_grad
    def step(self):
        # TODO: Implement real SGD.
        for param_group in self.param_groups:
            for p in param_group["params"]:
                p.update_values(p - self.defaults["lr"] * p.grad)
