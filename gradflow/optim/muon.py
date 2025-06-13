from .optimizer import Optimizer
from typing import Union, Any
from gradflow import Tensor, zeros_like
from gradflow.autograd.grad_mode import no_grad
import gradflow as gf
import numpy as np


class Muon(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 1e-2,
        momentum: Union[float, Tensor] = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0,
        steps: int = 5,
        *,
        maximize: bool = False,
    ):
        self.moment = {}
        defaults = dict(
            lr=lr,
            momentum=momentum,
            eps=eps,
            weight_decay=weight_decay,
            steps=steps,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None or p.requires_grad is False:
                continue
            if self.defaults["maximize"]:
                grad = -p.grad
            else:
                grad = p.grad

            if self.defaults["weight_decay"] > 1e-12:
                grad += self.defaults["weight_decay"] * p

            prev_moment = self.moment.get(id(p), zeros_like(grad))
            moment = self.defaults["momentum"] * prev_moment + grad
            self.moment[id(p)] = moment
            step = Muon.newton_schulz5(
                moment, self.defaults["steps"], self.defaults["eps"]
            )

            p -= self.defaults["lr"] * step

    @staticmethod
    def newton_schulz5(G: gf.Tensor, steps=5, eps=1e-7):
        """
        Newton-Schulz iteration for matrix square root approximation.
        Implementation from https://kellerjordan.github.io/posts/muon/
        """

        assert G.ndim == 2

        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.astype(np.float32)
        X = X / (X.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X
