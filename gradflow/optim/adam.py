from .optimizer import Optimizer
from typing import Optional, Union, Any
from gradflow import Tensor, zeros_like
from gradflow.autograd.grad_mode import no_grad
import gradflow as gf


class Adam(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        decoupled_weight_decay: bool = False,
    ):
        self.moment = {}
        self.velocity = {}
        self.velocity_max = {}
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)

    # @no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:  # or p.requires_grad is False:
                # print(p.grad, p.requires_grad)
                continue
            print("grad")
            if self.defaults["maximize"]:
                grad = -p.grad
            else:
                grad = p.grad

            if self.defaults["weight_decay"] > 1e-12:
                grad += self.defaults["weight_decay"] * p

            prev_moment = self.moment.get(id(p), zeros_like(grad))
            moment = (
                self.defaults["betas"][0] * prev_moment
                + (1 - self.defaults["betas"][0]) * grad
            )
            self.moment[id(p)] = moment

            prev_velocity = self.velocity.get(id(p), zeros_like(grad))
            velocity = (
                self.defaults["betas"][1] * prev_velocity
                + (1 - self.defaults["betas"][1]) * grad**2
            )
            self.velocity[id(p)] = velocity

            moment_hat = moment / (1 - self.defaults["betas"][0])
            velocity_hat = velocity / (1 - self.defaults["betas"][1])

            if self.defaults["amsgrad"]:
                velocity_hat_max = gf.maximum(
                    self.velocity_max.get(id(p), zeros_like(grad)), velocity_hat
                )
                self.velocity[id(p)] = velocity_hat_max
                p -= (
                    self.defaults["lr"]
                    * moment_hat
                    / (velocity_hat_max.sqrt() + self.defaults["eps"])
                )
            else:
                diff = (
                    self.defaults["lr"]
                    * moment_hat
                    / (velocity_hat.sqrt() + self.defaults["eps"])
                )

                p -= diff
