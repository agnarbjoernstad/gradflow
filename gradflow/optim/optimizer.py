from typing import Any


class Optimizer:
    def __init__(self, params, defaults: dict[str, Any]) -> None:
        self.params = [p for p in params]
        self.defaults = defaults

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
