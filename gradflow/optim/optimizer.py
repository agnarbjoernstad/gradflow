from typing import Any, Dict, List
from gradflow.tensor import Tensor, zeros_like


class Optimizer:
    def __init__(
        self, params: List[Tensor] | List[Dict[str, Any]], defaults: Dict[str, Any]
    ):
        self.defaults = defaults
        self.param_groups: List[Dict[str, Any]]

        if len(list(params)) == 0:
            raise ValueError("Optimizer got an empty parameter list.")
        if not isinstance(params[0], dict):
            self.param_groups = [{"params": params}]

    def zero_grad(self, set_to_none: bool = True):
        for param_group in self.param_groups:
            for p in param_group["params"]:
                p.grad = zeros_like(p) if not set_to_none else None
