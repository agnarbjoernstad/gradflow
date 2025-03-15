from typing import Any, defaultdict, OrderedDict, cast
from gradflow import Tensor


class Optimizer:

    def __init__(self, params, defaults: dict[str, Any]) -> None:
        self.params = params
        self.defaults = defaults
