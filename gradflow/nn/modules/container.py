from .module import Module
from collections import OrderedDict
from typing import Dict, Iterator


class Sequential(Module):
    _modules: Dict[str, Module]

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> "Sequential":
        self.add_module(str(len(self)), module)
        return self

    def extend(self, sequential) -> "Sequential":
        for layer in sequential:
            self.append(layer)
        return self

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
