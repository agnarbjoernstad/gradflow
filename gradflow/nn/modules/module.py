from typing import Dict, Optional
import gradflow


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Module:
    training: bool
    _parameters: Dict[str, Optional[gradflow.Tensor]]
    _modules: Dict[str, Optional["Module"]]

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_modules", {})

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        self._modules[name] = module

    def _get_name(self):
        return self.__class__.__name__

    def _register_parameter(self, name: str, param: Optional[gradflow.Tensor]) -> None:
        self._parameters[name] = param

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def extra_repr(self) -> str:
        return ""

    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            for param in module.parameters():
                yield param
