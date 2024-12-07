from gradflow.utils._contextlib import (
    _DecoratorContextManager,
    _NoParamDecoratorContextManager,
)
from typing import Any, Callable
import threading

_thread_local = threading.local()
_thread_local._is_grad_enabled = True


def _set_grad_mode(new_grad_mode: bool) -> None:
    global _thread_local
    _thread_local._is_grad_enabled = new_grad_mode


def is_grad_enabled() -> bool:
    global _thread_local
    return _thread_local._is_grad_enabled


class no_grad(_NoParamDecoratorContextManager):
    def __init__(self) -> None:
        self.prev = False

    def __enter__(self) -> None:
        self.prev = is_grad_enabled()
        _set_grad_mode(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _set_grad_mode(self.prev)


class enable_grad(_NoParamDecoratorContextManager):
    def __enter__(self) -> None:
        self.prev = is_grad_enabled()
        _set_grad_mode(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _set_grad_mode(self.prev)


class set_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None:
        self.prev = is_grad_enabled()
        self.mode = mode

    def __call__(self, orig_func: Callable[..., Any]) -> Callable[..., Any]:
        _set_grad_mode(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        _set_grad_mode(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _set_grad_mode(self.prev)
