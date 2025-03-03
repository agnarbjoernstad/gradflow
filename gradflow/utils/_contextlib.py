from typing import Any
from contextlib import ContextDecorator


class _DecoratorContextManager(ContextDecorator):
    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        return self.__class__()


class _NoParamDecoratorContextManager(_DecoratorContextManager):
    def __new__(cls, orig_func=None):
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)
