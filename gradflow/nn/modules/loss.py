from typing import Any


class MSELoss:
    def __call__(self, x: Any, y: Any) -> Any:
        return ((y - x) ** 2).mean()
