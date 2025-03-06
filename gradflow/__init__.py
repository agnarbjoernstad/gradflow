import importlib.util
from ._tensor import Tensor, zeros_like, ones_like, tensor  # noqa: F401
from importlib.metadata import version
import importlib

try:
    __version__ = version("gradflow")
except importlib.metadata.PackageNotFoundError:
    print("Package not installed.")

__all__ = ["Tensor", "zeros_like", "ones_like", "tensor"]
