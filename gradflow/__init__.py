import importlib
from ._tensor import Tensor, zeros_like, ones_like, tensor, numel  # noqa: F401
from importlib.metadata import version

try:
    __version__ = version("gradflow")
except importlib.metadata.PackageNotFoundError:
    print("Package not installed.")

__all__ = ["Tensor", "zeros_like", "ones_like", "tensor", "numel"]
