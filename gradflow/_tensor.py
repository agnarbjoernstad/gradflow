from typing import Any, Callable, List, Optional
from gradflow.autograd.grad_mode import no_grad, is_grad_enabled

from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def numel(t: "Tensor") -> int:
    return np.prod(t.shape)


def tensor(
    t_obj: List[Any] | float | int | bool | np.ndarray,
    dtype: Optional[np.dtypes] = None,
    requires_grad: bool = True,
) -> "Tensor":
    t_np = np.ascontiguousarray(np.array(t_obj, dtype=dtype))
    if dtype is None:
        dtype = t_np.dtype
    t_np = t_np.astype(dtype)
    t = Tensor(dtype=dtype, buffer=t_np, shape=t_np.shape, offset=0)
    t.requires_grad = requires_grad
    return t


def _topological_sort(curr_node: "Tensor", curr_stack, visited_nodes) -> List["Tensor"]:
    if curr_node in curr_stack:
        raise ValueError("Topological sort is not possible for a graph with cycles.")
    if curr_node in visited_nodes:
        return []
    curr_stack.add(curr_node)
    if curr_node.child_tensors is None:
        return [curr_node]

    sorted_nodes = []
    for node in curr_node.child_tensors:
        sorted_nodes.extend(_topological_sort(node, curr_stack, visited_nodes))
    visited_nodes.add(curr_node)
    curr_stack.remove(curr_node)

    sorted_nodes.append(curr_node)

    return sorted_nodes


def topological_sort(curr_node: "Tensor") -> List["Tensor"]:
    return _topological_sort(curr_node, set(), set())


@no_grad
def run_backward(t: "Tensor") -> None:
    topological_order = list(reversed(topological_sort(t)))
    start_index = topological_order.index(t)
    for node in topological_order[start_index:]:
        for idx, derivative_function in enumerate(node.derivative_functions):
            if node.child_tensors[idx].grad is None:
                node.child_tensors[idx].grad = zeros_like(node.child_tensors[idx])

            if node.grad is not None:
                curr_grad = node.grad
            else:
                curr_grad = ones_like(node)
            grad = derivative_function(curr_grad, *node.child_tensors)
            node.child_tensors[idx].grad += grad


class Tensor(np.ndarray):
    def __new__[T](cls: T, *args: Any, **kwargs: Any) -> T:
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __array_finalize__(self, obj, requires_grad: bool = False):
        super().__init__()
        self.gradient = np.zeros(self.shape)
        self.child_tensors = []
        self.derivative_functions = []
        self.grad = None
        self.requires_grad = requires_grad

    def __hash__(self) -> int:
        return hash((str(id(self))))

    def clone(self) -> "Tensor":
        c = tensor(self)
        for k, v in self.__dict__.items():
            setattr(c, k, deepcopy(v))
        return c

    def __array_wrap__(self, *args, **kwargs):
        pass

    def __array_ufunc__(
        self,
        ufunc,
        method,
        *inputs,
        derivative_functions: Optional[Callable] = None,
        child_tensors: Optional[List["Tensor"]] = None,
        only_tensors_as_children: bool = False,
        **kwargs,
    ):
        child_tensors = [] if child_tensors is None else child_tensors
        inputs_as_np = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                inputs_as_np.append(np.array(inp))
                child_tensors.append(inp)
            else:
                inputs_as_np.append(inp)
                if not only_tensors_as_children:
                    child_tensors.append(tensor(inp))
        t = super().__array_ufunc__(ufunc, method, *inputs_as_np, **kwargs)
        if t is NotImplemented:
            raise NotImplementedError(f"ufunc ({ufunc}) not implemented")
        parent_tensor = tensor(t)
        if is_grad_enabled():
            parent_tensor.child_tensors = child_tensors
            parent_tensor.derivative_functions = derivative_functions

            assert (child_tensors is None and derivative_functions is None) or (
                len(child_tensors) == len(derivative_functions)
            )
        return parent_tensor

    def backward(self, *args, **kwargs):
        run_backward(self, *args, **kwargs)

    def __abs__(self):
        return self.__array_ufunc__(
            np.abs,
            "__call__",
            self,
            derivative_functions=(lambda p_g, x: p_g * np.sign(x),),
        )

    def __eq__(self, other):
        return id(self) == id(other)

    def __le__(self, other):
        return self.__array_ufunc__(
            np.less_equal,
            "__call__",
            self,
            other,
            derivative_functions=(
                lambda _, x, __: zeros_like(x),
                lambda _, __, y: zeros_like(y),
            ),
        )

    def __lt__(self, other):
        return self.__array_ufunc__(
            np.less,
            "__call__",
            self,
            other,
            derivative_functions=(
                lambda _, x, __: zeros_like(x),
                lambda _, __, y: zeros_like(y),
            ),
        )

    def __ge__(self, other):
        return self.__array_ufunc__(
            np.greater_equal,
            "__call__",
            self,
            other,
            derivative_functions=(
                lambda _, x, __: zeros_like(x),
                lambda _, __, y: zeros_like(y),
            ),
        )

    def __gt__(self, other):
        return self.__array_ufunc__(
            np.greater,
            "__call__",
            self,
            other,
            derivative_functions=(
                lambda _, x, __: zeros_like(x),
                lambda _, __, y: zeros_like(y),
            ),
        )

    def __truediv__(self, y):
        return self.__array_ufunc__(
            np.divide,
            "__call__",
            self,
            y,
            derivative_functions=(
                lambda p_g, x, y: p_g * ones_like(x) / y,
                lambda p_g, x, y: -p_g * x / (y**2),
            ),
        )

    def __rtruediv__(self, y):
        return self.__array_ufunc__(
            np.divide,
            "__call__",
            y,
            self,
            derivative_functions=(
                lambda p_g, x, y: p_g * ones_like(x) / y,
                lambda p_g, x, y: -p_g * x / (y**2),
            ),
        )

    def __mul__(self, y):
        return self.__array_ufunc__(
            np.multiply,
            "__call__",
            self,
            y,
            derivative_functions=(
                lambda p_g, x, y: p_g * ones_like(x) * y,
                lambda p_g, x, y: p_g * x * ones_like(y),
            ),
        )

    def __rmul__(self, y):
        return self.__mul__(y)

    def __sub__(self, y):
        return self.__array_ufunc__(
            np.subtract,
            "__call__",
            self,
            y,
            derivative_functions=(
                lambda p_g, x, _: p_g * ones_like(x),
                lambda p_g, _, y: -p_g * ones_like(y),
            ),
        )

    def __add__(self, y):
        return self.__array_ufunc__(
            np.add,
            "__call__",
            self,
            y,
            derivative_functions=(
                lambda p_g, x, _: p_g * ones_like(x),
                lambda p_g, _, y: p_g * ones_like(y),
            ),
        )

    def __radd__(self, x):
        return self.__add__(x)

    def __iadd__(self, y):
        return self.__add__(y)

    def __pow__(self, p):
        return self.__array_ufunc__(
            np.power,
            "__call__",
            self,
            p,
            derivative_functions=(
                lambda p_g, x, p: p_g * p * (x ** (p - 1)),
                lambda p_g, x, p: p_g * (x**p) * np.log(x),
            ),
        )

    def __rpow__(self, p):
        return self.__array_ufunc__(
            np.power,
            "__call__",
            p,
            self,
            derivative_functions=(
                lambda p_g, x, p: p_g * p * (x ** (p - 1)),
                lambda p_g, x, p: p_g * (x**p) * np.log(x),
            ),
        )

    def unsqueeze(self, dim: int):
        return self.__array_ufunc__(
            np.expand_dims,
            "__call__",
            self,
            dim,
            derivative_functions=(
                lambda p_g, _, __: p_g.squeeze(dim),
                lambda _, __, ___: None,
            ),
        )

    def squeeze(self, dim: int):
        return self.__array_ufunc__(
            np.squeeze,
            "__call__",
            self,
            dim,
            derivative_functions=(
                lambda p_g, _, __: p_g.unsqueeze(dim),
                lambda _, __, ___: None,
            ),
        )

    def __matmul__(self, y):
        @no_grad
        def der_l(p_g, _, y):
            if len(p_g.size()) == 1:
                p_g = p_g.unsqueeze(0)
            if len(y.size()) == 1:
                y = y.unsqueeze(-1)
            ans = p_g @ y.T
            return ans

        @no_grad
        def der_r(p_g, x, _):
            if len(p_g.size()) == 1:
                p_g = p_g.unsqueeze(0)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
            ans = x.T @ p_g
            return ans

        return self.__array_ufunc__(
            np.matmul,
            "__call__",
            self,
            y,
            derivative_functions=(der_l, der_r),
        )

    def pow(self, p):
        return self**p

    def sqrt(self):
        return self.__array_ufunc__(
            np.sqrt,
            "__call__",
            self,
            derivative_functions=(lambda p_g, x: p_g / (2 * x.sqrt()),),
        )

    def sigmoid(self):
        return 1 / (1 + np.e ** (-self))

    def softmax(self, dim: Optional[int], dtype: Optional[Any] = None):
        if dim is None:
            dim = -1
        return np.e**self / (np.e**self).sum(axis=dim, keepdims=True)

    def __neg__(self):
        return self.__array_ufunc__(
            np.negative,
            "__call__",
            self,
            derivative_functions=(lambda p_g, x: -p_g * ones_like(x),),
        )

    def __imatmul__(self, y):
        return self.__matmul__(y)

    def __repr__(self):
        return np.array(self).__repr__()

    def __str__(self) -> str:
        return np.array(self).__str__()

    @staticmethod
    def swap(t1: "Tensor", t2: "Tensor") -> None:
        c = t1.clone()
        for k, v in t2.__dict__.items():
            setattr(t1, k, deepcopy(v))
        for k, v in c.__dict__.items():
            setattr(t2, k, deepcopy(v))
        t1.update_values(np.array(t2))
        t2.update_values(np.array(c))
        return c

    def __setitem__(self, index, value: int | float | np.ndarray) -> "Tensor":
        def filter_indexes(p_g, x, _):
            grad = zeros_like(x)
            grad[index] = p_g
            return grad

        c = self.clone()
        v = np.array(self)
        v[index] = value
        self.update_values(v)
        self.derivative_functions = [filter_indexes, lambda p_g, _, __: p_g[index]]
        self.child_tensors = [c, value]
        self.derivative_functions = []

    def __getitem__(self, index) -> "Tensor":
        def filter_indexes(p_g, x):
            grad = zeros_like(x)
            grad[index] = p_g
            return grad

        return self.__array_ufunc__(
            super().__getitem__,
            "__call__",
            index,
            derivative_functions=(filter_indexes,),
            child_tensors=[self],
            only_tensors_as_children=True,
        )

    def ascontiguousarray(self):
        return self.__array_ufunc__(
            np.ascontiguousarray,
            "__call__",
            self,
            derivative_functions=(lambda p_g, _: p_g,),
        )

    @property
    def T(self):
        return self.__array_ufunc__(
            np.transpose,
            "__call__",
            self,
            derivative_functions=(lambda p_g, _: p_g.T,),
        )

    def sum(self, *args, **kwargs):
        def der(p_g, x):
            return p_g.sum(*args, **kwargs) * ones_like(x)

        return self.__array_ufunc__(
            np.sum,
            "__call__",
            self,
            *args,
            derivative_functions=(lambda p_g, x: der(p_g, x),),
            **kwargs,
        )

    def mean(self):
        return self.sum() / self.shape[0]

    def update_values(self, new_values) -> None:
        np.copyto(self, new_values)

    def plot_dependency_graph(self, g=None):
        is_root = g is None
        if g is None:
            g = nx.DiGraph()
        g.add_edges_from([(self, child) for child in self.child_tensors])
        for child in self.child_tensors:
            child.print_dependencies(g)

        if is_root:
            plt.figure(figsize=(8, 6))
            pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
            nx.draw(
                g,
                pos,
                with_labels=True,
                node_color="lightblue",
                edge_color="gray",
                node_size=2000,
                font_size=12,
            )
            plt.title("Directed Acyclic Graph (DAG)")
            plt.show()

    @staticmethod
    def rand(*shape, dtype=None):
        return tensor(np.random.rand(*shape), dtype=dtype)

    @staticmethod
    def zeros(*shape, dtype=None):
        return tensor(np.zeros(shape), dtype=dtype)

    def dim(self) -> int:
        return len(self.shape)

    def size(self, dim=None) -> int:
        if dim is None:
            return self.shape
        return self.shape[dim]

    def uniform_(self, from_: float = 0, to_: float = 1):
        return Tensor.rand(*self.shape, dtype=self.dtype) * (to_ - from_) + from_

    def log(self):
        return self.__array_ufunc__(
            np.log,
            "__call__",
            self,
            derivative_functions=(lambda p_g, x: p_g / x,),
        )

    def detach(self):
        c = self.clone()
        c.requires_grad = False
        return c

    def numpy(self):
        if self.requires_grad:
            raise ValueError(
                "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead."
            )
        return np.array(self)


def zeros_like(t, dtype=None) -> Tensor:
    if dtype is None:
        dtype = t.dtype
    return tensor(np.zeros(t.shape, dtype=dtype))


def ones_like(t, dtype=None) -> Tensor:
    if dtype is None:
        dtype = t.dtype
    return tensor(np.ones(t.shape, dtype=dtype))
