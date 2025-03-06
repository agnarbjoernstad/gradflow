from gradflow._tensor import topological_sort
from gradflow.autograd.grad_mode import no_grad
from gradflow.nn.functional import relu, sigmoid, softmax
from typing import Callable, Optional
import gradflow as gf
import pytest
import numpy as np


t_1_f_1 = None
t_1_f_2 = None
t_1_f_3 = None
t_3_f_1 = None
t_3_f_2 = None
t_3_f_3 = None
t_3_f_4 = None
t_3_f_5 = None
t_2x1_f = None
t_2x2_f = None


@pytest.fixture(autouse=True)
def setup():
    global t_1_f_1, t_1_f_2, t_1_f_3, t_3_f_1, t_3_f_2, t_3_f_3, t_3_f_4, t_3_f_5, t_2x1_f, t_2x2_f

    t_1_f_1 = gf.tensor([1], dtype=np.float64)
    t_1_f_2 = gf.tensor([2], dtype=np.float64)
    t_1_f_3 = gf.tensor([3], dtype=np.float64)
    t_3_f_1 = gf.tensor([1, 2, 3], dtype=np.float64)
    t_3_f_2 = gf.tensor([1, 0, 0], dtype=np.float64)
    t_3_f_3 = gf.tensor([7, 2, 8], dtype=np.float64)
    t_3_f_4 = gf.tensor([3, 1, -4], dtype=np.float64)
    t_3_f_5 = gf.tensor([3, 1, 4], dtype=np.float64)
    t_2x1_f = gf.tensor([[3], [2]], dtype=np.float64)
    t_2x2_f = gf.tensor([[1, 2], [4, 5]], dtype=np.float64)


@no_grad
def tet_2x2_fst_tensor_add():
    assert np.isclose(t_3_f_1 + t_3_f_2, np.array(t_3_f_1) + np.array(t_3_f_2)).all()


@no_grad
def test_tensor_subtract():
    assert np.isclose(t_3_f_1 - t_3_f_2, np.array(t_3_f_1) - np.array(t_3_f_2)).all()


@no_grad
def test_tensor_multiply():
    assert np.isclose(t_3_f_1 * t_3_f_2, np.array(t_3_f_1) * np.array(t_3_f_2)).all()


@no_grad
def test_tensor_divide():
    assert np.isclose(t_3_f_1 / t_3_f_3, np.array(t_3_f_1) / np.array(t_3_f_3)).all()


def test_topological_sort():
    t4 = t_1_f_1 + t_1_f_2
    t5 = t4 + t_1_f_3

    order = topological_sort(t5)

    assert len(order) == 5
    assert order.index(t_1_f_1) < order.index(t4)
    assert order.index(t_1_f_2) < order.index(t4)
    assert order.index(t_1_f_3) < order.index(t5)
    assert order.index(t4) < order.index(t5)


def test_topological_sort_only_from_root():
    t4 = t_1_f_1 + t_1_f_2
    t5 = t4 + t_1_f_3

    order = topological_sort(t4)

    assert order.index(t_1_f_1) < order.index(t4)
    assert order.index(t_1_f_2) < order.index(t4)
    assert len(order) == 3
    assert t_1_f_3 not in order
    assert t5 not in order


def test_topological_sort_cycle():
    t4 = t_1_f_1 + t_1_f_2
    t5 = t4 + t_1_f_3

    t_1_f_1.child_tensors.append(t5)

    with pytest.raises(ValueError):
        topological_sort(t4)


def test_topological_sort_reuse():
    t4 = t_1_f_1 + t_1_f_2
    t5 = t_1_f_1 + t4

    order = topological_sort(t5)

    assert len(order) == 4
    assert order.index(t_1_f_1) < order.index(t4)
    assert order.index(t_1_f_2) < order.index(t4)
    assert order.index(t4) < order.index(t5)


@no_grad
def finite_difference(
    x: gf.Tensor, graph: Callable[[gf.Tensor], gf.Tensor], h: float = 1e-12
) -> gf.Tensor:
    with no_grad():
        gradient = gf.zeros_like(x)
        for idx in range(gradient.size):
            _x = gf.tensor(x)
            multi_index = np.unravel_index(idx, x.shape)
            f1 = graph(_x)
            _x[multi_index] += h
            f2 = graph(_x)
            gradient[multi_index] = (f2 - f1) / h
    return gradient


def approximate_gradient(
    t: gf.Tensor,
    graph: Callable[[gf.Tensor], gf.Tensor],
    exact_answer: Optional[np.ndarray] = None,
) -> None:
    r = graph(t)
    r.backward()
    with no_grad():
        estimated_diff = finite_difference(t, graph)
        assert np.isclose(t.grad, estimated_diff, rtol=1e-2, atol=1e-2).all()
        if exact_answer is not None:
            assert np.isclose(t.grad, exact_answer).all()


def test_autograd_l_add():
    approximate_gradient(t_3_f_1, lambda t: (t + t_3_f_2).sum())


def test_autograd_r_add():
    approximate_gradient(t_3_f_1, lambda t: (t_3_f_2 + t).sum())


def test_autograd_l_subtract():
    approximate_gradient(t_3_f_1, lambda t: (t - t_3_f_2).sum())


def test_autograd_r_subtract():
    approximate_gradient(t_3_f_1, lambda t: (t_3_f_2 - t).sum())


def test_autograd_l_multiply():
    approximate_gradient(t_3_f_1, lambda t: (t * t_3_f_2).sum())


def test_autograd_r_multiply():
    approximate_gradient(t_3_f_1, lambda t: (t_3_f_2 * t).sum())


def test_autograd_nominator_divide():
    approximate_gradient(t_3_f_1, lambda t: (t / t_3_f_3).sum())


def test_autograd_denominator_divide():
    approximate_gradient(t_3_f_1, lambda t: (t_3_f_2 / t).sum())


def test_autograd_index_2d_to_scalar():
    approximate_gradient(t_2x2_f, lambda t: t[1, 0])


def test_autograd_index_2d_to_vector():
    approximate_gradient(t_3_f_1, lambda t: t[0])


def test_autograd_index_2d_to_vector_range():
    approximate_gradient(t_3_f_1, lambda t: t[0:1])


def test_autograd_index_2d_to_matrix_range():
    approximate_gradient(t_2x2_f, lambda t: t[0:1, 0:1])


def test_autograd_l_matmul():
    approximate_gradient(t_2x2_f, lambda t: (t @ t_2x1_f).sum())


def test_autograd_r_matmul():
    approximate_gradient(t_2x1_f, lambda t: (t_2x2_f @ t).sum())


def test_autograd_transpose():
    approximate_gradient(t_2x1_f, lambda t: t.T.sum())


def test_autograd_transpose_transpose():
    approximate_gradient(t_2x1_f, lambda t: t.T.T.sum())


def test_autograd_mean():
    approximate_gradient(t_2x1_f, lambda t: t.mean())


def test_autograd_power_base():
    approximate_gradient(t_2x1_f, lambda t: (t**4).mean())


def test_autograd_power_exponent():
    approximate_gradient(t_2x1_f, lambda t: (4**t).mean())


def test_autograd_power_base_exponent():
    approximate_gradient(t_3_f_1, lambda t: (t_3_f_3**t).mean())


def test_autograd_matmul_index():
    approximate_gradient(t_2x1_f, lambda t: (t_2x2_f @ t)[0:1].mean())


def test_autograd_matmul_identical_power_index():
    approximate_gradient(t_2x1_f, lambda t: (t_2x2_f.T @ (np.e**t))[0:1].mean())


def test_autograd_functional_relu():
    with no_grad():
        t = (np.array(t_3_f_4) >= 0).astype(float)
        exact_answer = t / t.shape
    approximate_gradient(t_3_f_4, lambda t: relu(t).mean(), exact_answer)


def test_autograd_functional_relu_inplace():
    with no_grad():
        t = np.ones_like(t_3_f_4, dtype=float)
        exact_answer = t / t.shape

    def f(x):
        relu(x, inplace=True)
        return x.mean()

    approximate_gradient(t_3_f_4, f, exact_answer)

    with no_grad():
        t_3_f_4_before_swap = t_3_f_4.child_tensors[0]
        t = (np.array(t_3_f_4_before_swap) >= 0).astype(float)
        exact_answer = t / t.shape
        assert np.isclose(t_3_f_4_before_swap.grad, exact_answer).all()


def test_autograd_functional_sigmoid():
    approximate_gradient(t_3_f_4, lambda t: sigmoid(t).mean())


def test_autograd_functional_softmax():
    t = t_3_f_1
    for idx in range(t.size):
        _t = t.copy()
        multi_index = np.unravel_index(idx, t.shape)
        approximate_gradient(_t, lambda t: softmax(t, dim=0)[multi_index])
