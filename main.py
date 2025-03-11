import gradflow as gf
from gradflow.nn.modules import linear, module, container
from gradflow.nn.modules.loss import MSELoss
from gradflow.autograd.grad_mode import no_grad


w = gf.tensor([1, 2, 3], dtype=float)
x = gf.tensor([2], dtype=float, requires_grad=False)

lr = 0.01

l = linear.Linear(1, 10, bias=True)
m = linear.Linear(10, 1, bias=True)
s = container.Sequential(l, m)
loss_fn = MSELoss()


y = gf.tensor([0.5], dtype=float)

for i in range(100):
    p = s(x)
    loss = loss_fn(p, y)
    loss.backward()
    print(f"Loss: {loss}")
    print("p", p, "y", y)
    with no_grad():
        l.weight = l.weight - lr * l.weight.grad
        m.weight = m.weight - lr * m.weight.grad
        l.bias = l.bias - lr * l.bias.grad
        m.bias = m.bias - lr * m.bias.grad
    l.weight.grad = None
    l.bias.grad = None
    m.weight.grad = None
    m.bias.grad = None
