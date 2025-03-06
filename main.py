import gradflow as gf


w = gf.tensor([1, 2, 3], dtype=float)
x = gf.tensor([2, 0.5, 1], dtype=float, requires_grad=False)
lr = 0.5

for i in range(10):
    loss = (w.softmax(dim=0) * x).sum()
    loss.backward()
    w = w - lr * w.grad
    print(f"Loss: {loss}, w: {w}")
