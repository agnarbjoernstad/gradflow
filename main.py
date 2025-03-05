from gradflow.optim.sgd import SGD
from gradflow.nn.modules.loss import MSELoss
import gradflow as gf


if __name__ == "__main__":
    t = gf.tensor([1, 2, 3], dtype=float)
    optim = SGD([t])

    i = gf.tensor([[1, 4], [1, 2]], dtype=float).reshape(2, 2)
    optim = SGD([i])

    for idx in range(1000):
        t = gf.tensor([1, 1], dtype=float).reshape(2, 1)
        s = i.T @ t
        j = i[1, 0]
        optim.zero_grad()
        loss = MSELoss()(s, gf.tensor([-2, 4], dtype=float).reshape(2, 1)) + j**2
        loss.backward()
        optim.step()
        print(loss, i)
