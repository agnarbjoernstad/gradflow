from gradflow.tensor import Tensor
from gradflow.optim.sgd import SGD
from gradflow.nn.modules.loss import MSELoss


if __name__ == "__main__":
    t = Tensor.tensor([1, 2, 3], dtype=float)
    optim = SGD([t])

    # for i in range(2000):
    #    loss = MSE()(t, Tensor.tensor([-2, 3, 0], dtype=float).T)
    #    # print(i, loss)
    #    optim.zero_grad()
    #    loss.backward()
    #    print("current grad", t.grad)
    #    optim.step()
    #    print(t, loss)

    i = Tensor.tensor([[1, 4], [1, 2]], dtype=float).reshape(2, 2)
    optim = SGD([i])

    for idx in range(1000):
        t = Tensor.tensor([1, 1], dtype=float).reshape(2, 1)
        s = i.T @ t
        j = i[1, 0]
        optim.zero_grad()
        loss = MSELoss()(s, Tensor.tensor([-2, 4], dtype=float).reshape(2, 1)) + j**2
        loss.backward()
        optim.step()
        print(loss, i)
