# GradFlow


[![numpy](https://img.shields.io/badge/powered_by-numpy-blue)](https://github.com/numpy/numpy)
![test](https://github.com/agnarbjoernstad/gradflow/actions/workflows/test.yml/badge.svg)


## Roadmap
 - [X] Basic tensor operations
 - [X] nn.Module
 - [X] nn.Sequential
 - [X] nn.Linear
 - [X] nn.MSELoss
 - [X] nn.BCELoss
 - [X] Batch
 - [ ] nn.BCEWithLogitsLoss
 - [ ] nn.CrossEntropyLoss
 - [X] optim.Adam
 - [X] Train on MNIST
 - [ ] nn.Conv1D
 - [ ] nn.Conv2D
 - [ ] nn.Dropout
 - [ ] nn.Flatten
 - [ ] nn.Reshape
 - [ ] nn.BatchNorm1d
 - [ ] nn.BatchNorm2d
 - [ ] nn.LayerNorm
 - [ ] nn.RMSNorm
 - [ ] nn.Transformer
 - [ ] Train on Titanic
 - [ ] Loading of weights



## Installation

1. Clone the repository: ```git clone https://github.com/agnarbjoernstad/gradflow```
2. Install with pip: ```python3 -m pip install .```

## Examples

Check the installed version of the package:

```python
import gradflow as gf


print(gf.__version__)
```

Example optimization problem:
```python
import gradflow as gf


w = gf.tensor([1, 2, 3], dtype=float)
x = gf.tensor([2, 0.5, 1], dtype=float, requires_grad=False)
lr = 0.5

for i in range(10):
    loss = (w.softmax(dim=0) * x).sum()
    loss.backward()
    w = w - lr * w.grad
    w.grad = None
    print(f"Loss: {loss}, w: {w}")
```

### Train on MNIST

```python3 -m script.mnist```

### Fit sine function with a neural network
```python3 -m script.sin```

<video width=640 src='https://github.com/user-attachments/assets/a69488cc-ef97-4cc8-b09d-8c3d99aded28' />
