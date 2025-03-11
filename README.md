# GradFlow


[![numpy](https://img.shields.io/badge/powered_by-numpy-blue)](https://github.com/numpy/numpy)
![test](https://github.com/agnarbjoernstad/gradflow/actions/workflows/test.yml/badge.svg)


## Roadmap
 - [X] nn.Module
 - [X] nn.Sequential
 - [X] nn.Linear
 - [ ] nn.MSELoss
 - [ ] nn.BCELoss
 - [ ] Batch
 - [ ] nn.BCEWithLogitsLoss
 - [ ] nn.CrossEntropyLoss
 - [ ] optim.Adam
 - [ ] optim.SGD
 - [ ] Train on MNIST
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
 - [ ] Finetuning of a ViT
 - [ ] RoPE



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
    print(f"Loss: {loss}, w: {w}")
```
