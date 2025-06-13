from gradflow.optim.optimizer import Optimizer
from typing import List


class CombinedOptimizer(Optimizer):
    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers
        super().__init__({}, {})

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
