import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from problem import Problem
from solver import Solver


class Example:
    def __init__(self, S: np.ndarray, K: np.ndarray, k: int):
        self.k = k
        self.S = S
        self.K = K
example1 = pd.read_pickle(r"examples_01.pkl")
# example2 = pd.read_pickle(r"examples_02.pkl")


goal_epsilon = 1e-2
ex = example1[0]
p = Problem(ex.S, ex.k, np.ones(ex.S.shape))
solver = Solver(p.objective, p.gradient, p.hessian, p.output)

K_approx = solver.solve(p.init)
print(np.linalg.norm(ex.K - K_approx))
print(np.linalg.norm(ex.K - K_approx) <= goal_epsilon)

print(pd.DataFrame(K_approx))
df = pd.DataFrame(solver.recorder).T
df.plot(figsize=(8,5))
plt.show()