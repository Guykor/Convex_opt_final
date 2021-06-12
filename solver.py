import numpy as np


class Solver:
    def __init__(self, objective, graident, hessian, out_func):
        self.f = objective
        self.gf = graident
        self.hf = hessian
        self.out = out_func
        self.recorder = {}

    LR = [0.5, 0.3, 0.2, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    def line_search(self, current_sol, grad_val):
        min_res = np.inf
        min_lr = 1
        for lr in self.LR:
            f_val = self.f(current_sol - lr * grad_val)
            if ~np.isnan(f_val) and f_val < min_res:
                min_res = f_val
                min_lr = lr
        return lr

    def progress(self, iter, lr, previous_grad_val, grad_val, eps):
        cosine = 0
        if iter > 1:
            cosine = prod_trace(previous_grad_val.T, grad_val) / (
                        np.linalg.norm(previous_grad_val) * np.linalg.norm(grad_val))
        self.recorder[iter] = {"lr": lr, "diff": eps, "cosine_gradient_similarity": cosine}

    def grad_descent(self, X0, max_iter):
        X, X_tag = 0, X0
        eps = np.inf
        iter = 0
        previous_gradient = None
        while eps > 1e-8 and iter < max_iter:
            iter += 1
            grad = self.gf(X_tag)
            lr = self.line_search(X_tag, grad)
            X = X_tag - lr * grad
            eps = np.linalg.norm(X - X_tag)
            X_tag = X
            self.progress(iter, lr, previous_gradient, grad, eps)
            previous_gradient = grad

        print(f"Done: iteration {iter}")
        return self.out(X)

    def solve(self, X0, max_iter=10_000):
        return self.grad_descent(X0, max_iter)