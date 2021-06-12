import numpy as np
from scipy.linalg import sqrtm
import scipy

stable_vec = lambda X: np.ones(X.shape[0]) * 1e-12

def stable_diagonal_vec(X):
    return np.diag(X) + stable_vec(X)


def prod_trace(A, B):
    return (A * B).sum()


def tri_inverse(L):
    return scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)


def zeros_K(S):
    return np.zeros(S.shape)


def inv_S(S, k):
    return scipy.linalg.pinvh(S)


class Problem:
    def __init__(self, S, k, L0):
        self.S = S
        self.k = k
        self.init = self.projection(L0)

    def objective(self, L, lambda_=1):
        #
        L = self.projection(L)

        trace = prod_trace(self.S, L @ L.T)
        log_det = 2 * np.log(stable_diagonal_vec(L)).sum()

        # Sanity check:
        S = self.S
        if np.abs(trace - np.trace(S @ L @ L.T)) > 1e-2:
            print("ERROR, trace mismatch")
            print("ours: ", trace, " vs ", np.trace(S @ L @ L.T))

        if np.abs(np.linalg.slogdet(L @ L.T)[1] - log_det) > 1:
            print("ERROR: logdet mismatch")
            print("ours: ", log_det, " vs ", np.linalg.slogdet(L @ L.T)[1])
        #
        return trace - log_det  # + lambda_ * self.penalty(L)

    def projection(self, L):
        # function that allows control in one place on which projection is being executed for all class.
        return self._project(L)

    def _projectK(self, L):
        # an attempt to use derive projection to convex cone by derivation in class
        sigma, U = np.linalg.eig(L @ L.T)
        print(sigma)
        print(U)
        sigma = np.where(sigma < 0, 0, sigma)
        projected = U @ np.diag(sigma) @ U.T
        return np.linalg.cholesky(projected)

    def _penalty(self, L):
        n = L.shape[0]
        res = np.zeros(L.shape)
        for i in range(n - 1, self.k, -1):
            res += np.diag(np.diag(L, -i), -i)
        return np.linalg.norm(res) ** 2

    def _project(self, L):
        # this function takes a lot of time for big matrix
        lam_min = np.min(np.diag(L))  # maybe expensive but can save corrections on objective, grad and hessian.
        if lam_min <= 0:
            L = L + (np.eye(L.shape[0]) * lam_min)
        res = np.diag(np.diagonal(L, offset=0), 0)
        for i in range(1, self.k + 1):
            res += np.diag(np.diagonal(L, offset=i), -i)
        res[np.abs(res) <= 1e-4] = 0
        return res


    def gradient(self, L):
        L = self.projection(L)
        L_diag = stable_diagonal_vec(L)
        inverse_diag = np.eye(L.shape[0]) * (1 / L_diag)
        # Sanity check
        if len(inverse_diag.shape) != 2:
            print("Error, Gradiend suppose to be matrix")
        #
        grad = 2 * ((self.S @ L) - inverse_diag)
        return np.where(grad == 0, 0, grad / np.linalg.norm(grad))

    def hessian(self, L):
        L = self.projection(L)
        L_diag_sq = (np.diag(L) ** 2) + stable_vec(L)
        inverse_diag_sq = np.eye(L.shape[0]) * (1 / L_diag_sq)
        # Sanity check
        if len(inverse_diag_sq.shape) != 2:
            print("Error, Hessian suppose to be matrix")
        #
        hess = 2 * (self.S + inverse_diag_sq)
        return np.where(hess == 0, 0, hess / np.linalg.norm(hess))

    def optimality_condition(self, S, L):
        L = self.projection(L)
        L_diag = stable_diagonal_vec(L)
        inverse_diag = np.eye(L.shape[0]) * (1 / L_diag)
        return S - inverse_diag @ tri_inverse(L)

    def output(self, L):
        L = self.projection(L)
        return L @ L.T
