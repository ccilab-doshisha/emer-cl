import typing as t

import numpy as np
from nptyping import NDArray


class LinearCCA:
    def __init__(self, dim: int):
        self.dim = dim
        self.w = np.zeros([2, dim, dim])
        self.m = np.zeros([2, dim])

    def fit(self, f1_hat: NDArray, f2_hat: NDArray) -> None:
        r1 = 1e-4
        r2 = 1e-4

        m = f1_hat.shape[0]
        o1 = f1_hat.shape[1]
        o2 = f2_hat.shape[1]

        self.m[0] = np.mean(f1_hat, axis=0)
        self.m[1] = np.mean(f2_hat, axis=0)
        f1_bar = f1_hat - np.tile(self.m[0], (m, 1))
        f2_bar = f2_hat - np.tile(self.m[1], (m, 1))

        sigma_hat12 = (1.0 / (m - 1)) * np.dot(f1_bar.T, f2_bar)
        sigma_hat11 = (1.0 / (m - 1)) * np.dot(
            f1_bar.T, f1_bar
        ) + r1 * np.identity(o1)
        sigma_hat22 = (1.0 / (m - 1)) * np.dot(
            f2_bar.T, f2_bar
        ) + r2 * np.identity(o2)

        [D1, V1] = np.linalg.eigh(sigma_hat11)
        [D2, V2] = np.linalg.eigh(sigma_hat22)
        sigma_hat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        sigma_hat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(
            np.dot(sigma_hat11RootInv, sigma_hat12), sigma_hat22RootInv
        )

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        self.w[0] = np.dot(sigma_hat11RootInv, U[:, 0 : self.dim])
        self.w[1] = np.dot(sigma_hat22RootInv, V[:, 0 : self.dim])

    def _get_result(self, x: NDArray, idx: int) -> NDArray:
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w[idx])
        return result

    def test(self, f1_hat: NDArray, f2_hat: NDArray) -> t.List[NDArray]:
        return [self._get_result(f1_hat, 0), self._get_result(f2_hat, 1)]
