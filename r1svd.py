import numpy as np
import pandas as pd
from numpy.linalg import norm

import matplotlib.pyplot as plt


def reorder(M: np.ndarray, idx_r: np.ndarray, idx_c: np.ndarray):
    return M[np.ix_(idx_r, idx_c)]


def reorder_df(df: pd.DataFrame, idx_r: np.ndarray, idx_c: np.ndarray):
    tmp = df.copy()
    tmp = tmp.iloc[idx_r, :]
    tmp = tmp.iloc[:, idx_c]
    return tmp


def _rand_unit(n: int):
    x = np.random.rand(n)
    x = np.abs(x)
    x = x / norm(x)
    return x


def _init_diagonal_matrices(A):
    Dr = A.sum(axis=1)
    Dc = A.sum(axis=0)

    Dr[Dr == 0] = 1
    Dc[Dc == 0] = 1

    Dr1_A = np.diag(1/Dr) @ A
    Dc1_A = np.diag(1/Dc) @ A.T

    return Dr1_A, Dc1_A


class RankOneSvd:
    """ """

    def __init__(self, threshold=1E-4, max_iter=1000):
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, A: np.ndarray):
        self.A_ = A
        Dr1_A, Dc1_A = _init_diagonal_matrices(A)

        n, m = self.A_.shape

        u = _rand_unit(n)
        v = _rand_unit(m)

        u_prev = u
        v_prev = v

        gamma_prev = -np.inf

        it = 0

        while it < self.max_iter:
            v = Dc1_A @ u_prev
            v = v/norm(v)

            u = Dr1_A @ v_prev
            u = u/norm(u)

            gamma = norm(u - u_prev) + norm(v - v_prev)

            if abs(gamma - gamma_prev) <= self.threshold:
                break

            gamma_prev = gamma
            u_prev = u
            v_prev = v

            it += 1

        # if it == self.max_iter:
        #     print(f'did not converge in {self.max_iter} iterations')
        # print(f'Converged in {it} iterations.')
        # print('u = {u}\n v = {v}')
        self.u_ = u
        self.v_ = v
        self.u_order_ = np.argsort(u)
        self.v_order_ = np.argsort(v)
        return self

    def fit_transform(self, A: np.ndarray):
        self.fit(A)
        return A[np.ix_(self.u_order_, self.v_order_)]

    def get_row_labels(self):
        du = np.concatenate([[0], np.diff(self.u_[self.u_order_])])
        m = np.mean(du)
        return np.cumsum(du > m)

    def get_col_labels(self):
        dv = np.concatenate([[0], np.diff(self.v_[self.v_order_])])
        m = np.mean(dv)
        return np.cumsum(dv > m)

    def get_co_clusters(self):
        r_idx = self.get_row_labels()
        c_idx = self.get_col_labels()
        A_sorted = self.A_[np.ix_(self.u_order_, self.v_order_)]

        nr = r_idx[-1]+1
        nc = c_idx[-1]+1
        M = np.zeros((nr, nc))
        for row in range(nr):
            for col in range(nc):
                M[row, col] = A_sorted[
                    np.ix_(r_idx == row, c_idx == col)].sum()
        reg_mat = M > M.mean()
        regions = [*zip(*np.where(reg_mat))]
        return regions, reg_mat

    def plot_matrix(self, matrix='A', ordered=False):
        if ordered:
            if matrix == 'A':
                plt.spy(self.A_[np.ix_(self.u_order_, self.v_order_)])
                return
            if matrix == 'Sr':
                Sr = self.A_ @ self.A_.T
                plt.spy(Sr[np.ix_(self.u_order_, self.u_order_)])
                return
            if matrix == 'Sc':
                Sc = self.A_.T @ self.A_
                plt.spy(Sc[np.ix_(self.v_order_, self.v_order_)])
                return
        else:
            if matrix == 'A':
                plt.spy(self.A_)
                return
            if matrix == 'Sr':
                plt.spy(self.A_ @ self.A_.T)
                return
            if matrix == 'Sc':
                plt.spy(self.A_.T @ self.A_)
                return

    def plot_u(self, transpose=False):
        if transpose:
            plt.plot(np.sort(self.u_), np.arange(self.u_.shape[0]))
        else:
            plt.plot(np.sort(self.u_))

    def plot_v(self):
        plt.plot(np.sort(self.v_))
