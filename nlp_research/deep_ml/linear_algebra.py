import numpy as np


def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float] | int:
    c = []
    for a_i in a:
        if len(a_i) != len(b):
            return -1
        c.append(sum(a_j * b_j for (a_j, b_j) in zip(a_i, b, strict=False)))
    return c


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    def _cov(vectors, X_n, i, j):
        c = 0
        for v1, v2 in zip(vectors[i], vectors[j], strict=False):
            c += (v1 - X_n[i]) * (v2 - X_n[j])
        return c / (len(vectors[i]) - 1)

    X_n = [sum(vector) / len(vector) for vector in vectors]
    cov_matrix = []
    for i in range(len(vectors)):
        cov_i = []
        for j in range(len(X_n)):
            cov_i.append(_cov(vectors, X_n, i, j))
        cov_matrix.append(cov_i)

    return cov_matrix


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros_like(b, dtype=float)
    x_hold = np.zeros_like(b, dtype=float)

    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1 / d_a[i]) * (b[i] - sum(nda[i] * x))
        x = x_hold.copy()
    return x.round(4).tolist()
