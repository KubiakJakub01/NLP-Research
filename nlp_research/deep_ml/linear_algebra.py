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


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    a = A

    a_t = np.transpose(a)
    a_2 = a_t @ a

    v = np.eye(2)

    for _ in range(1):
        if a_2[0, 0] == a_2[1, 1]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2 * a_2[0, 1], a_2[0, 0] - a_2[1, 1])

        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        d = np.transpose(r) @ a_2 @ r
        a_2 = d
        v = v @ r

    s = np.sqrt([d[0, 0], d[1, 1]])
    s_inv = np.array([[1 / s[0], 0], [0, 1 / s[1]]])

    u = a @ v @ s_inv

    return (u, s, v.T)


def transform_basis(B: np.ndarray, C: np.ndarray) -> np.ndarray:
    C_inv = np.linalg.inv(C)
    P = C_inv @ B
    P = np.round(P, 4)
    return P


def make_diagonal(x: np.ndarray) -> np.ndarray:
    return x * np.eye(x.shape[0])


def rref(matrix):
    row_n = matrix.shape[0]
    for i in range(row_n):
        row = matrix[i]
        pivot = row[i]
        if pivot != 0:
            matrix[i] = row / pivot
        for j in range(row_n):
            if i == j:
                continue
            matrix[j] -= matrix[j, i] * row
    return matrix
