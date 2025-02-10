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


def linear_regression_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    theta = np.round(np.linalg.inv(X.T @ X) @ X.T @ y, 4)
    return theta


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    theta = np.zeros(X.shape[1])
    m = len(y)
    for _ in range(iterations):
        theta = theta - alpha / m * (X.T @ (X @ theta - y))
    return theta.round(4)


def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)
    standardized_data = np.round((data - mean) / std, 4, 4)
    normalized_data = np.round(((data - min_) / (max_ - min_)), 4)
    return standardized_data, normalized_data


def euclidean_distance(point_1, point_2):
    dist = sum((p_2 - p_1) ** 2 for p_1, p_2 in zip(point_1, point_2, strict=False)) ** (1 / 2)
    return dist


def k_means_clustering(
    points: list[tuple[float, float]],
    initial_centroids: list[tuple[float, float]],
    max_iterations: int,
) -> list[tuple[float, float]]:
    points_ = np.array(points)
    centroids = np.array(initial_centroids)

    for _ in range(max_iterations):
        distances = np.array(
            [[euclidean_distance(point, centroid) for centroid in centroids] for point in points]
        )
        cluster_assignment = np.argmin(distances, axis=1)
        new_centroids = np.array(
            [points_[cluster_assignment == i].mean(axis=0) for i in range(len(centroids))]
        )

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids.tolist()
