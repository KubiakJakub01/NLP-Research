import numpy as np


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


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.round(np.sqrt(np.sum((a - b) ** 2)), 4)


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


def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list:
    np.random.seed(seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    splits = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        train_set = data[train_idx]
        test_set = data[test_idx]

        splits.append([train_set.tolist(), test_set.tolist()])

    return splits


def pca(data: np.ndarray, k: int) -> np.ndarray:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized = (data - mean) / std
    cov = np.cov(normalized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = sorted_eigenvectors[:, :k]
    return np.round(principal_components, 4)


def shuffle_data(
    X: np.ndarray, y: np.ndarray, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def chunk(x: np.ndarray, batch_size: int = 1):
    for i in range(0, x.shape[0], batch_size):
        yield x[i : i + batch_size]


def batch_iterator(X, y=None, batch_size=64):
    if y is None:
        return list(chunk(X, batch_size))
    return list(zip(chunk(X, batch_size), chunk(y, batch_size), strict=False))


def divide_on_feature(X, feature_i, threshold):
    return (X[X[:, feature_i] >= threshold], X[X[:, feature_i] < threshold])
