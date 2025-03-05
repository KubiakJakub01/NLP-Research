from itertools import combinations_with_replacement

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
    standardized_data = np.round((data - mean) / std, 4)
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
            [[euclidean_distance(point, centroid) for centroid in centroids] for point in points_]
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


def _chunk(x: np.ndarray, batch_size: int = 1):
    for i in range(0, x.shape[0], batch_size):
        yield x[i : i + batch_size]


def batch_iterator(X, y=None, batch_size=64):
    if y is None:
        return list(_chunk(X, batch_size))
    return list(zip(_chunk(X, batch_size), _chunk(y, batch_size), strict=False))


def divide_on_feature(X, feature_i, threshold):
    return (X[X[:, feature_i] >= threshold], X[X[:, feature_i] < threshold])


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    feats = []
    bias = [1]
    for X_i in X:
        out = bias + [
            np.prod(x) for i in range(1, degree + 1) for x in combinations_with_replacement(X_i, i)
        ]
        feats.append(out)
    return np.array(feats)


def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)

    n, _ = X.shape
    subset_size = n if replacements else n // 2
    idx = np.array(
        [np.random.choice(n, subset_size, replace=replacements) for _ in range(n_subsets)]
    )
    return [(X[idx][i].tolist(), y[idx][i].tolist()) for i in range(n_subsets)]


def to_categorical(x: np.ndarray, n_col: int | None = None) -> np.ndarray:
    n_row = x.max() + 1
    n_col = n_row if n_col is None else n_col
    diagonal = np.eye(n_row, n_col)
    return diagonal[x]


def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).sum() / y_true.shape[0]


def calculate_correlation_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the correlation matrix for dataset X (with optional dataset Y).

    Parameters:
        X: 2D numpy array of shape (n_samples, n_features)
        Y: Optional 2D numpy array of shape (n_samples, n_features)

    Returns:
        correlation_matrix: 2D numpy array of shape (X_features, Y_features)
    """
    X = X.T
    Y = Y.T if Y is not None else X

    mean_X = np.mean(X, axis=1, keepdims=True)
    std_X = np.std(X, axis=1, keepdims=True, ddof=0)

    mean_Y = np.mean(Y, axis=1, keepdims=True)
    std_Y = np.std(Y, axis=1, keepdims=True, ddof=0)

    norm_X = (X - mean_X) / std_X
    norm_Y = (Y - mean_Y) / std_Y

    correlation_matrix = (norm_X @ norm_Y.T) / X.shape[1]

    return correlation_matrix


def log_softmax(x: np.ndarray) -> np.ndarray:
    return x - x.max() - np.log(np.sum(np.exp(x - x.max())))


def precision(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    return TP / (TP + FP)


def l1_regularization_gradient_descent(
    X: np.array,
    y: np.array,
    alpha: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        y_pred = X @ weights + bias
        grad_weights = X.T @ (y_pred - y) / n_samples + alpha * np.sign(weights)
        grad_bias = np.sum(y_pred - y) / n_samples
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
        if np.sum(abs(grad_weights)) < tol:
            break
    return weights, bias
