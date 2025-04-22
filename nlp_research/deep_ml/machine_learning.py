from collections import defaultdict
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


def confusion_matrix(data: list[list[int]]) -> list[list[int]]:
    """
    Create a confusion matrix from a list of lists.

    :param data: List of lists where each inner list contains the true and predicted labels.
                 Example: [[true_label1, pred_label1], [true_label2, pred_label2], ...]
    :return: Confusion matrix as a list of lists.
    """
    TP, FP, TN, FN = 0, 0, 0, 0
    for y_true, y_pred in data:
        if y_true == 0 and y_pred == 0:
            TN += 1
        elif y_true == 0 and y_pred == 1:
            FN += 1
        elif y_true == 1 and y_pred == 1:
            TP += 1
        else:
            FP += 1
    return [[TP, FP], [FN, TN]]


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
    return round(TP / (TP + FP), 3)


def recall(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    denominator = TP + FN
    if denominator == 0:
        return 0.0
    return round(TP / denominator, 3)


def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f_score_value = (
        (1 + beta**2)
        * (precision_value * recall_value)
        / (beta**2 * precision_value + recall_value)
    )
    return round(f_score_value, 3)


def l1_regularization_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
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


def adaboost_fit(X: np.ndarray, y: np.ndarray, n_clf: int) -> list:
    """
    Fit AdaBoost classifier on the training data.

    Parameters:
        X: Training data of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        n_clf: Number of weak classifiers to use

    Returns:
        List of tuples where each tuple contains:
        - feature_idx: Index of the feature used for the weak classifier
        - threshold: Threshold used for classification
        - polarity: Direction of the inequality sign (-1 or 1)
        - alpha: Weight assigned to this classifier
    """
    n_samples, n_features = X.shape

    # Initialize weights to 1/N
    w = np.full(n_samples, (1 / n_samples))

    # Initialize classifier list
    classifiers = []

    # Iterate through classifiers
    for _ in range(n_clf):
        # Initialize error and classifier parameters
        min_error = float('inf')
        best_feature_idx = 0
        best_threshold = 0
        best_polarity = 1

        # Loop through all features
        for feature_idx in range(n_features):
            # Get all values of the current feature
            feature_values = X[:, feature_idx]

            # Find unique values to test as thresholds
            thresholds = np.unique(feature_values)

            # Loop through all possible thresholds
            for threshold in thresholds:
                # Try both polarities (> and <)
                for polarity in [-1, 1]:
                    # Predict with current setup
                    predictions = np.ones(n_samples)

                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    # Calculate weighted error
                    misclassified = w[y != predictions]
                    error = np.sum(misclassified)

                    # If we have a better error, save the classifier
                    if error < min_error:
                        min_error = error
                        best_feature_idx = feature_idx
                        best_threshold = threshold
                        best_polarity = polarity

        # Calculate alpha (weight of this classifier)
        eps = 1e-10  # Small value to avoid division by zero
        alpha = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))

        # Make predictions using the best classifier found
        predictions = np.ones(n_samples)
        if best_polarity == 1:
            predictions[X[:, best_feature_idx] < best_threshold] = -1
        else:
            predictions[X[:, best_feature_idx] >= best_threshold] = -1

        # Update sample weights
        w *= np.exp(-alpha * y * predictions)
        w /= np.sum(w)  # Normalize weights

        # Save classifier
        classifiers.append((best_feature_idx, best_threshold, best_polarity, alpha))

    return classifiers


def adaboost_predict(X: np.ndarray, classifiers: list) -> np.ndarray:
    """
    Make predictions using fitted AdaBoost classifier.

    Parameters:
        X: Data to make predictions on, shape (n_samples, n_features)
        classifiers: List of tuples with classifier parameters from adaboost_fit

    Returns:
        Predicted class labels (-1 or 1)
    """
    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples)

    # Sum predictions from all classifiers, weighted by their alpha
    for clf in classifiers:
        feature_idx, threshold, polarity, alpha = clf

        # Make predictions using this classifier
        predictions = np.ones(n_samples)
        if polarity == 1:
            predictions[X[:, feature_idx] < threshold] = -1
        else:
            predictions[X[:, feature_idx] >= threshold] = -1

        # Weight predictions by alpha and add to final prediction
        y_pred += alpha * predictions

    # Return sign of prediction
    return np.sign(y_pred)


def gini_impurity(y):
    """
    Calculate Gini Impurity for a list of class labels.

    :param y: List of class labels
    :return: Gini Impurity rounded to three decimal places
    """
    N = len(y)
    d = defaultdict(int)
    for y_i in y:
        d[y_i] += 1

    val = 1 - sum((v / N) ** 2 for v in d.values())

    return round(val, 3)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted values.

    :param y_true: Numpy array of true values
    :param y_pred: Numpy array of predicted values

    :return: RMSE rounded to three decimal places
    """
    rmse_res = np.mean((y_true - y_pred) ** 2) ** (1 / 2)
    return round(rmse_res, 3)


def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Jaccard Index (Intersection over Union) between true and predicted values.

    :param y_true: Numpy array of true values
    :param y_pred: Numpy array of predicted values

    :return: Jaccard Index rounded to three decimal places
    """
    A = (y_true == 1).sum()
    B = (y_pred == 1).sum()
    inter = ((y_true == 1) & (y_pred == 1)).sum()
    denominator = A + B - inter
    if denominator == 0:
        return 0.0
    result = inter / denominator
    return round(result, 3)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Dice Score between true and predicted values.

    :param y_true: Numpy array of true values
    :param y_pred: Numpy array of predicted values

    :return: Dice Score rounded to three decimal places
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    divisor = 2 * tp + fp + fn
    if divisor == 0:
        return 0.0
    res = 2 * tp / divisor
    return round(res, 3)


def _generate_bipolar_hv(dim: int, seed: int) -> np.ndarray:
    """Generates a random bipolar hypervector using a specific seed."""
    rng = np.random.default_rng(seed)
    return rng.choice([1, -1], size=dim)


def create_row_hv(row: dict, dim: int, random_seeds: dict) -> np.ndarray:
    """
    Generates a composite hypervector for a dataset row using HDC.

    Each feature is represented by binding hypervectors for the feature name
    and its value. The value hypervector uses a seed derived from the feature's
    seed in random_seeds and the value itself. All feature hypervectors are
    then bundled.

    Args:
        row: Dictionary representing a dataset row {feature_name: value}.
        dim: Dimensionality of the hypervectors.
        random_seeds: Dictionary mapping feature names to integer seeds for value HVs.

    Returns:
        A numpy array representing the composite bipolar hypervector (+1/-1) for the row.
    """
    feature_hvs = []
    for feature_name, feature_value in row.items():
        if feature_name not in random_seeds:
            raise ValueError(f'Seed not found for feature: {feature_name} in random_seeds')

        name_seed = hash(feature_name)
        hv_name = _generate_bipolar_hv(dim, name_seed)
        value_hash = hash(feature_value)
        value_seed = random_seeds[feature_name] + value_hash
        hv_value = _generate_bipolar_hv(dim, value_seed)
        bound_hv = hv_name * hv_value
        feature_hvs.append(bound_hv)

    composite_hv = np.zeros(dim) if not feature_hvs else np.sum(feature_hvs, axis=0)
    final_hv = np.where(composite_hv >= 0, 1, -1).astype(int)

    return final_hv


def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors.

    :param v1: Numpy array of vector 1
    :param v2: Numpy array of vector 2

    :return: Cosine similarity rounded to three decimal places
    """
    simm = np.sum(v1 * v2) / (np.sum(v1**2) ** (1 / 2) * np.sum(v2**2) ** (1 / 2))
    return round(simm, 3)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination) between true and predicted values.

    :param y_true: Numpy array of true values
    :param y_pred: Numpy array of predicted values

    :return: R-squared rounded to three decimal places
    """
    ssr = np.sum((y_true - y_pred) ** 2)
    mean = np.mean(y_true)
    sst = np.sum((y_true - mean) ** 2)
    return round(1 - (ssr / sst), 3)
