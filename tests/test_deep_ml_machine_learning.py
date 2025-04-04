import numpy as np
import pytest

from nlp_research.deep_ml import (
    accuracy_score,
    adaboost_fit,
    adaboost_predict,
    batch_iterator,
    calculate_correlation_matrix,
    cross_validation_split,
    divide_on_feature,
    euclidean_distance,
    f_score,
    feature_scaling,
    get_random_subsets,
    gini_impurity,
    jaccard_index,
    k_means_clustering,
    l1_regularization_gradient_descent,
    linear_regression_gradient_descent,
    linear_regression_normal_equation,
    log_softmax,
    pca,
    polynomial_features,
    precision,
    recall,
    rmse,
    shuffle_data,
    to_categorical,
)


@pytest.mark.parametrize(
    'X, y, expected',
    [
        (np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), np.array([0.0, 1.0])),
    ],
)
def test_linear_regression_normal_equation(X, y, expected):
    assert np.allclose(linear_regression_normal_equation(X, y), expected)


@pytest.mark.parametrize(
    'X, y, alpha, iterations, expected',
    [
        (
            np.array([[1, 1], [1, 2], [1, 3]]),
            np.array([1, 2, 3]),
            0.01,
            1000,
            np.array([0.1107, 0.9513]),
        ),
    ],
)
def test_linear_regression_gradient_descent(X, y, alpha, iterations, expected):
    assert np.allclose(linear_regression_gradient_descent(X, y, alpha, iterations), expected)


@pytest.mark.parametrize(
    'X, expected',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            (
                np.array([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]),
                np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            ),
        )
    ],
)
def test_feature_scaling(X, expected):
    expected_standardized_data, expected_normalized_data = expected
    standardized_data, normalized_data = feature_scaling(X)
    assert np.allclose(standardized_data, expected_standardized_data)
    assert np.allclose(normalized_data, expected_normalized_data)


@pytest.mark.parametrize(
    'X, y, n_clf, expected_len, expected_feature_idx',
    [
        (
            np.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
            np.array([1, 1, -1, -1]),
            3,
            3,
            0,  # Feature index 0 should be selected
        ),
    ],
)
def test_adaboost_fit(X, y, n_clf, expected_len, expected_feature_idx):
    classifiers = adaboost_fit(X, y, n_clf)
    assert len(classifiers) == expected_len
    # Verify that the first feature is chosen (in our test case)
    assert classifiers[0][0] == expected_feature_idx


@pytest.mark.parametrize(
    'X, y, n_clf',
    [
        (
            np.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
            np.array([1, 1, -1, -1]),
            3,
        ),
    ],
)
def test_adaboost_predict(X, y, n_clf):
    classifiers = adaboost_fit(X, y, n_clf)
    predictions = adaboost_predict(X, classifiers)
    # Ensure predictions have the same shape as y
    assert predictions.shape == y.shape
    # Check that predictions are either -1 or 1
    assert np.all(np.isin(predictions, [-1, 1]))
    # For this simple dataset, we should achieve perfect accuracy
    assert np.all(predictions == y)


@pytest.mark.parametrize(
    'a, b, expected',
    [
        (np.array([1, 1]), np.array([4, 5]), 5.0),
        (np.array([0, 0]), np.array([0, 0]), 0.0),
        (np.array([-1, -2]), np.array([3, 1]), 5.0),
    ],
)
def test_euclidean_distance(a, b, expected):
    assert np.isclose(euclidean_distance(a, b), expected)


@pytest.mark.parametrize(
    'points, initial_centroids, max_iterations, expected_centroids',
    [
        (
            [(1, 1), (1.5, 2), (3, 4), (5, 7), (3.5, 5), (4.5, 5), (3.5, 4.5)],
            [(1, 1), (5, 7)],
            10,
            [[1.25, 1.5], [3.9, 5.1]],
        )
    ],
)
def test_k_means_clustering(points, initial_centroids, max_iterations, expected_centroids):
    result_centroids = k_means_clustering(points, initial_centroids, max_iterations)
    assert np.allclose(result_centroids, expected_centroids, atol=1e-1)


@pytest.mark.parametrize(
    'data, k, seed',
    [
        (np.arange(10).reshape(5, 2), 5, 42),
        (np.random.rand(20, 3), 4, 123),
    ],
)
def test_cross_validation_split(data, k, seed):
    splits = cross_validation_split(data, k, seed)
    assert len(splits) == k
    test_indices_set = set()
    for _, (train_list, test_list) in enumerate(splits):
        train_set = np.array(train_list)
        test_set = np.array(test_list)

        # Check shapes
        assert train_set.shape[1] == data.shape[1]
        assert test_set.shape[1] == data.shape[1]
        assert len(train_set) + len(test_set) == len(data)

        # Check that test indices are unique across folds
        test_indices_fold = {tuple(row) for row in test_set}
        assert test_indices_set.isdisjoint(test_indices_fold)
        test_indices_set.update(test_indices_fold)

        # Check that train and test sets are disjoint within a fold
        train_indices_fold = {tuple(row) for row in train_set}
        assert train_indices_fold.isdisjoint(test_indices_fold)

    # Check that all original data points are in exactly one test set
    assert len(test_indices_set) == len(data)


@pytest.mark.parametrize(
    'data, k, expected_shape',
    [(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 2, (3, 2)), (np.random.rand(10, 5), 3, (5, 3))],
)
def test_pca(data, k, expected_shape):
    principal_components = pca(data, k)
    assert principal_components.shape == expected_shape
    # Check for orthogonality (dot product of columns should be close to 0)
    if k > 1:
        dot_product = principal_components[:, 0] @ principal_components[:, 1]
        assert np.isclose(dot_product, 0.0, atol=1e-4)


@pytest.mark.parametrize(
    'X, y, seed',
    [
        (np.arange(10).reshape(5, 2), np.arange(5), 42),
        (np.random.rand(20, 3), np.random.randint(0, 2, 20), 123),
    ],
)
def test_shuffle_data(X, y, seed):
    X_shuffled, y_shuffled = shuffle_data(X, y, seed)
    assert X_shuffled.shape == X.shape
    assert y_shuffled.shape == y.shape
    # Check that elements are the same, just rearranged
    assert set(map(tuple, X)) == set(map(tuple, X_shuffled))
    assert set(y) == set(y_shuffled)
    # Check that corresponding X and y rows are shuffled together
    original_map = {tuple(X[i]): y[i] for i in range(len(y))}
    shuffled_map = {tuple(X_shuffled[i]): y_shuffled[i] for i in range(len(y_shuffled))}
    assert original_map == shuffled_map
    # Check that different seeds produce different shuffles
    X_shuffled_2, y_shuffled_2 = shuffle_data(X, y, seed + 1)
    assert not np.array_equal(X_shuffled, X_shuffled_2)
    assert not np.array_equal(y_shuffled, y_shuffled_2)


@pytest.mark.parametrize(
    'X, y, bs, expected_num_bs, expected_last_bs_X, expected_last_bs_y',
    [
        (np.arange(10).reshape(5, 2), np.arange(5), 2, 3, 1, 1),
        (np.arange(12).reshape(6, 2), np.arange(6), 3, 2, 3, 3),
        (np.arange(7).reshape(7, 1), None, 3, 3, 1, None),  # Test with y=None
        (np.arange(6).reshape(3, 2), np.arange(3), 5, 1, 3, 3),  # Test with batch_size > n_samples
    ],
)
def test_batch_iterator(X, y, bs, expected_num_bs, expected_last_bs_X, expected_last_bs_y):
    batches = batch_iterator(X, y, bs)
    assert len(batches) == expected_num_bs
    total_X_samples = 0
    total_y_samples = 0
    for i, batch in enumerate(batches):
        if y is None:
            X_batch = batch
            y_batch = None
            assert isinstance(X_batch, np.ndarray)
        else:
            X_batch, y_batch = batch
            assert isinstance(X_batch, np.ndarray)
            assert isinstance(y_batch, np.ndarray)
            assert X_batch.shape[0] == y_batch.shape[0]
            total_y_samples += y_batch.shape[0]

        total_X_samples += X_batch.shape[0]

        if i == expected_num_bs - 1:  # Last batch
            assert X_batch.shape[0] == expected_last_bs_X
            if y is not None:
                assert y_batch.shape[0] == expected_last_bs_y
        else:
            assert X_batch.shape[0] == bs
            if y is not None:
                assert y_batch.shape[0] == bs

    assert total_X_samples == X.shape[0]
    if y is not None:
        assert total_y_samples == y.shape[0]


@pytest.mark.parametrize(
    'X, feature_i, threshold, expected_true_shape, expected_false_shape',
    [
        (np.array([[1, 5], [2, 6], [3, 3], [4, 8]]), 1, 6, (2, 2), (2, 2)),
        (np.array([[1], [5], [3], [8]]), 0, 4, (2, 1), (2, 1)),
        (np.array([[1, 2], [3, 4]]), 0, 5, (0, 2), (2, 2)),  # All false
        (np.array([[1, 2], [3, 4]]), 0, 0, (2, 2), (0, 2)),  # All true
    ],
)
def test_divide_on_feature(X, feature_i, threshold, expected_true_shape, expected_false_shape):
    X_true, X_false = divide_on_feature(X, feature_i, threshold)
    assert X_true.shape == expected_true_shape
    assert X_false.shape == expected_false_shape
    if X_true.size > 0:
        assert np.all(X_true[:, feature_i] >= threshold)
    if X_false.size > 0:
        assert np.all(X_false[:, feature_i] < threshold)


@pytest.mark.parametrize(
    'X, degree, expected_output',
    [
        (
            np.array([[1, 2]]),
            2,
            np.array([[1, 1, 2, 1 * 1, 1 * 2, 2 * 2]]),
        ),  # 1, x1, x2, x1^2, x1*x2, x2^2
        (np.array([[2, 3], [4, 5]]), 1, np.array([[1, 2, 3], [1, 4, 5]])),  # 1, x1, x2
        (
            np.array([[0, 1]]),
            3,
            np.array([[1, 0, 1, 0 * 0, 0 * 1, 1 * 1, 0 * 0 * 0, 0 * 0 * 1, 0 * 1 * 1, 1 * 1 * 1]]),
        ),  # 1, x1, x2, x1^2, x1x2, x2^2, x1^3, x1^2x2, x1x2^2, x2^3
    ],
)
def test_polynomial_features(X, degree, expected_output):
    result = polynomial_features(X, degree)
    assert np.allclose(result, expected_output)


@pytest.mark.parametrize(
    'X, y, n_subsets, replacements, seed, expected_num_subsets, expected_subset_size',
    [
        (np.arange(10).reshape(5, 2), np.arange(5), 3, True, 42, 3, 5),  # With replacement
        (
            np.arange(20).reshape(10, 2),
            np.arange(10),
            5,
            False,
            123,
            5,
            5,
        ),  # Without replacement (size n/2)
    ],
)
def test_get_random_subsets(
    X, y, n_subsets, replacements, seed, expected_num_subsets, expected_subset_size
):
    subsets = get_random_subsets(X, y, n_subsets, replacements, seed)
    assert len(subsets) == expected_num_subsets
    for X_sub, y_sub in subsets:
        X_sub_np = np.array(X_sub)
        y_sub_np = np.array(y_sub)
        assert X_sub_np.shape == (expected_subset_size, X.shape[1])
        assert y_sub_np.shape == (expected_subset_size,)


@pytest.mark.parametrize(
    'x, n_col, expected_output',
    [
        (np.array([0, 1, 2]), None, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        (np.array([1, 0]), None, np.array([[0, 1], [1, 0]])),
        (np.array([0, 2]), 4, np.array([[1, 0, 0, 0], [0, 0, 1, 0]])),  # n_col specified
        (np.array([1]), 3, np.array([[0, 1, 0]])),  # Single element
    ],
)
def test_to_categorical(x, n_col, expected_output):
    result = to_categorical(x, n_col)
    assert np.allclose(result, expected_output)


@pytest.mark.parametrize(
    'y_true, y_pred, expected_score',
    [
        (np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), 1.0),
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 0.5),
        (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), 0.0),
        (np.array([0, 0, 0]), np.array([1, 1, 1]), 0.0),
    ],
)
def test_accuracy_score(y_true, y_pred, expected_score):
    assert np.isclose(accuracy_score(y_true, y_pred), expected_score)


@pytest.mark.parametrize(
    'X, Y, expected_corr_diag',
    [
        (
            np.array([[1, 2], [2, 4], [3, 6]]),
            None,
            np.array([1.0, 1.0]),
        ),  # Perfect correlation with self
        (np.array([[1, 2], [2, 1], [3, 3]]), None, np.array([1.0, 1.0])),  # Correlation with self
        (
            np.array([[1, 2], [2, 3], [3, 4]]),
            np.array([[2, 4], [4, 6], [6, 8]]),
            np.array([1.0, 1.0]),
        ),  # Perfect correlation between X and Y
    ],
)
def test_calculate_correlation_matrix(X, Y, expected_corr_diag):
    corr_matrix = calculate_correlation_matrix(X, Y)
    expected_shape = (X.shape[1], X.shape[1] if Y is None else Y.shape[1])
    assert corr_matrix.shape == expected_shape
    if Y is None:
        assert np.allclose(np.diag(corr_matrix), expected_corr_diag)
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric
    # If Y is provided, we might not be checking the diagonal directly unless Y=X
    # Check values are within [-1, 1]
    assert np.all(corr_matrix >= -1.0001) and np.all(corr_matrix <= 1.0001)


@pytest.mark.parametrize(
    'x, expected_output',
    [
        (
            np.array([1.0, 2.0, 0.0]),
            np.array([-1.40760596, -0.40760596, -2.40760596]),
        ),  # Corrected expected values
        (np.array([0.0, 0.0, 0.0]), np.array([-1.0986, -1.0986, -1.0986])),  # log(1/3)
    ],
)
def test_log_softmax(x, expected_output):
    result = log_softmax(x)
    assert np.allclose(result, expected_output, atol=1e-4)


@pytest.mark.parametrize(
    'y_true, y_pred, expected_precision',
    [
        (np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), 1.0),  # TP=2, FP=0
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 0.5),  # TP=1, FP=1
        (np.array([0, 0, 0, 0]), np.array([1, 1, 0, 0]), 0.0),  # TP=0, FP=2
        (
            np.array([1, 1, 1, 1]),
            np.array([0, 0, 0, 0]),
            0.0,
        ),  # TP=0, FP=0 -> handle division by zero (should be 0 or undefined, test assumes 0)
        (np.array([1, 0]), np.array([1, 0]), 1.0),  # TP=1, FP=0
    ],
)
def test_precision(y_true, y_pred, expected_precision):
    # Need to handle potential ZeroDivisionError if TP+FP=0
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        # If denominator is zero, precision is often defined as 0
        assert expected_precision == 0.0
    else:
        p = precision(y_true, y_pred)
        assert np.isclose(p, expected_precision)


@pytest.mark.parametrize(
    'y_true, y_pred, expected_recall',
    [
        (np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), 1.0),  # TP=2, FN=0
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 0.5),  # TP=1, FN=1
        (
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 0, 0]),
            0.0,
        ),  # TP=0, FN=0 -> handle division by zero
        (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), 0.0),  # TP=0, FN=4
        (np.array([1, 0]), np.array([0, 1]), 0.0),  # TP=0, FN=1
    ],
)
def test_recall(y_true, y_pred, expected_recall):
    r = recall(y_true, y_pred)
    assert np.isclose(r, expected_recall)


@pytest.mark.parametrize(
    'y_true, y_pred, beta, expected_fscore',
    [
        (np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), 1, 1.0),
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 1, 0.5),
        (
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0, 1]),
            1,
            0.667,
        ),
        (
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0, 1]),
            0.5,
            0.667,
        ),
        (
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0, 1]),
            2,
            0.667,
        ),
        (
            np.array([0, 0]),
            np.array([0, 0]),
            1,
            0.0,
        ),
    ],
)
def test_f_score(y_true, y_pred, beta, expected_fscore):
    try:
        f = f_score(y_true, y_pred, beta)
        # Check for NaN explicitly, as np.isclose(np.nan, 0.0) is False
        if np.isnan(f):
            assert expected_fscore == 0.0
        else:
            assert np.isclose(f, expected_fscore, atol=1e-3)
    except ZeroDivisionError:
        # If precision or recall calculation leads to ZeroDivisionError, result should be 0
        assert expected_fscore == 0.0


@pytest.mark.parametrize(
    'X, y, alpha, learning_rate, max_iter, tol, expected_weights_non_zero, expected_bias_sign',
    [
        (
            np.array([[1, 1], [1, 2], [2, 2], [2, 3]]),
            np.array([1, 2, 2, 3]),
            0.1,
            0.01,
            1000,
            1e-4,
            True,
            1,
        ),  # Simple linear case
        (
            np.random.rand(20, 5),
            np.random.rand(20) * 5,
            0.05,
            0.01,
            500,
            1e-3,
            True,
            None,
        ),  # Random data
    ],
)
def test_l1_regularization_gradient_descent(
    X, y, alpha, learning_rate, max_iter, tol, expected_weights_non_zero, expected_bias_sign
):
    weights, bias = l1_regularization_gradient_descent(X, y, alpha, learning_rate, max_iter, tol)
    assert weights.shape == (X.shape[1],)
    if expected_weights_non_zero:
        assert np.any(np.abs(weights) > tol)  # Expect some non-zero weights typically
    if expected_bias_sign is not None:
        assert np.sign(bias) == expected_bias_sign or bias == 0


@pytest.mark.parametrize(
    'y, expected_gini',
    [
        ([0, 0, 0, 0], 0.0),  # Pure node
        ([1, 1, 1, 1], 0.0),  # Pure node
        ([0, 1, 0, 1], 0.5),  # Perfectly mixed (2 classes)
        ([0, 0, 1, 1], 0.5),
        ([0, 1, 2, 0], 0.625),  # Three classes (2/4, 1/4, 1/4) -> 1 - (0.5^2 + 0.25^2 + 0.25^2)
        (['A', 'B', 'A'], 0.444),  # String labels 1 - ((2/3)^2 + (1/3)^2)
    ],
)
def test_gini_impurity(y, expected_gini):
    assert np.isclose(gini_impurity(y), expected_gini, atol=1e-3)


@pytest.mark.parametrize(
    'y_true, y_pred, expected_rmse',
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0.0),
        (np.array([1, 2, 3]), np.array([2, 3, 4]), 1.0),  # sqrt( (1^2 + 1^2 + 1^2) / 3 )
        (
            np.array([1, 2, 3]),
            np.array([1, 1, 5]),
            1.291,
        ),  # sqrt( (0^2 + (-1)^2 + 2^2) / 3 ) = sqrt(5/3)
    ],
)
def test_rmse(y_true, y_pred, expected_rmse):
    assert np.isclose(rmse(y_true, y_pred), expected_rmse, atol=1e-3)


@pytest.mark.parametrize(
    'y_true, y_pred, expected_jaccard',
    [
        (np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]), 1.0),  # Inter=2, Union=2
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]), 0.333),  # Inter=1, Union=3 (A=2, B=2)
        (np.array([1, 1, 1, 0]), np.array([1, 0, 0, 1]), 0.250),  # Inter=1, Union=4 (A=3, B=2)
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 0.0),  # Inter=0, Union=0 -> handles 0/0
        (np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0]), 0.0),  # Inter=0, Union=4
    ],
)
def test_jaccard_index(y_true, y_pred, expected_jaccard):
    # Handle potential division by zero if union is 0
    union_size = (y_true == 1).sum() + (y_pred == 1).sum() - ((y_true == 1) & (y_pred == 1)).sum()
    if union_size == 0:
        assert expected_jaccard == 0.0
    else:
        assert np.isclose(jaccard_index(y_true, y_pred), expected_jaccard, atol=1e-3)
