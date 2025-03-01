import numpy as np
import pytest

from nlp_research.deep_ml import (
    calculate_covariance_matrix,
    cross_validation_split,
    euclidean_distance,
    matrix_dot_vector,
    solve_jacobi,
    svd_2x2_singular_values,
)


@pytest.mark.parametrize(
    'a, b, expected',
    [
        ([[1, 2, 3], [4, 5, 6]], [1, 2, 3], [14, 32]),
        ([[1, 2, 3], [4, 5, 6]], [1, 2], -1),
    ],
)
def test_matrix_dot_vector(a, b, expected):
    assert matrix_dot_vector(a, b) == expected


@pytest.mark.parametrize(
    'vectors, expected',
    [
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_calculate_covariance_matrix(vectors, expected):
    assert np.allclose(calculate_covariance_matrix(vectors), expected)


@pytest.mark.parametrize(
    'A, b, n, expected',
    [
        (np.array([[4, 1], [1, 3]]), np.array([1, 2]), 10, [0.0909, 0.6364]),
    ],
)
def test_solve_jacobi(A, b, n, expected):
    assert solve_jacobi(A, b, n) == expected


@pytest.mark.parametrize(
    'A',
    [
        (np.array([[2, 1], [1, 2]])),
        (np.array([[1, 2], [2, 1]])),
        (np.array([[1, 0], [0, 1]])),
    ],
)
def test_svd_2x2_singular_values(A):
    _, s_expected, _ = np.linalg.svd(A)
    _, s, _ = svd_2x2_singular_values(A)
    assert np.allclose(s, s_expected)


@pytest.mark.parametrize(
    'a, b, expected',
    [
        ([0, 0], [3, 4], 5.0),
        ([0, 0], [1, 1], 1.4142),
    ],
)
def test_euclidean_distance(a, b, expected):
    assert euclidean_distance(np.array(a), np.array(b)) == expected


@pytest.mark.parametrize(
    'dataset, n_folds, expected',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            3,
            [
                [
                    np.array([[5, 6], [1, 2], [7, 8]]),
                    np.array([[3, 4], [9, 10]]),
                ],
                [
                    np.array([[3, 4], [9, 10], [7, 8]]),
                    np.array([[5, 6], [1, 2]]),
                ],
                [
                    np.array([[3, 4], [9, 10], [5, 6], [1, 2]]),
                    np.array([[7, 8]]),
                ],
            ],
        )
    ],
)
def test_cross_validation_split(dataset, n_folds, expected):
    splits = cross_validation_split(dataset, n_folds)
    for split, expected_split in zip(splits, expected, strict=False):
        assert np.allclose(split[0], expected_split[0])
        assert np.allclose(split[1], expected_split[1])
