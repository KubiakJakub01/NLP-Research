import pytest

from nlp_research.deep_ml import osa


@pytest.mark.parametrize(
    'source, target, expected',
    [
        ('kitten', 'sitting', 3),
        ('flaw', 'lawn', 2),
        ('test', 'test', 0),
        ('test', 'testing', 3),
        ('test', 'tester', 2),
    ],
)
def test_osa(source, target, expected):
    assert osa(source, target) == expected
