import pytest
from pepperonai.unsupervised.distance import *


def test_euclidean():
    assert euclidian_distance([1, 1], [1, 1]) == 0
    assert euclidian_distance([1, 0], [1, 1]) == 1
    with pytest.raises(Exception):
        euclidian_distance([1], [1, 2])


def test_jaccard():
    apple = set(["apple"])
    applepie = set(["apple", "pie"])
    pie = set(["pie"])
    fruit = set(["apple", "pear", "mango"])
    assert jaccard_similarity(apple, apple) == 1
    assert jaccard_similarity(apple, applepie) == 0.5
    assert jaccard_similarity(apple, fruit) == (1 / 3)
