import math
import numpy as np


def euclidian_distance(first: "List[float]", second: "List[float]") -> float:
    """
    Given two vectors, returns the euclidian distance between them
    Requires that they are the same dimension

    :param first: a vector
    :param second: a vector

    :return: the distance as a float
    """
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")
    return math.sqrt(sum([pow(x - y, 2) for x, y in zip(first, second)]))


def minkowski_distance(
    first: "List[float]", second: "List[float]", order: int
) -> float:
    """
    Given two vectors, returns the Minkowski distance between them
    Requires that they are the same dimension

    :param first: a vector
    :param second: a vector
    :param order: int

    :return: the distance as a float
    """
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")
    return pow(
        sum([pow(np.abs(x - y), order) for x, y in zip(first, second)]), 1 / order
    )


def manhatten_distance(first: "List[float]", second: "List[float]") -> float:
    """
    Given two vectors, returns the manhatten distance between them
    Requires that they are the same dimension

    :param first: a vector
    :param second: a vector

    :return: the distance as a float
    """
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")

    return sum([abs(x - y) for x, y in zip(first, second)])


def cosine_similarity(first: "List[float]", second: "List[float]") -> float:
    """
    Given two vectors, returns the cosine similarity between them
    Requires that they are the same dimension

    :param first: a vector
    :param second: a vector

    :return: the similarity as a float
    """
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")
    numerator = sum(x * y for x, y in zip(first, second))
    denominator = math.sqrt(sum(pow(x, 2) for x in first)) * math.sqrt(
        sum(pow(y, 2) for y in second)
    )
    if denominator == 0:
        return 0
    return numerator / denominator


def jaccard_similarity(first: "Set[object]", second: "Set[object]") -> float:
    """
    Given two sets, returns the jaccard similarity between them

    :param first: a set
    :param second: a set

    :return: the similarity as a float
    """
    return len(first & second) / len(first | second)
