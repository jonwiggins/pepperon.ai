"""
One day I will implement these clustering algorthms
Here are some cleaned helper methods
Until then, feel free to call these methods just for fun!
"""

__author__ = "Jon Wiggins"

import math


def euclidian_distance(first, second):
    """
    Given two vectors, returns the euclidian distance between them
    Requires that they are the same dimension

    :param first: a vector
    :param second: a vector

    :return: the distance as a double
    """
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")
    return math.sqrt(
        sum([pow(first[index] - second[index], 2) for index in range(len(first))])
    )


def jaccard_similarity(first, second):
    """
    Given two sets, returns the jaccard similarity between them

    :param first: a set
    :param second: a set

    :return: the similarity as a double
    """
    return len(first & second) / len(first | second)


def fowlkesmallowsindex(first_clustering, second_clustering, all_data_points):
    """
    Given two clusterings and a list of all the points, calculates the Fowlkes-Mallows Index

    :param first_clustering: the first set of iterables to compare
    :param second_clustering: the second set of iterables to compare
    :param all_data_points: all of the datapoints in the two clusterings as an iterable

    :return: the Fowlkes Mallows Index as a double
    """

    # TP = the number of points that are present in the same cluster in both clusterings
    # FP = the number of points that are present in the same cluster in clustering1 but not clustering2
    # FN = the number of points that are present in the same cluster in clustering2 but not clustering1
    # TN = the number of points that are in different clusters in both clusterings
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for element in all_data_points:
        elements_first_location = None
        for cluster in first_clustering:
            if element in cluster:
                elements_first_location = cluster
                break

        elements_second_location = None
        for cluster in second_clustering:
            if element in cluster:
                elements_second_location = cluster
                break

        for comparison_element in all_data_points:
            comparisons_first_cluster = None
            for cluster in first_clustering:
                if comparison_element in cluster:
                    comparisons_first_cluster = cluster
            comparisons_second_cluster = None
            for cluster in second_clustering:
                if comparison_element in cluster:
                    comparisons_second_cluster = cluster

            if (
                elements_first_location == comparisons_first_cluster
                and elements_second_location == comparisons_second_cluster
            ):
                TP += 1
            elif (
                elements_first_location == comparisons_first_cluster
                and not elements_second_location == comparisons_second_cluster
            ):
                FP += 1
            elif (
                not elements_first_location == comparisons_first_cluster
                and elements_second_location == comparisons_second_cluster
            ):
                FN += 1
            elif (
                not elements_first_location == comparisons_first_cluster
                and not elements_second_location == comparisons_second_cluster
            ):
                TN += 1

    if TP + FP == 0 or TP + FN == 0:
        return 0

    return math.sqrt((TP / (TP + FP))) * (TP / (TP + FN))


def purity(first_clustering, second_clustering):
    """
    Returns the purity of the given two clusterings

    :param first_clusterings: a set of iterables to compare
    :param second_clusterings: a set of iterables to compare

    :return: the purity index as a double
    """
    summation = 0

    for cluster in first_clustering:
        highest = 0
        for comparer in second_clustering:
            next_element = len(cluster.intersection(comparer))
            if next_element > highest:
                highest = next_element

        summation += highest

    # find total number of data points
    N = sum(len(cluster) for cluster in first_clustering)

    if N == 0:
        return 0

    return summation / N


def kmeans_pp():
    pass


def gonzales():
    pass

