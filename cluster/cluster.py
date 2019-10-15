"""
Holds several static methods used for clustering unlabeled data
Also holds comparison methods for datpoints and clusterings

"""

__author__ = "Jon Wiggins"

import math
import random


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


def get_nearest_center(element, centers, distance_function):
    """
    Get the center that is nearest to the given point
    
    :param element: a point to search based on
    :param centers: an iteratable of the center points
    :param distance_function: a function that will compare two datapoints

    :return: an element from centers that is closest to the distance function
    """
    closest_distance = None
    closest_center = None

    for center in centers:
        current_distance = distance_function(element, center)

        if not closest_center or closest_distance > current_distance:
            closest_distance = current_distance
            closest_center = center

    return closest_center


def get_distance_to_center(element, centers, distance_function):
    """
    Returns  the distance from the given point to its center

    :param element: a point to get the distance for
    :param centers: an iteratable of the center points
    :param distance_function: a function that will compare two datapoints

    :return: the distance from the element to its center
    """
    return distance_function(
        element, get_nearest_center(centers, element, distance_function)
    )


def kmeans_pp(points, center_count, distance_function=euclidian_distance):
    """
    Clusters based on the kmeans++ algorithm

    :param points: all of the datapoints to cluster
    :param center_count: the number of clusters to produce
    :param distance_function: a function that will compare two datapoints

    :return: a dictionary that maps centers to the points in its cluster
    """

    # start with any arbitrary center
    centers = set(points[0])

    while len(centers) < center_count:

        decider_value = random.uniform(0.0, 1.0)

        distances_to_centers = [
            pow(get_distance_to_center(element, centers, distance_function), 2)
            for element in points
        ]

        # normalize the distances
        total = sum(distances_to_centers)
        distances_to_centers = [element / total for element in distances_to_centers]

        counter = 0.0

        for index, element in enumerate(distances_to_centers):
            counter += distances_to_centers

            if counter >= decider_value:
                centers.append(points[index])
                break

    # make a dict that maps a center to the points in its cluster
    to_return = {}

    for center in centers:
        to_add = set()
        for index, element in points:
            if get_nearest_center(element, centers, distance_function) == center:
                to_add.add(element)
        to_return[center] = to_add

    return to_return


def gonzales(points, center_count, distance_function=euclidian_distance):
    """
    Clusters the given points based on the Greedy Gonzales algorithmn

    :param points: all of the datapoints to cluster
    :param center_count: the number of clusters to produce
    :param distance_function: a function that will compare two datapoints

    :return: a dictionary that maps centers to the points in its cluster
    """
    centers = set(points[0])

    while len(centers) < center_count:
        max_distance = None
        max_point = None
        for index, element in enumerate(points):
            lowest_distance_to_center = min(
                [distance_function(element, center) for center in centers]
            )

            if not max_distance or lowest_distance_to_center > max_distance:
                max_distance = lowest_distance_to_center
                max_point = element
        centers.append(element)

    # make a dict that maps a center to the points in its cluster
    to_return = {}

    for center in centers:
        to_add = set()
        for index, element in points:
            if get_nearest_center(element, centers, distance_function) == center:
                to_add.add(element)
        to_return[center] = to_add

    return to_return


def single_link():
    pass


def mean_link():
    pass


def complete_link():
    pass
