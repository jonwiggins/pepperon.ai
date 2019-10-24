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
    return math.sqrt(sum([pow(x - y, 2) for x, y in zip(first, second)]))


def manhatten_distance(first, second):
    """
    Given two vectors, returns the manhatten distance between them
    Requires that they are the same dimension

    :param first: a vector
    :param second: a vector

    :return: the distance as a double
    """
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")

    return sum([abs(x - y) for x, y in zip(first, second)])


def cosine_similarity(first, second):
    """
    Given two vectors, returns the cosine similarity between them
    Requires that they are the same dimension

    :param first: a vector
    :param second: a vector

    :return: the similarity as a double
    """
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")
    numerator = sum(x * y for x, y in zip(first, second))
    denominator = math.sqrt(sum(pow(x, 2) for x in first)) * math.sqrt(sum(pow(y, 2) for y in second))
    if denominator == 0:
        return 0
    return numerator / denominator

    
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


def heirarchial_cluster(
    points, center_count, distance_function=euclidian_distance, link=mean_link
):
    """
    Clusters the given points based on heirarchial clustering: the distance between two clusters is calculated using the given link function

    :param points: all of the datapoints to cluster
    :param center_count: the number of clusters to produce
    :param distance_function: a function that will compare two datapoints
    :param link: function that defines how two clusters will be compared, this library includes single_link, complete_link, and mean_link

    :return: a dictionary that maps centers to the points in its cluster
    """
    clusters = [set(point) for point in points]

    while center_count < len(clusters):
        first_neighbor_index = None
        second_neighbor_index = None
        min_neighbor_distance = None

        for neighbor1 in range(len(clusters)):
            for neighbor2 in range(neighbor1 + 1, len(clusters)):
                current_distance = link(
                    clusters[neighbor1], clusters[neighbor2], distance_function
                )

                if not min_neighbor_distance or current_distance < current_distance:
                    first_neighbor_index = neighbor1
                    second_neighbor_index = neighbor2
                    min_neighbor_distance = current_distance

        clusters[first_neighbor_index] = clusters[first_neighbor_index].union(
            clusters[second_neighbor_index]
        )
        del clusters[second_neighbor_index]

    return clusters


def single_link(first_clustering, second_clustering, distance_function):
    """
    Finds the smallest distance between any two points in the two given clusters

    :param first_clustering: a set of points
    :param second_clustering: a set of points
    :param distance_function: a function that compares two points

    :return: the smallest distance between a pair of points in the two clusters
    """
    min_neighbor_distance = None
    for first_neighbor in first_clustering:
        for second_neighbor in second_clustering:
            current_distance = distance_function(first_neighbor, second_neighbor)
            if not min_neighbor_distance or min_neighbor_distance > current_distance:
                min_neighbor_distance = current_distance
    return min_neighbor_distance


def mean_link(first_clustering, second_clustering, distance_function):
    """
    Finds the average distance between each pair of points in the two clusters

    :param first_clustering: a set of points
    :param second_clustering: a set of points
    :param distance_function: a function that compares two points

    :return: the average distance between each pair of points in the two clusters
    """
    total_distance = 0.0
    for first_neighbor in first_clustering:
        for second_neighbor in second_clustering:
            total_distance += distance_function(first_neighbor, second_neighbor)

    return total_distance / (len(first_clustering) * len(second_clustering))


def complete_link(first_clustering, second_clustering, distance_function):
    """
    Finds the largest distance between any two points in the two given clusters

    :param first_clustering: a set of points
    :param second_clustering: a set of points
    :param distance_function: a function that compares two points

    :return: the largest distance between a pair of points in the two clusters
    """
    max_neighbor_distance = None
    for first_neighbor in first_clustering:
        for second_neighbor in second_clustering:
            current_distance = distance_function(first_neighbor, second_neighbor)
            if not max_neighbor_distance or max_neighbor_distance < current_distance:
                max_neighbor_distance = current_distance
    return max_neighbor_distance

