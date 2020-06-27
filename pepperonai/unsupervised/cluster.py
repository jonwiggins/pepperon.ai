"""
Holds several static methods used for clustering unlabeled data
Also holds comparison methods for datpoints and clusterings

"""

__author__ = "Jon Wiggins"

import math
import random


def get_nearest_center(
    element: object, centers: "Set[object]", distance_function: "function"
) -> object:
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


def get_distance_to_center(
    element: object, centers: "Set[object]", distance_function: "function"
) -> float:
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


def kmeans_pp(
    points: "List[object]", center_count: int, distance_function: "function",
) -> dict:
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


def gonzales(
    points: "List[object]", center_count: int, distance_function: "function",
) -> dict:
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


def single_link(
    first_clustering: "Set[object]",
    second_clustering: "Set[object]",
    distance_function: "function",
) -> float:
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


def mean_link(
    first_clustering: "Set[object]",
    second_clustering: "Set[object]",
    distance_function: "function",
) -> float:
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


def complete_link(
    first_clustering: "Set[object]",
    second_clustering: "Set[object]",
    distance_function: "function",
) -> float:
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


def heirarchial_cluster(
    points: "List[object]",
    center_count: int,
    distance_function: "function",
    link: "function" = mean_link,
) -> dict:
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
