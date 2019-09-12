import random


# Gets the center associated with the given metric, using the given distance function.
def get_center(centers, metric, distance_function):
    min_distance = distance_function(metric, centers[0])
    min_center = centers[0]

    for center in centers[1:]:
        distance = distance_function(metric, center)
        if distance < min_distance:
            min_distance = distance
            min_center = center

    return min_center


# Gets the distance to the center associated with the given metric, using the given distance function
def distance_to_center(centers, metric, distance_function):
    return distance_function(metric, get_center(centers, metric, distance_function))


def centers_to_sets(centers, metrics, distance_function):
    sets = []

    for center in centers:
        subset = set()

        for metric in metrics:
            if get_center(centers, metric, distance_function) == center:
                subset.add(metric)

        sets.append(subset)

    return sets


def k_means_pp(metrics, center_count, distance_function):
    centers = [metrics[0]]

    while len(centers) < center_count:
        distances = [pow(distance_to_center(centers, metric, distance_function), 2) for metric in metrics]
        total_distance = sum(distances)

        for index in range(len(distances)):
            distances[index] /= total_distance

        uniform = random.uniform(0.0, 1.0)

        cumulative = 0.0

        for index in range(len(distances)):
            cumulative += distances[index];

            if uniform <= cumulative:
                centers.append(metrics[index])
                break

    return centers_to_sets(centers, metrics, distance_function)


def gonzales(metrics, center_count, distance_function):
    centers = list()
    centers.append(metrics[0])
    dict_clusters = {}
    for point in metrics:
        dict_clusters[point] = centers[0]

    for index in range(1, center_count):
        centers.append(metrics[0])
        max_min_distance = 0
        new_center = None
        for comparison_index in range(0, len(metrics)):
            min_distance = min([distance_function(metrics[comparison_index], c) for c in centers])
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                new_center = metrics[comparison_index]
        centers[index] = new_center

        for comparison_index in range(0, len(metrics)):
            if distance_function(metrics[comparison_index], dict_clusters[metrics[comparison_index]]) > distance_function(metrics[comparison_index], centers[index]):
                dict_clusters[metrics[comparison_index]] = centers[index]

    # return list of sets
    return centers_to_sets(centers, metrics, distance_function)


def single_link(metrics, center_count, point_distance_function):
    return hierarchical_cluster(metrics, center_count, point_distance_function, single_link_distance)


def complete_link(metrics, center_count, point_distance_function):
    return hierarchical_cluster(metrics, center_count, point_distance_function, complete_link_distance)


def hierarchical_cluster(metrics, desired_cluster_count, point_distance_function, set_distance_function):
    clusters = []

    for point in metrics:
        point_set = set()
        point_set.add(point)
        clusters.append(point_set)

    while desired_cluster_count < len(clusters):
        closest_cluster1 = 0
        closest_cluster2 = 1
        cluster_distance = set_distance_function(clusters[0], clusters[1], point_distance_function)
        for index in range(0, len(clusters)):
            for comparison_index in range(index + 1, len(clusters)):
                new_cluster_distance = set_distance_function(clusters[index], clusters[comparison_index], point_distance_function)
                if new_cluster_distance < cluster_distance:
                    closest_cluster1 = index
                    closest_cluster2 = comparison_index
                    cluster_distance = new_cluster_distance
        merge_clusters(clusters, closest_cluster1, closest_cluster2)

    return clusters


def single_link_distance(set1, set2, distance_function):
    shortest_distance = float("inf")
    for point1 in set1:
        for point2 in set2:
            if distance_function(point1, point2) < shortest_distance:
                shortest_distance = distance_function(point1, point2)
    return shortest_distance


def complete_link_distance(set1, set2, distance_function):
    largest_distance = 0
    for point1 in set1:
        for point2 in set2:
            if distance_function(point1, point2) > largest_distance:
                largest_distance = distance_function(point1, point2)
    return largest_distance


def merge_clusters(cluster_data, index1, index2):
    cluster_data[index1] = cluster_data[index1].union(cluster_data[index2])
    cluster_data.pop(index2)


def get_max_distance(instructor, cluster, distanceFunction):
    maxDistance = 0

    for otherInstructor in cluster:
        distance = distanceFunction(instructor, otherInstructor)

        if distance > maxDistance:
            maxDistance = distance

    return maxDistance


def get_representative(cluster, distance_function):
    representative = None
    representative_distance = None

    for element in cluster:
        distance = get_max_distance(element, cluster, distance_function)

        if representative is None or distance < representative_distance:
            representative = element
            representative_distance = distance

    return representative


def refine_clusters(clusters, instructors, distance_function):
    new_centers = []

    for cluster in clusters:
        new_centers.append(get_representative(cluster, distance_function))

    return centers_to_sets(new_centers, instructors, distance_function)

