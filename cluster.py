"""
One day I will implement these two Streaming algorithms here
Until then, feel free to call these methods just for fun!
"""

__author__ = "Jon Wiggins"

import math

def euclidian_distance(first, second):
    if len(first) != len(second):
        raise Exception("These vectors must be the same size")
    return math.sqrt(sum([pow(first[index] - second[index], 2) for index in range(len(first))]))

def jaccard_similarity(first, second):
    return len(first & second) / len(first | second)


class kmeans_pp:
    pass

