"""
This heavy hitters streaming model esitmates occurance rates of streamed data
Use it by initilizing a cache and then adding elements to it
Then probe the cache for occurance estimates

This will always give an upper bound estimate that is erroneous by a factor of the size of the cache and the total number of elements
"""
__author__ = "Jon Wiggins"

import numpy as np
import random


class CountMinSketch:
    """
    This class holds a count-sketch cache
    """

    def __init__(self, function_count, cache_length, warmend_cache=None):
        """
        Create a new cache

        :param function_count: the number of hash functions
        :param cache_length: the length of the cache for each function
        :param warmed_cache: initilizes this cache from a given one
        """
        if not warmend_cache:
            self.cache = np.zeros((function_count, cache_length))
        self.function_count = function_count
        self.cache_length = cache_length
        self.count = 0

    def stream_input(self, element):
        """
        Inputs a single element to the cache

        :param input: an element to input
        """
        self.count += 1
        random.seed(hash(element))

        for hash_index in range(self.function_count):
            length_index = random.randint(0, self.cache_length)
            self.cache[hash_index, length_index] += 1

    def get_cache(self):
        """
        Returns the current state of the cache

        :return: the cache
        """
        return self.cache

    def probe(self, element):
        """
        Probes the cache for the given element

        :param element: the element to probe for

        :return: the lowest estimated value for the value in the cache
        """
        random.seed(hash(element))
        return min(
            [
                self.cache[index, random.randint(0, self.cache_length)]
                for index in range(self.function_count)
            ]
        )

    def get_count(self):
        """
        Gets the count of the total ammount of things seen by the cache

        :return: the total count of elements streamed
        """
        return self.count
