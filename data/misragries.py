"""
This heavy hitters streaming model esitmates occurance rates of streamed data
Use it by initilizing a cache and then adding elements to it
Then probe the cache for occurance estimates

This will always give a lower bound estimate that is erroneous by a factor of the size of the cache and the total number of elements
"""

__author__ = "Jon Wiggins"


class MisraGries:
    """
    This class holds a Mirsa-Gries cache
    """

    def __init__(self, cache_size, warmed_cache={}):
        """
        Create a new cache

        :param cache_size: the maxium capacity of the cache to make
        :param warmed_cache: initilizes this cache from a given one
        """
        self.cache = warmed_cache
        self.cache_size = cache_size
        self.count = 0

    def stream_multiple(self, input):
        """
        Inputs an iteratable to the cache
        
        :param input: an iteratable
        """
        for element in input:
            self.stream_input(element)

    def stream_input(self, input):
        """
        Inputs a single element to the cache

        :param input: an element to input
        """
        self.count += 1
        if input in self.cache:
            self.cache[input] += 1
        elif len(self.cache) < self.cache_size - 1:
            self.cache[input] = 1
        else:
            for key, value in self.cache.items():
                self.cache[key] = value - 1
                if self.cache[key] == 0:
                    del self.cache[key]

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

        :return: 0 iff element is not in cache, otherwise returns the cache's count for element
        """
        if element in self.cache:
            return self.cache[element]
        else:
            return 0

    def get_count(self):
        """
        Gets the count of the total ammount of things seen by the cache

        :return: the total count of elements streamed
        """
        return self.count

