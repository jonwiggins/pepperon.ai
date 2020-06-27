"""
K Nearest Neighbors
"""

__author__ = "Jon Wiggins"

from collections import Counter
import pandas as pd


class KNN:
    def __init__(self):
        pass

    def train(
        self, data: "pd.DataFrame", target_label: str, distance_function: "function",
    ):
        """
        Feed a KNN model a dataset

        :param data: a dataframe
        :param target_label: a column in the dataframe
        :param distance_function: a function to calculate distance with
            Must be callable via distance_function(first, second) and return a float 
        """
        self.data = data
        self.target_labal = target_label
        self.distance_function = distance_function

    def probe(self, example: "dict", k: int, return_counts: bool = True) -> str:
        """
        Returns prediction for example

        :param example: a dataframe row/dict
        :param k: number of near-neighbors to consider
        :param return_counts: if true, returns counter of k near-neighbor labels
        """
        distances = []

        for element in data:
            distances.append(
                (self.distance_function(element, example), element[self.target_labal])
            )

        # sort by distance
        distances.sort(key=lambda x: x[0])
        # select k nearest neighbors
        k_neighbors = distances[:k]
        # return the most common neighbor class
        classes = [element[1] for element in k_neighbors]
        counter = Counter(classes)
        to_return = counter.most_common(1)

        if return_counts:
            return to_return
        return to_return[0]
