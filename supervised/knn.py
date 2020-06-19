"""
K Nearest Neighbors
"""

__author__ = "Jon Wiggins"

from collections import Counter
import pandas as pd
from model import *
import utils


class KNN(Model):
    def __init__(self):
        pass

    def train(
        self,
        data: pd.DataFrame,
        target_label: Any,
        distance_function=euclidian_distance,
    ):
        self.data = data
        self.target_labal = target_label
        self.distance_function = distance_function

    def probe(self, example: Any, k: int, return_counts: Bool = True) -> Any:
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
