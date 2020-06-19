"""
A Random forest model
"""
__author__ = "Jon Wiggins"


from decisiontree import *
import pandas as pd
import operator

from model import *


class RandomForest(Model):
    """
    A Random forest model
    """

    def __init__(self):
        self.forest = []
        self.target_label = None

    def train(
        self,
        examples: "dataframe",
        target_label: str,
        depth_budget: int = 1,
        tree_count: int = 10,
        train_size: int = 100,
        feature_size: int = 50,
    ):
        """
        Trains a random forest model

        :param examples: a dataframe to train the model on
        :param target_label: the name of a column in the dataframe
        :param depth_budget: the max depth of the trees
        :param tree_count: the number of trees in the forest
        :param train_size: the number of elements from examples to train each tree on
        :param feature_size: the number of columns from examples to train each tree on
        """
        self.target_label = target_label
        for _ in range(tree_count):
            # select train_size rows from examples
            sample = examples.sample(n=train_size)

            # select feature_size columns randomly from the dataframe
            # drop and then add target_label, so it is guarenteed to be included
            cols = (
                sample.drop(columns=[target_label])
                .columns.to_series()
                .sample(feature_size)
                .tolist()
            )
            cols.append(target_label)
            sample = sample[cols]

            # train a model on the view of examples
            self.forest.append(
                ID3(sample, self.target_label, depth_budget=depth_budget)
            )

    def probe(self, example: "dataframe", return_vector: bool = False) -> str:
        """
        Probes the model on the example

        :param example: a pd series
        :param return_vector: if true, returns a vector with each trees vote

        :return: the model's prediction
        """
        results = [tree.traverse(example) for tree in self.forest]
        if return_vector:
            return results
        else:
            return max(set(results), key=results.count)
