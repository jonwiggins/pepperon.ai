"""
SVM Via SGD
"""
__author__ = "Jon Wiggins"


import pandas as pd
import numpy as np

from model import *

def sign(number: int) -> int:
    """ Maps the sign of the number to -1 or 1"""
    if number < 0:
        return -1
    else:
        return 1


class SVM(Model):
    """
    SVM Via SGD
    """

    def __init__(self):
        self.weight = None
        self.bias = None
        self.update_count = 0
        self.epoch_saves = []

    def train(
        self,
        examples: "dataframe",
        target_labels: "List[float]",
        initial_learning_rate: float = 0.1,
        loss_tradeoff: float = 0,
        epochs: int = 1,
        rate_decay: bool = True,
    ):
        """
        Trains a SVM on examples

        :param examaples: a matrix wherein each row is one training example
        :param target_labels: an array where each element's index corresponds to the label for examples
        :param inital_learning_rate: as a double
        :param loss_tradeoff: as a double
        :param epochs: count to train the model to
        :param rate_date: decreases learning_rate with each epoch iff true
        """

        self.weight = np.random.uniform(low=-0.01, high=0.01, size=(len(examples[0]),))
        self.bias = np.random.uniform(low=-0.01, high=0.01)
        self.update_count = 0
        self.epoch_saves = []

        for count in range(epochs):
            indicese = np.arange(len(target_labels))
            np.random.shuffle(indicese)

            if rate_decay:
                learning_rate = initial_learning_rate / (1 + count)

            for index in indicese:
                x = examples[index].T
                y = target_labels[index]
                prediction = y * self.weight.T.dot(x) + self.bias

                if prediction <= 1:

                    self.weight = (1 - learning_rate) * self.weight + (
                        learning_rate * loss_tradeoff * y * x
                    )
                    self.bias = self.bias + (learning_rate * y)
                else:
                    self.weight = (1 - learning_rate) * self.weight
            self.epoch_saves.append((self.weight, self.bias))

    def probe(self, x: "List[float]", epoch: int = None) -> int:
        """
        Probes the model for a guess on the given input

        :param x: An array with which to probe the model
        :param epoch: If desired, give an epoch number and this will probe the state of the model after that training epoch
        """
        if epoch:
            return sign(
                self.epoch_saves[epoch][0].dot(x.T) + self.epoch_saves[epoch][1]
            )
        else:
            return sign(self.weight.T.dot(x) + self.bias)
