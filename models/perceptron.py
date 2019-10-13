"""
Perceptron

This file holds classes for two seperate implementations of the Perceptron Algorithm

    - Simple Perceptron: A classic renditon of this mistake bound linear classifier with no funny business
    - Average Perceptron: An improvement on the simple version which probes based on the average training weights

Both of these classes are built to take sparse matrices as inputs and return 1 or -1 when probed
"""

__author__ = "Jon Wiggins"


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def sign(number):
    """
    This helper method returns 0, 1, or -1 based on the sign of the given number

    :param number: a number

    :return: 0 iff number is 0, -1 iff number is negative, and 1 iff number is positive 
    """
    if number == 0:
        return 0
    if number < 0:
        return -1
    else:
        return 1


class SimplePerceptron:
    """
    This class implements a perceptron algorithm - with no funny business
    """

    def __init__(self):
        self.weight = None
        self.bias = None
        self.update_count = 0
        self.epoch_saves = []

    def train(
        self,
        examples,
        target_labels,
        dimens,
        learning_rate=0.1,
        epochs=1,
        rate_decay=True,
    ):
        """
        Trains a Simple perceptron with the given input
        
        :param exaples: A sparse matrix wherein each row is a vector example corresponding to target_labels
        :param target_labels: An array wherein each element is a label corresponding to target_labels
        :param dimens: An int indicating the dimensions of the example space
        :param learning_rate: The rate at which the model should adjust its weight/bias on incorrect guesses, default is 0.1
        :param epochs: The number of trining epochs
        :param rate_decay: Boolean, true for decayed rate of learning in later epochs, default is true
        """
        decay = learning_rate
        self.weight = np.random.uniform(low=-0.01, high=0.01, size=(1, dimens))
        self.bias = np.random.uniform(low=-0.01, high=0.01)
        self.update_count = 0
        self.epoch_saves = []

        for count in range(epochs):
            indicese = np.arange(len(target_labels))
            np.random.shuffle(indicese)

            if rate_decay:
                learning_rate = decay / (1 + count)

            for index in indicese:
                x = examples.getrow(index).todense()
                y = target_labels[index]

                prediction = sign(self.weight.dot(x.T) + self.bias)

                if prediction != y:
                    # update if prediction is incorrect
                    self.weight = self.weight + (learning_rate * (y * x))
                    self.bias = self.bias + (learning_rate * y)
                    self.update_count += 1

            self.epoch_saves.append((self.weight, self.bias))

    def probe(self, x, epoch=None):
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
            return sign(self.weight.dot(x.T) + self.bias)


class AveragedPerceptron:
    """
    This class implements an averaged perceptron algorithm
    """

    def __init__(self):
        self.weight = None
        self.model = None
        self.bias = None
        self.update_count = 0
        self.epoch_saves = []

    def train(
        self,
        examples,
        target_labels,
        dimens,
        learning_rate=0.1,
        epochs=1,
        rate_decay=True,
    ):
        """
        Trains an averaged perceptron with the given input
        
        :param exaples: A sparse matrix wherein each row is a vector example corresponding to target_labels
        :param target_labels: An array wherein each element is a label corresponding to target_labels
        :param dimens: An int indicating the dimensions of the example space
        :param learning_rate: The rate at which the model should adjust its weight/bias on incorrect guesses, default is 0.1
        :param epochs: The number of trining epochs
        :param rate_decay: Boolean, true for decayed rate of learning in later epochs, default is true
        """
        self.bias = np.random.uniform(low=-0.01, high=0.01)
        self.weight = np.random.uniform(low=-0.01, high=0.01, size=(1, dimens))
        weight_summation = self.weight
        bias_summation = 0
        self.update_count = 0
        self.epoch_saves = []
        decay = learning_rate

        for count in range(epochs):
            indicese = np.arange(len(target_labels))
            np.random.shuffle(indicese)

            if rate_decay:
                learning_rate = decay / (1 + count)

            for index in indicese:
                x = examples.getrow(index).todense()
                y = target_labels[index]

                prediction = sign(self.weight.dot(x.T) + self.bias)

                if prediction != y:
                    # update if pidiction is incorrect
                    self.weight = self.weight + (learning_rate * (y * x))
                    self.bias = self.bias + (learning_rate * y)
                    self.update_count += 1

                # update average
                weight_summation = weight_summation + self.weight
                bias_summation += self.bias

            self.epoch_saves.append(
                (
                    weight_summation / len(target_labels),
                    bias_summation / len(target_labels),
                )
            )

        self.model = weight_summation / len(target_labels)
        self.bias = bias_summation / len(target_labels)

    def probe(self, x, epoch=None):
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
            return sign(self.model.dot(x.T) + self.bias)

