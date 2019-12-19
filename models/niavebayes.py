"""
This is a Niave Bayes model
"""
__author__ = "Jon Wiggins"

import pandas as pd
import numpy as np
import math
import collections


class NiaveBayes:
    def bernoulli_probability(self, attribute, subset_with_label, example):
        """
        Uses the bernoulli formula to calculate the probabilty of seeing the attribute in the 
        subset with the value it has in example

        :param attribute: a column in the dataframe
        :param subset_with_labe: a subset of the trianing set dataframe where target_label is the current label being tested
        :param example: the example being tested

        :return: the bernoulli probability
        """
        return (
            subset_with_label[subset_with_label[attribute] == example[attribute]].shape[
                0
            ]
            + self.smoothing_term
        ) / (
            subset_with_label.shape[0]
            + self.smoothing_term
            + self.unique_counts[attribute]
        )

    def gaussian_probability(self, attribute, label, example):
        """
        Approximates a gaussian distribution to calculate the probability the the attribute appearing with label in example
        """
        # this attribute may not be useful for this label
        if attribute not in self.means[label]:
            return 1.0

        # it the variance or mean is zero, it may try to divide by zero
        inner_denominator = self.variances[label][attribute] * 2
        if inner_denominator == 0.0:
            return 1.0
        outer_denominator = np.sqrt(
            2 * np.pi * self.variances[label][attribute]
        ) * np.exp(
            (-(example[attribute] - self.means[label][attribute]) ** 2)
            / (2 * inner_denominator)
        )
        if outer_denominator == 0.0:
            return 1.0

        return 1.0 / outer_denominator

    def __init__(self):
        """ Nothing is needed in the init, train handles the setup """
        pass

    def train(
        self,
        dataframe,
        target_label="target",
        smoothing_term=1,
        continuous_classes=True,
    ):
        """
        Builds NB model from the given data frame

        :param daraframe: a dataframe to train the model on
        :param target_label: a column in the dataframe to build the model on
        :param smoothing_term: add_k smoothing on this int
        :param continuous_classes: true for guassian, false for bernoulli
        """
        self.data = dataframe
        self.smoothing_term = smoothing_term
        self.target_label = target_label
        self.variances = collections.defaultdict(dict)
        self.means = collections.defaultdict(dict)
        self.probability_class = probability_class
        self.attributes = set()
        self.continuous_classes = continuous_classes

        if not self.continuous_classes:
            self.unique_counts = [
                self.data[attribute].unique().shape[0]
                for attribute in self.data.columns.values
            ]
        else:

            for label in self.data[self.target_label].unique():
                subset_with_label = self.data[self.data[self.target_label] == label]
                for attribute in self.data.columns.values:
                    current_variance = subset_with_label[attribute].var()
                    current_mean = subset_with_label[attribute].mean()

                    # skip useless attributes
                    if current_mean == 0 and current_variance == 0:
                        continue
                    self.attributes.add(attribute)
                    self.variances[label][attribute] = subset_with_label[
                        attribute
                    ].var()
                    self.means[label][attribute] = subset_with_label[attribute].mean()

    def probe(self, example):
        """
        Probes the model with the given example
        
        :param example: a pd series
        
        :return: a predicted label
        """
        best_label = None
        best_prob = None
        # calculate the value for each possible label, return the one with the highest prob
        for label in self.data[self.target_label].unique():
            subset_with_label = self.data[self.data[self.target_label] == label]
            label_prob = math.log(subset_with_label.shape[0] / self.data.shape[0], 2)

            for attribute in example.to_dict().keys():
                if attribute == self.target_label or attribute not in self.attributes:
                    continue
                if not self.continuous_classes:
                    label_prob += math.log(
                        self.bernoulli_probability(
                            attribute, subset_with_label, example
                        ),
                        2,
                    )
                else:
                    current_prob = self.gaussian_probability(attribute, label, example)
                    label_prob += math.log(current_prob, 2)

            if best_label is None or best_prob < label_prob:
                best_label = label
                best_prob = label_prob

        return best_label
