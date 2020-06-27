"""
Decision Tree

This file holds a Decision Tree class along with an implementation of the ID3 algorthim.

The ID3 Algorithm is used for creating new Decision Trees.

DataFrame entropies, information gain, and gini index can also be calculated using fucntions in this module.

"""

__author__ = "Jon Wiggins"

import math


def entropy(data: "dataframe", target_label: str) -> float:
    """
    Calculates the entroy of the given dataframe with respect to the column given by the target_label

    :param data: A dataframe of examples
    :param target_label: A name of a column in the dataframe 

    :returns: the entrophy
    """
    to_return = 0
    value_probs = [count / data.shape[0] for count in data[target_label].value_counts()]
    to_return += sum(value_prob * math.log2(value_prob) for value_prob in value_probs)

    return to_return * -1


def gini_index(data: "dataframe", target_label: str) -> float:
    """
    Calculates the gini index of the given dataframe with respect to the column given by the target_label

    :param data: A dataframe of examples
    :param target_label: A name of a column in the dataframe 

    :returns: the gini index
    """
    to_return = 1

    for value in data[target_label].unique():
        subset = data[data[target_label] == value]
        value_prob = subset.shape[0] / data.shape[0]
        to_return -= value_prob * value_prob
    return to_return


def information_gain(
    data: "dataframe",
    target_label: str,
    attribute: str,
    impurity_metric: "function" = entropy,
) -> float:
    """
    Calculates the information gain of the given attribute with respect to the target_label in the given dataframe

    :param data: A dataframe of examples
    :param target_label: A name of a column in the dataframe 
    :param attribute:  A name of a column in the dataframe 
    :param impurity_metric: A function for calculating impurity, defaults to entropy

    :returns: the information gain of the attribute
    """
    to_return = impurity_metric(data, target_label)

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        value_prob = subset.shape[0] / data.shape[0]
        to_return -= value_prob * impurity_metric(subset, target_label)
    return to_return


def fast_information_gain_on_entropy(
    data: "dataframe", target_label: str, attribute: str
) -> float:
    """
    Calculates the information gain of the given attribute with respect to the target_label in the given dataframe
    Does this a bit quicker than the above method as it only uses entropy

    :param data: A dataframe of examples
    :param target_label: A name of a column in the dataframe 
    :param attribute:  A name of a column in the dataframe 

    :returns: the information gain of the attribute
    """
    to_return = 0
    value_probs = [count / data.shape[0] for count in data[target_label].value_counts()]
    to_return += sum(value_prob * math.log2(value_prob) for value_prob in value_probs)
    to_return *= -1

    counts = data[attribute].value_counts()
    for index, count in zip(counts.index, counts):
        value_probs = [
            subset_count / data.shape[0]
            for subset_count in data[data[attribute] == index][
                target_label
            ].value_counts()
        ]
        to_return -= sum(
            value_prob * math.log2(value_prob) for value_prob in value_probs
        )

    return to_return


class DecisionTree:
    """
    This class acts as a single node in a Decison Tree. 
    It holds:
        - attribute: The attribute this node acts on
        - is_leaf: a flag for if this is a leaf node
        - label: if this node is a leaf, this will contain the target label value predicted
        - gain: the gain calculated by ID3 when creating this node 
    """

    def __init__(
        self,
        attribute: str = None,
        is_leaf: bool = False,
        label: str = None,
        gain: float = None,
    ):
        """
        Create a new Decision Tree
        """
        self.is_leaf = is_leaf
        self.label = label
        self.attribute = attribute
        self.branches = {}
        self.gain = gain

    def add_branch(self, label: str, decision_tree: "DecisionTree"):
        """
        Adds a new branch to this node's children

        :param label: the value of this node's attribute that this child should be called on
        :param decison_tree: a decison tree to add as a child
        """
        self.branches[label] = decision_tree

    def traverse(self, values: "dataframe") -> str:
        """
        Probes this decision tree, and it's children recursively for the target_label
        Predicted by the given values
        
        :param values: A dictionary of the examples attributes

        :returns: The predicted target label for this set of values
        """
        if self.is_leaf or values[self.attribute] not in self.branches:
            return self.label

        return self.branches[values[self.attribute]].traverse(values)

    def find_max_depth(self) -> int:
        """
        A helper method that find the depth of this tree from its childre

        :returns: The depth as an int
        """
        if len(self.branches) == 0:
            return 0

        return max(branch.find_max_depth() for branch in self.branches.values()) + 1


def ID3(
    examples: "dataframe",
    target_label: str,
    metric: "function" = fast_information_gain_on_entropy,
    depth_budget: int = None,
) -> DecisionTree:
    """
    This function implements the ID3 algorithm
    It will return a DecisionTree constructed for the target_label using the givne dataframe of examples
    :param examples: Set of possible labels
    :param target_label: Target Attribute
    :param metric: entrophy method, defaults to information_gain
    :param depth_budget: max depth of the tree

    :return: a DecisionTree
    """

    # if all examples have the same value, or the budget is expired, or there are no attributes left to pick
    #  return a node with the most popular label
    if (
        depth_budget == 0
        or len(examples[target_label].unique()) == 1
        or examples.shape[1] == 2
    ):
        return DecisionTree(
            is_leaf=True, label=examples[target_label].value_counts().idxmax()
        )

    examples = examples.sample(frac=1)

    max_metric = None
    attribute = None
    for element in examples.columns.values:
        if element == target_label:
            continue

        # next_metric = metric(examples, target_label, element, entropy)
        next_metric = metric(examples, target_label, element)
        if attribute is None or next_metric > max_metric:
            max_metric = next_metric
            attribute = element
    most_common_label = examples[target_label].value_counts().idxmax()
    to_return = DecisionTree(
        attribute=attribute, label=most_common_label, gain=max_metric
    )
    # for each possible value of this attribute
    for value in examples[attribute].unique():
        example_subset = examples[examples[attribute] == value]
        example_subset = example_subset.drop(attribute, axis=1)
        if depth_budget is None:
            to_return.add_branch(value, ID3(example_subset, target_label))
        else:
            to_return.add_branch(
                value, ID3(example_subset, target_label, depth_budget=depth_budget - 1)
            )

    return to_return
