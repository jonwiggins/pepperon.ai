import math


def entropy(data, target_label):
    to_return = 0
    for value in data[target_label].unique():
        value_prob = (data[data[target_label] == value].shape[0]) / data.shape[0]
        to_return += (value_prob * math.log2(value_prob))
    return to_return * -1


def information_gain(data, target_label, attribute, impurity_metric):
    to_return = impurity_metric(data, target_label)

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        value_prob = subset.shape[0] / data.shape[0]
        to_return -= value_prob * impurity_metric(subset, target_label)
    return to_return


def gini_index(data, target_label):
    to_return = 1

    for value in data[target_label].unique():
        subset = data[data[target_label] == value]
        value_prob = subset.shape[0] / data.shape[0]
        to_return -= (value_prob * value_prob)
    return to_return


def ID3(examples, target_label, metric=information_gain, depth_budget=None):
    """
    ID3 Alg
    :param examples: Set of possible labels
    :param target_label: Target Attribute
    :param metric: entrophy method
    :param depth_budget: max depth of the tree

    :return: a DecisionTree
    """

    # if all examples have the same value, or the budget is expired, or there are no attributes left to pick
    #  return a node with the most popular label
    if depth_budget == 0 or len(examples[target_label].unique()) == 1 or examples.shape[1] == 2:
        # print("made leaf")
        return DecisionTree(is_leaf=True, label=examples[target_label].value_counts().idxmax())
    
    examples = examples.sample(frac=1)

    max_metric = None
    attribute = None
    for element in examples.columns.values:
        if element == target_label:
            continue

        next_metric = metric(examples, target_label, element, entropy)
        if attribute is None or next_metric > max_metric:
            max_metric = next_metric
            attribute = element
    # print("picked attribute: ", attribute)
    most_common_label = examples[target_label].value_counts().idxmax()
    to_return = DecisionTree(attribute=attribute, label=most_common_label, gain=max_metric)
    # print("picked most common label: ", most_common_label)
    # for each possible value of this attribute
    for value in examples[attribute].unique():
        example_subset = examples[examples[attribute] == value]
        example_subset = example_subset.drop(attribute, axis=1)
        # if example_subset.shape[1] == 2:
        #     continue  # the only columns left are the label and index, parsing will yield most_common_label
        if depth_budget is None:
            to_return.add_branch(value, ID3(example_subset, target_label))
        else:
            to_return.add_branch(value, ID3(example_subset, target_label, depth_budget=depth_budget - 1))

    return to_return


class DecisionTree:

    def __init__(self, attribute=None, is_leaf=False, label=None, gain=None):
        self.is_leaf = is_leaf
        self.label = label
        self.attribute = attribute
        self.branches = {}
        self.gain = gain

    def add_branch(self, label, decision_tree):
        self.branches[label] = decision_tree

    def traverse(self, values):
        if self.is_leaf or values[self.attribute] not in self.branches:
            return self.label

        return self.branches[values[self.attribute]].traverse(values)
        # return self.label  # or does it go to the most popular branch?

    def find_max_depth(self):
        if len(self.branches) == 0:
            return 0

        return max(branch.find_max_depth() for branch in self.branches.values()) + 1
