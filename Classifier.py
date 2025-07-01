from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):

    class Tree():
        def __init__(self, feature_index: int = None, threshold: float = None, left: 'Classifier.Tree' = None, right: 'Classifier.Tree' = None, lvalue: int = None, rvalue: int = None):
            """
            A node in the decision tree.
            :param feature_index: Index of the feature to split on.
            :param threshold: Threshold value for the split.
            :param left: Left child node.
            :param right: Right child node.
            :param value: Class label if it's a leaf node.
            """
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.lvalue = lvalue
            self.rvalue = rvalue
            
    def gini_impurity(self, x):
        """
        Calculate the Gini impurity for a given set of labels.

        :param x: A list of labels. (Series, list, DataFrame-column, etc.)
        :return: The Gini impurity value.
        """
        if x.size == 0:
            return 0.0

        counts = np.bincount(x)
        prob_sq = (counts / len(x)) ** 2
        return 1 - prob_sq.sum()
    
    def weighted_average(self, gi1, gi2, w1, w2):
        """Calculate Weighted Gini Impurity Average of two Gini impurity values"""
        return {0:(gi1 * w1)/(w1+w2), 1:(gi2 * w2) / (w1 + w2)}