import pandas as pd
from abc import ABC, abstractmethod
from collections import Counter
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

class DecisionTreeClassifier(Classifier):
    def __init__(self, max_depth:int =None):
        self.max_depth = max_depth
        self.root = None
        
    def __check_GI(self, X, y):
        """
        This method should implement the logic to find the best feature and threshold to split on based on Gini impurity.
        """
        col_lbl_cts =[np.unique(X[:, i]) for i in range(X.shape[1])]
        thresholds = [(a[:-1]+a[1:])/2 if a.size>1 else [] for a in col_lbl_cts]
        GIs = {}
        for i in range(len(thresholds)):
            for j in thresholds[i]:
                mask = X[:, i] <= j
                X_l, X_r, y_l, y_r = X[mask], X[~mask], y[mask], y[~mask]
                gi_l = self.gini_impurity(y_l)
                gi_r = self.gini_impurity(y_r)
                
                w1, w2 = len(X_l), len(X_r)
                gi_avg = self.weighted_average(gi_l, gi_r, w1, w2)
                GIs[gi_avg[0] + gi_avg[1]] = (i, j, gi_l, gi_r)
        
        return GIs[min(GIs.keys())] if GIs else (None, None, None, None, None, None)
    
    def __build_tree(self, X, y, root, depth=1):
        """
        Recursively build the decision tree.
        :param root: The current node in the tree.
        :param depth: Current depth of the tree.
        """
        fi, thr, gi_l, gi_r, lc_l, lc_l = None, None, None, None, None, None
        if self.max_depth >= depth:
            fi, thr, gi_l, gi_r = self.__check_GI(X, y)
            root.feature_index = fi
            root.threshold = thr
            
        lc_l, lc_r = Counter(y[X[:, fi] <= thr]), Counter(y[X[:, fi] > thr])
        
        if self.max_depth == depth:
            root.lvalue = lc_l.most_common(1)[0][0]
            root.rvalue = lc_r.most_common(1)[0][0]
        else:
            if gi_l != 0:
                root.left = self.Tree()
                self.__build_tree(X[X[:, fi] <= thr], y[X[:, fi] <= thr], root.left, depth + 1)
            else:
                root.lvalue = lc_l.most_common(1)[0][0]
            if gi_r != 0:
                root.right = self.Tree()
                self.__build_tree(X[X[:, fi] > thr], y[X[:, fi] > thr], root.right, depth + 1)
            else:
                root.rvalue = lc_r.most_common(1)[0][0]
            
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        self.root = self.Tree()
        self.__build_tree(X.to_numpy(), y.to_numpy(), root=self.root)
    
    def predict(self, X):
        y_pred = []
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        for _, row in X.iterrows():
            node = self.root
            while node.left or node.right:
                if row.iloc[node.feature_index] <= node.threshold:
                    if node.left:
                        node = node.left
                    else:
                        y_pred.append(node.lvalue)
                        break
                else:
                    if node.right:
                        node = node.right
                    else:
                        y_pred.append(node.rvalue)
                        break
            else:
                y_pred.append(node.lvalue if row.iloc[node.feature_index] <= node.threshold else node.rvalue)
                
        return pd.Series(y_pred)

    def score(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        y.reset_index(drop=True, inplace=True)
        y_pred = self.predict(X)
        return float((y_pred == y).mean())
    
    def visualize_tree(self):
        """
        Visualize the decision tree.
        This method can be implemented using libraries like graphviz or matplotlib.
        """
        raise NotImplementedError("Visualization method is not implemented yet.")