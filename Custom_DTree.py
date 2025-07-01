import pandas as pd
from collections import Counter
import numpy as np
from Classifier import Classifier

class DecisionTreeClassifier(Classifier):
    def __init__(self, max_depth:int =None, max_features:int = None):
        self.max_depth = max_depth
        self.root = None
        self.max_features = max_features
        
    def __check_GI(self, X, y):
        """
        This method should implement the logic to find the best feature and threshold to split on based on Gini impurity.
        """
        selected_features = np.random.choice(np.arange(X.shape[1]), size=self.max_features, replace=False) if self.max_features else range(X.shape[1])
        
        col_lbl_cts = [np.unique(X[:, i]) for i in selected_features]
        thresholds = [(a[:-1]+a[1:])/2 if a.size>1 else [] for a in col_lbl_cts]
        GIs = {}
        for i in range(len(selected_features)):
            for j in thresholds[i]:
                mask = X[:, selected_features[i]] <= j
                X_l, X_r, y_l, y_r = X[mask], X[~mask], y[mask], y[~mask]
                gi_l = self.gini_impurity(y_l)
                gi_r = self.gini_impurity(y_r)
                
                w1, w2 = len(X_l), len(X_r)
                gi_avg = self.weighted_average(gi_l, gi_r, w1, w2)
                s = gi_avg[0] + gi_avg[1]
                if s not in GIs:
                    GIs[s] =[(selected_features[i], j, gi_l, gi_r, s)]
                else:
                    GIs[s].append((selected_features[i], j, gi_l, gi_r, s))
                #This gi_avg sum is the Gini impurity for the split, which we want to minimize. 
        l = GIs[min(GIs.keys())] if GIs else None
        return l[np.random.choice(len(l), size=1)[0]] if l else (None, None, None, None, None)
    
    def __build_tree(self, X, y, root, gi=None, depth=1):
        """
        Recursively build the decision tree.
        :param root: The current node in the tree.
        :param depth: Current depth of the tree.
        """
        fi, thr, gi_l, gi_r = None, None, None, None
        if self.max_depth==None or (self.max_depth >= depth):
            fi, thr, gi_l, gi_r, GI = self.__check_GI(X, y)
            root.feature_index = fi
            root.threshold = thr
        
        if GI is None:
            return Counter(y).most_common(1)[0][0]
        
        lc_l, lc_r = Counter(y[X[:, fi] <= thr]), Counter(y[X[:, fi] > thr])
        
        if (self.max_depth and self.max_depth == depth)or(gi and (gi-GI)<0.01):
            root.lvalue = lc_l.most_common(1)[0][0]
            root.rvalue = lc_r.most_common(1)[0][0]
        else:
            if gi_l != 0:
                root.left = self.Tree()
                lrt = self.__build_tree(X[X[:, fi] <= thr], y[X[:, fi] <= thr], root.left, GI, depth + 1)
                
                if lrt is not None:
                    root.left = None
                    root.lvalue = lrt
                    
            else:
                root.lvalue = lc_l.most_common(1)[0][0]
            if gi_r != 0:
                root.right = self.Tree()
                rrt = self.__build_tree(X[X[:, fi] > thr], y[X[:, fi] > thr], root.right, GI, depth + 1)
                
                if rrt is not None:
                    root.right = None
                    root.rvalue = rrt
                    
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
        rt = self.__build_tree(X.to_numpy(), y.to_numpy(), root=self.root)
        if rt:
            self.root.lvalue = rt
            self.root.rvalue = rt
        
    def predict(self, X):
        y_pred = []
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        for _, row in X.iterrows():
            node = self.root
            if node.threshold is None:
                y_pred.append(node.lvalue)
                continue
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