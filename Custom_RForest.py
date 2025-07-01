
from Classifier import Classifier
from Custom_DTree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from collections import Counter

class RandomForestClassifier(Classifier):
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features #if None, it will use sqrt(n_features)
        self.Dtrees = []
    
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(X.index, size=len(X), replace=True)
            X_bootstrap = X.loc[bootstrap_indices]
            y_bootstrap = y.loc[bootstrap_indices]
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features if self.max_features else int(np.sqrt(X.shape[1])))
            tree.fit(X_bootstrap, y_bootstrap)
            self.Dtrees.append(tree)
        
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        predictions = np.array([tree.predict(X) for tree in self.Dtrees])
        return pd.Series([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(predictions.shape[1])], index=X.index)