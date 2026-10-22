"""
Trees, Ensembles and SVMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_boston
import os
import tempfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

ISLP_BASE = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/"


def load_islp(name):
    # TODO: download <name>.csv once into tempfile.gettempdir(); return pd.read_csv(path)
    filename = f"{name}.csv"
    path = os.path.join(tempfile.gettempdir(), filename)

    if not os.path.exists(path):
        url = ISLP_BASE + filename
        urllib.request.urlretrieve(url, path)
    
    return pd.read_csv(path)
    

def load_boston():
    # TODO: load_islp('Boston')
    return load_islp('Boston')
     
    

def split_xy(df, target):
    # TODO: (features without target, target series)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y



def train_test(X, y, test_size=0.25, random_state=0):
    # TODO: train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Step 2 - regression_tree
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

def fit_tree(X, y, max_depth=None, ccp_alpha=0.0):
    # TODO: DecisionTreeRegressor(max_depth, ccp_alpha, random_state=0).fit
    model = DecisionTreeRegressor(max_depth=max_depth, ccp_alpha=ccp_alpha, random_state=0)
    return model.fit(X, y)

def tree_summary(tree, feature_names):
    # TODO: n_leaves, depth, root_feature, root_threshold (3 decimals)
    n_leaves = tree.get_n_leaves()
    depth = tree.get_depth()
    root_index = tree.tree_.feature[0]
    if root_index == -2:
        root_feature = None
        root_threshold = None
    else:
        root_feature = feature_names[root_index]
        root_threshold = round(tree.tree_.threshold[0], 3)
    return {
    "n_leaves": n_leaves,
    "depth": depth,
    "root_feature": root_feature,
    "root_threshold": root_threshold
}
    

def rmse(model, X, y):
    # TODO: sqrt(MSE) rounded to 3 decimals
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    return round(np.sqrt(mse),3)

