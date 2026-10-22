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

