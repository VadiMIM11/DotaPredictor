from math import e
import numpy as np
import pandas as pd
import json
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config

# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [0.01, 0.1, 1, 10, 100]
# }

rbfModel = svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.1)

def train_rbf(X_train, y_train):
    global rbfModel

    # cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # grid = GridSearchCV(
    #     estimator=rbfModel,
    #     param_grid=param_grid,
    #     cv=cv,
    #     scoring='accuracy',  
    #     n_jobs=6,           
    #     verbose=3
    # )
    # grid.fit(X_train, y_train)
    # rbfModel = grid.best_estimator_
    # params = grid.best_params_
    # print("Best rbf parameters:", params)

    rbfModel.fit(X_train, y_train)

def predict_rbf(X):
    return rbfModel.predict(X)

def predict_proba_rbf(X):
    return rbfModel.predict_proba(X)

def evaluate_rbf(X_test, y_test):
    y_pred = rbfModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

def predict_by_match_id(match_id):
    exit(1)
    # TODO
    try:
        with open(f"{config.DATA_FOLDER}/raw_train.json", "r") as f:
            matches = json.load(f)
    except IOError:
        print("Raw training data not found. Please run the data fetching script first.")
        return None