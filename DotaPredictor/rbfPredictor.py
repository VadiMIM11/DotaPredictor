import numpy as np
import json
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
import DotaPredictor


model = svm.SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE,  C=1, gamma=0.1)
def find_best_params():
    param_grid = {
        'C': [0.25, 0.5, 1, 2, 3],
        'gamma': [0.05, 0.1, 0.2, 0.3, 0.5]
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=rbfModel,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',  
        n_jobs=6,           
        verbose=3
    )
    grid.fit(X_train, y_train)

    params = grid.best_params_
    rbfModel = grid.best_estimator_
    print("Best rbf parameters:", params)
    return params

def train(X_train, y_train, C=1, gamma=0.1):
    global model

    model = svm.SVC(kernel='rbf', probability=True, C=C, gamma=gamma, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)


def predict(X):
    return model.predict(X)

def predict_proba(X):
    return model.predict_proba(X)

def evaluate(X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

def predict_by_match_id(match_id):
    match = DotaPredictor.get_match_by_id(match_id)
    X = DotaPredictor.generate_feature_vector(match)
    return predict_proba([X])[0]
    print(f"Match id {match_id} not found in data")
    return None
