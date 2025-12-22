import os
import numpy as np
import json
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from joblib import dump
from joblib import load

import config
import DotaPredictor

model = svm.SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE,  C=1, gamma=0.1)
#model = CalibratedClassifierCV(svcModel, method='isotonic', cv=5)

def load_model(path):
    global model
    model = load(path)

def find_best_params(X_train, y_train):
    global model
    param_grid = {
        'C': [0.25, 0.5, 1, 2, 3],
        'gamma': [0.05, 0.1, 0.2, 0.3, 0.5]
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',  
        n_jobs=6,           
        verbose=3
    )
    grid.fit(X_train, y_train)

    params = grid.best_params_
    model = grid.best_estimator_
    print("Best rbf parameters:", params)

    model_path = os.path.join(config.MODELS_FOLDER, 'rbf_model.joblib')
    if not os.path.exists(config.MODELS_FOLDER):
        os.makedirs(config.MODELS_FOLDER)
        print(f"Created folder: {config.MODELS_FOLDER}")
    dump(model, model_path)
    print(f"Model saved in '{model_path}'")

    return params

def train(X_train, y_train, C=1, gamma=0.1):
    global model

    model = svm.SVC(kernel='rbf', probability=True, C=C, gamma=gamma, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)

    if not os.path.exists(config.MODELS_FOLDER):
        os.makedirs(config.MODELS_FOLDER)
        print(f"Created folder: {config.MODELS_FOLDER}")
    model_path = os.path.join(config.MODELS_FOLDER, 'rbf_model.joblib')

    dump(model, model_path)
    print(f"Model saved in '{model_path}'")


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
