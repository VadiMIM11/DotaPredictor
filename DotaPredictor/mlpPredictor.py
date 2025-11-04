from tabnanny import verbose
import numpy as np
import json
from numpy.__config__ import CONFIG
from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample


import DotaPredictor

import config

mlp_model = MLPClassifier(hidden_layer_sizes=(1024, 512, 256, 128), activation='relu', solver='adam', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=config.RANDOM_STATE)
model = CalibratedClassifierCV(mlp_model, method='isotonic', cv=5)

def train(X_train, y_train):
    global mlpModel
    global model

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
