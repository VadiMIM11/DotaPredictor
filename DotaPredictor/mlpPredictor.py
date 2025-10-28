from tabnanny import verbose
import numpy as np
import json
from numpy.__config__ import CONFIG
from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

import DotaPredictor

import config

baggingModel = MLPClassifier(hidden_layer_sizes=(1024, 512, 256, 128), activation='relu', solver='adam', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=config.RANDOM_STATE)
#baggingModel = BaggingClassifier(mlpModel, n_estimators=10, random_state=config.RANDOM_STATE, verbose=2, n_jobs=10)

def train_mlp(X_train, y_train):
    global mlpModel
    global baggingModel

    #mlpModel.fit(X_train, y_train)
    baggingModel.fit(X_train, y_train)
    #print()

def predict_mlp(X):
    return baggingModel.predict(X)

def predict_proba_mlp(X):
    return baggingModel.predict_proba(X)

def evaluate_mlp(X_test, y_test):
    y_pred = baggingModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

def predict_by_match_id(match_id):
    #exit(1)
    # TODO
    match = DotaPredictor.get_match_by_id(match_id)
    X = DotaPredictor.generate_feature_vector(match)
    return predict_proba_mlp([X])[0]
    print(f"Match id {match_id} not found in data")
    return None
