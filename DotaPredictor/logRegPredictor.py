import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import DotaPredictor

import config

model = LogisticRegression(max_iter=200)

def train(X_train, y_train):
    global model
    model.fit(X_train, y_train)
    #print()

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
    #exit(1)
    # TODO
    match = DotaPredictor.get_match_by_id(match_id)
    X = DotaPredictor.generate_feature_vector(match)
    return predict_proba([X])[0]
    print(f"Match id {match_id} not found in data")
    return None
