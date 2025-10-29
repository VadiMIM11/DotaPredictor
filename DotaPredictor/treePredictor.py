from tabnanny import verbose
import numpy as np
import json
from numpy.__config__ import CONFIG
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix

import DotaPredictor
import config

tree = DecisionTreeClassifier(max_depth=5, random_state=config.RANDOM_STATE)
#adaboost = AdaBoostClassifier(estimator=tree, n_estimators=100, random_state=config.RANDOM_STATE)
randomForest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_STATE)
model = CalibratedClassifierCV(randomForest, method='isotonic', cv=5)

def train(X_train, y_train):
    global model

    model.fit(X_train, y_train)
    #print()

def predict(X):
    return model.predict(X)

def predict_proba(X):
    return model.predict_proba(X)

# Evaluate model performance on a test set
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

