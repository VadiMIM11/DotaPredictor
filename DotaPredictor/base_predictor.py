import os
import sys
from joblib import dump, load
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import DotaPredictor
import config

class BasePredictor:
    def __init__(self, model, filename):
        self.model = model
        self.filename = filename

    def load_model(self):
        path = os.path.join(config.MODELS_FOLDER, self.filename)
        try:
            self.model = load(path)
            print(f"Loaded {self.filename}", file=sys.stderr)
        except (OSError, ValueError) as e:
            print(f"Could not load {self.filename}: {e}", file=sys.stderr)

    def train(self, X_train, y_train):
        print(f"Training {self.filename}...", file=sys.stderr)
        self.model.fit(X_train, y_train)
        self.save_model()

    def save_model(self):
        if not os.path.exists(config.MODELS_FOLDER):
            os.makedirs(config.MODELS_FOLDER)
            print(f"Created folder: {config.MODELS_FOLDER}", file=sys.stderr)
        
        path = os.path.join(config.MODELS_FOLDER, self.filename)
        dump(self.model, path)
        print(f"Model saved in '{path}'", file=sys.stderr)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, cm

    def predict_by_match_id(self, match_id):
        match = DotaPredictor.get_match_by_id(match_id)
        if match is None:
             print(f"Match id {match_id} not found in data", file=sys.stderr)
             return None
        
        X = DotaPredictor.generate_feature_vector(match)
        feature_vector_2d = [feature_vector]
        scaler = joblib.load(os.path.join(config.MODELS_FOLDER, "scaler.joblib"))
        feature_vector = scaler.transform(feature_vector_2d)
        return self.predict_proba([X])[0]
