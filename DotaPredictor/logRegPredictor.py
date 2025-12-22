from sklearn.linear_model import LogisticRegression
from base_predictor import BasePredictor

class LogRegPredictor(BasePredictor):
    def __init__(self):
        model = LogisticRegression(max_iter=200)
        super().__init__(model, 'logreg_model.joblib')
