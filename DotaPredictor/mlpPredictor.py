from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import config
from base_predictor import BasePredictor

class MlpPredictor(BasePredictor):
    def __init__(self):
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), 
            activation='relu', 
            solver='adam',
            early_stopping=True,
            max_iter=500, 
            random_state=config.RANDOM_STATE
        )

        model = CalibratedClassifierCV(mlp, method='isotonic', cv=5)
        super().__init__(model, 'mlp_model.joblib')