from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import config
from base_predictor import BasePredictor

class TreePredictor(BasePredictor):
    def __init__(self):
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        )
        model = CalibratedClassifierCV(rf, method='isotonic', cv=5)
        super().__init__(model, 'tree_model.joblib')