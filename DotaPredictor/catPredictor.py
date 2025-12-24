from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
import config
from base_predictor import BasePredictor

class CatPredictor(BasePredictor):
    def __init__(self):
        # CatBoost is a Gradient Boosting library that handles numerical features very well.
        # We enable 'auto_class_weights' to fix the Radiant/Dire bias.
        model  = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            loss_function='Logloss',
            auto_class_weights='Balanced', # Automatically penalize mistakes on the minority class (Dire wins)
            verbose=0,                   # Log progress every _ iterations
            random_seed=config.RANDOM_STATE,
            allow_writing_files=False      # Prevents creating 'catboost_info' folder
        )
        model = CalibratedClassifierCV(model, method='isotonic', cv=5)
        super().__init__(model, 'cat_model.joblib')