from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import config
from base_predictor import BasePredictor

class RbfPredictor(BasePredictor):
    def __init__(self):
        # Default initialization
        model = svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.1, random_state=config.RANDOM_STATE, class_weight='balanced')
        super().__init__(model, 'rbf_model.joblib')

    # OVERRIDE the train method to accept C and gamma arguments
    def train(self, X_train, y_train, C=1, gamma=0.1):
        # Re-initialize model with new params
        self.model = svm.SVC(
            kernel='rbf', probability=True, C=C, gamma=gamma, random_state=config.RANDOM_STATE
        )
        # Call the parent train to handle fitting and saving
        super().train(X_train, y_train)

    def find_best_params(self, X_train, y_train):
        param_grid = {
            'C': [0.25, 0.5, 1, 2, 3],
            'gamma': [0.05, 0.1, 0.2, 0.3, 0.5]
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',  
            n_jobs=6,           
            verbose=3
        )
        grid.fit(X_train, y_train)
        
        self.model = grid.best_estimator_
        print("Best rbf parameters:", grid.best_params_)
        
        self.save_model()
        return grid.best_params_