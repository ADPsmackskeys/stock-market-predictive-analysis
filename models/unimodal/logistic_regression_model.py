# logistic_regression_model.py
from sklearn.linear_model import LogisticRegression
from models.unimodal.base_model import BaseModel
import pandas as pd

class LogisticRegressionModel(BaseModel):
    def build_model(self):
        self.label_map = {"Decrease": 0, "Neutral": 1, "Increase": 2}
        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial"
        )

    def train(self, X_train, y_train):
        y_encoded = y_train.map(self.label_map)
        self.model.fit(X_train, y_encoded)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        inv_map = {v: k for k, v in self.label_map.items()}
        return pd.Series(y_pred).map(inv_map)
