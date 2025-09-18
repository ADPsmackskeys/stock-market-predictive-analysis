# catboost_model.py
from catboost import CatBoostClassifier
from models.unimodal.base_model import BaseModel
import pandas as pd

class CatBoostModel(BaseModel):
    def build_model(self):
        self.label_map = {"Decrease": 0, "Neutral": 1, "Increase": 2}
        self.model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False
        )

    def train(self, X_train, y_train):
        y_encoded = y_train.map(self.label_map)
        self.model.fit(X_train, y_encoded)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        inv_map = {v: k for k, v in self.label_map.items()}
        return pd.Series(y_pred.ravel()).map(inv_map)
