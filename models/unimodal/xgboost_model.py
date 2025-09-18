# xgboost_model.py
from xgboost import XGBClassifier
from models.unimodal.base_model import BaseModel
import pandas as pd

class XGBoostModel(BaseModel):
    def build_model(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            eval_metric='mlogloss',
            use_label_encoder=False
        )

    def train(self, X_train, y_train):
        # Dynamically map labels present in y_train
        unique_labels = sorted(y_train.unique())
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = y_train.map(self.label_map)
        self.model.fit(X_train, y_encoded)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        # Decode predicted integers back to original labels
        inv_map = {v: k for k, v in self.label_map.items()}
        return pd.Series(y_pred).map(inv_map)
