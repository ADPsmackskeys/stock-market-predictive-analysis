from xgboost import XGBClassifier
from models.baseline_unimodal import BaseModel
import pandas as pd

class XGBoostModel(BaseModel):
    def build_model(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def train(self, X_train, y_train):
        # Encode labels
        self.label_map = {"Decrease": 0, "Increase": 1}
        y_encoded = y_train.map(self.label_map)
        self.model.fit(X_train, y_encoded)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        # Decode back to original labels
        inv_map = {v: k for k, v in self.label_map.items()}
        return pd.Series(y_pred).map(inv_map)
