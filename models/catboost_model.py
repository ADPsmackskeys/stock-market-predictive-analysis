from catboost import CatBoostClassifier
from models.baseline_unimodal import BaseModel

class CatBoostModel(BaseModel):
    def build_model(self):
        self.model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            train_dir=None 
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
