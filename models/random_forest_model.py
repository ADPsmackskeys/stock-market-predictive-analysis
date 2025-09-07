from sklearn.ensemble import RandomForestClassifier
from models.baseline_unimodal import BaseModel

class RandomForestModel(BaseModel):
    def build_model(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
