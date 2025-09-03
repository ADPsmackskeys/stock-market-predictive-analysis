from sklearn.linear_model import LogisticRegression
from models.baseline_unimodal import BaseModel

class LogisticRegressionModel(BaseModel):
    def build_model(self):
        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto"
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
