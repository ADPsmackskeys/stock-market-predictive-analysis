import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

class BaseModel:
    all_accuracies = []

    def __init__(self, ticker, data_dir, reports_dir):
        self.ticker = ticker
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self.model = None
        self.acc = None

    def load_data(self):
        file_path = os.path.join(self.data_dir, f"{self.ticker}_data.csv")
        df = pd.read_csv(file_path, parse_dates=["Date"])
        return df
    def evaluate_and_plot(self, df, y_test, y_pred, X, dates_test, prices_test):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

        # --- Flatten and convert predictions to strings ---
        if hasattr(y_pred, "values"):
            y_pred_array = y_pred.values.ravel()
        else:
            y_pred_array = np.ravel(y_pred)

        # Convert integer predictions to string labels if necessary
        if np.issubdtype(y_pred_array.dtype, np.integer):
            label_mapping = getattr(self, "label_map", {0: "Decrease", 1: "Increase"})
            inv_map = {v: k for k, v in label_mapping.items()}
            y_pred_array = np.array([inv_map[i] for i in y_pred_array])

        # Align with y_test index and ensure string type
        y_pred_array = pd.Series(y_pred_array, index=y_test.index).astype(str)
        y_test_str = y_test.astype(str)

        # --- Accuracy & Classification Report ---
        acc = accuracy_score(y_test_str, y_pred_array)
        self.acc = acc
        BaseModel.all_accuracies.append(acc)

        print(f"\n{self.ticker} - Accuracy: {acc:.4f}")
        print(classification_report(y_test_str, y_pred_array, zero_division=0))

        # --- Class Distribution ---
        plt.figure(figsize=(6, 4))
        df["Label"].value_counts().plot(kind="bar", color=["red", "blue", "green"])
        plt.title(f"{self.ticker} - Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, f"{self.ticker}_class_distribution.png"))
        plt.close()

        # --- Confusion Matrix ---
        classes = sorted(list(set(y_test_str) | set(y_pred_array)))  # dynamic classes
        cm = confusion_matrix(y_test_str, y_pred_array, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{self.ticker} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, f"{self.ticker}_confusion_matrix.png"))
        plt.close()

        # --- Feature Importances ---
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            indices = importances.argsort()[-15:]
            plt.figure(figsize=(8, 6))
            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
            plt.title(f"{self.ticker} - Top 15 Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(self.reports_dir, f"{self.ticker}_feature_importances.png"))
            plt.close()

        # --- Price with Buy/Sell Predictions ---
        plt.figure(figsize=(10, 6))
        plt.plot(dates_test, prices_test, label="Close Price", color="black")

        buy_signals = dates_test[y_pred_array == "Increase"]
        buy_prices = prices_test[y_pred_array == "Increase"]
        sell_signals = dates_test[y_pred_array == "Decrease"]
        sell_prices = prices_test[y_pred_array == "Decrease"]

        plt.scatter(buy_signals, buy_prices, label="Buy Signal", marker="^", color="green")
        plt.scatter(sell_signals, sell_prices, label="Sell Signal", marker="v", color="red")
        plt.title(f"{self.ticker} - Price with Buy/Sell Predictions")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, f"{self.ticker}_price_signals.png"))
        plt.close()

        return acc

    
    def build_model(self):
        raise NotImplementedError

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    @classmethod
    def print_average_accuracy(cls):
        if cls.all_accuracies:
            avg_acc = sum(cls.all_accuracies) / len(cls.all_accuracies)
            print(f"\n=== Average Accuracy across all tickers: {avg_acc:.4f} ===")
        else:
            print("No accuracies recorded yet.")
