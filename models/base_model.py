import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,mean_squared_error, mean_absolute_error, r2_score


class BaseModel:
    all_scores = []
    def __init__(self, ticker, task="classification"):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, "data", "technical")
        self.reports_dir = os.path.join(project_root, "reports", ticker)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.ticker = ticker
        self.task = task
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

        if hasattr(y_pred, "values"):
            y_pred_arr = np.ravel(y_pred.values) if isinstance(y_pred, pd.Series) or hasattr(y_pred, "values") else np.ravel(y_pred)
        else:
            y_pred_arr = np.ravel(y_pred)

        y_pred_arr = np.asarray(y_pred_arr)

        if np.issubdtype(y_pred_arr.dtype, np.integer):
            label_map = getattr(self, "label_map", {0: "Decrease", 1: "Increase"})
            inv_map = {v: k for k, v in label_map.items()}
            try:
                y_pred_arr = np.array([inv_map[int(i)] for i in y_pred_arr])
            except Exception:
                y_pred_arr = y_pred_arr.astype(str)

        y_test_len = len(y_test)
        y_pred_len = len(y_pred_arr)

        if y_pred_len == y_test_len:
            aligned_index = y_test.index
        elif y_pred_len < y_test_len:
            offset = y_test_len - y_pred_len
            aligned_index = y_test.index[offset:]
            y_test = y_test.iloc[offset:]
            dates_test = dates_test.iloc[offset:]
            prices_test = prices_test.iloc[offset:]
        else:
            y_pred_arr = y_pred_arr[-y_test_len:]
            aligned_index = y_test.index

        y_pred_series = pd.Series(y_pred_arr, index=aligned_index).astype(str)
        y_test_str = y_test.astype(str)

        acc = accuracy_score(y_test_str, y_pred_series)
        self.acc = acc
        BaseModel.all_scores.append(acc) if hasattr(BaseModel, "all_scores") else None

        print(f"\n{self.ticker} - Accuracy: {acc:.4f}")
        print(classification_report(y_test_str, y_pred_series, zero_division=0))

        plt.figure(figsize=(6, 4))
        df["Label"].value_counts().plot(kind="bar")
        plt.title(f"{self.ticker} - Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, f"{self.ticker}_class_distribution.png"))
        plt.close()

        classes = sorted(list(set(y_test_str) | set(y_pred_series)))
        cm = confusion_matrix(y_test_str, y_pred_series, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{self.ticker} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, f"{self.ticker}_confusion_matrix.png"))
        plt.close()

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

        plt.figure(figsize=(10, 6))
        plt.plot(dates_test, prices_test, label="Close Price", color="black")

        buy_mask = (y_pred_series == "Increase")
        sell_mask = (y_pred_series == "Decrease")

        buy_dates = dates_test.loc[buy_mask.index][buy_mask.values]
        buy_prices = prices_test.loc[buy_mask.index][buy_mask.values]
        sell_dates = dates_test.loc[sell_mask.index][sell_mask.values]
        sell_prices = prices_test.loc[sell_mask.index][sell_mask.values]

        plt.scatter(buy_dates, buy_prices, label="Buy Signal", marker="^", color="green")
        plt.scatter(sell_dates, sell_prices, label="Sell Signal", marker="v", color="red")
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
    def print_average_scores(cls):
        if cls.all_scores:
            avg_acc = sum(cls.all_scores) / len(cls.all_scores)
            print(f"\nAverage Accuracy across all tickers: {avg_acc:.4f}")
        else:
            print("No scores recorded yet.")
