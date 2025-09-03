import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config.tickers import TICKERS

from models.catboost_model import CatBoostModel
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.baseline_unimodal import BaseModel

MODEL_CLASSES = [
    CatBoostModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel
]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data", "technical")
reports_dir = os.path.join(project_root, "reports")
os.makedirs(reports_dir, exist_ok=True)

# Only include tickers for which CSV exists
available_tickers = [t for t in TICKERS if os.path.exists(os.path.join(data_dir, f"{t}_data.csv"))]
print(f"Tickers with CSVs: {available_tickers}")

# Prepare summary dataframe
summary = []

for ticker in available_tickers:
    print(f"\n=== Evaluating models for {ticker} ===")
    ticker_report_dir = os.path.join(reports_dir, ticker)
    os.makedirs(ticker_report_dir, exist_ok=True)

    best_acc = -1
    best_model_name = None

    for ModelClass in MODEL_CLASSES:
        model_name = ModelClass.__name__.replace('Model', '')
        print(f"\n--- Running {model_name} ---")
        model = ModelClass(ticker, data_dir, ticker_report_dir)
        model.build_model()

        # Load data
        df = model.load_data()

        # Create label
        df["Label"] = df["Close"].shift(-1) > df["Close"]
        df["Label"] = df["Label"].map({True: "Increase", False: "Decrease"})
        df.dropna(inplace=True)

        X = df.drop(columns=["Date", "Label"])
        y = df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        dates_test = df["Date"].iloc[y_test.index]
        prices_test = df["Close"].iloc[y_test.index]

        acc = model.evaluate_and_plot(df, y_test, y_pred, X, dates_test, prices_test)

        # Update best model
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name

    print(f"\n*** Best model for {ticker}: {best_model_name} with accuracy {best_acc:.4f} ***")
    summary.append({"Ticker": ticker, "Best Model": best_model_name, "Accuracy": best_acc})

# Save summary to CSV
summary_df = pd.DataFrame(summary)
summary_csv_path = os.path.join(reports_dir, "best_model_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSummary of best models saved to: {summary_csv_path}")

# Print overall average accuracy
BaseModel.print_average_accuracy()
