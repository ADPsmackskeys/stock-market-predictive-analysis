import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config.tickers import TICKERS

from models.unimodal.catboost_model import CatBoostModel
from models.unimodal.logistic_regression_model import LogisticRegressionModel
from models.unimodal.random_forest_model import RandomForestModel
from models.unimodal.xgboost_model import XGBoostModel
from models.unimodal.base_model import BaseModel

MODEL_CLASSES = [
    CatBoostModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel
]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data", "technical")
reports_dir = os.path.join(project_root, "reports", "unimodal")
os.makedirs(reports_dir, exist_ok=True)

staged_file = os.path.join(reports_dir, "staged_run.txt")
processed_tickers = set()
if os.path.exists(staged_file):
    with open(staged_file, "r") as f:
        processed_tickers = set(line.strip() for line in f if line.strip())

available_tickers = [t for t in TICKERS if os.path.exists(os.path.join(data_dir, f"{t}_data.csv"))]
available_tickers = [t for t in available_tickers if t not in processed_tickers]
print(f"Tickers to process: {available_tickers}")

overall_summary = []

for ticker in available_tickers:
    print(f"\n=== Evaluating models for {ticker} ===")

    ticker_report_dir = os.path.join(reports_dir, ticker)
    os.makedirs(ticker_report_dir, exist_ok=True)

    best_acc = -1
    best_model_name = None

    for ModelClass in MODEL_CLASSES:
        model_name = ModelClass.__name__.replace("Model", "")
        print(f"\n--- Running {model_name} ---")
        model = ModelClass(ticker)
        model.build_model()

        # Load data
        df = model.load_data()
        df["Label"] = df["Close"].shift(-1) > df["Close"]
        df["Label"] = df["Label"].map({True: "Increase", False: "Decrease"})
        df.dropna(inplace=True)

        # Skip if not enough data
        if df.empty or len(df) < 10:
            print(f"Skipping {ticker}: insufficient data")
            break

        X = df.drop(columns=["Date", "Label"])
        y = df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        dates_test = df["Date"].iloc[y_test.index]
        prices_test = df["Close"].iloc[y_test.index]

        acc = model.evaluate_and_plot(df, y_test, y_pred, X, dates_test, prices_test)

        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name

    if best_model_name:
        summary_df = pd.DataFrame([{
            "Ticker": ticker,
            "Best Model": best_model_name,
            "Accuracy": best_acc
        }])
        summary_csv_path = os.path.join(ticker_report_dir, f"{ticker}_best_model_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary for {ticker} at: {summary_csv_path}")

        overall_summary.append({"Ticker": ticker, "Best Model": best_model_name, "Accuracy": best_acc})

        # Mark ticker as processed
        with open(staged_file, "a") as f:
            f.write(f"{ticker}\n")

# Save overall summary
overall_summary_df = pd.DataFrame(overall_summary)
overall_summary_csv = os.path.join(reports_dir, "overall_best_model_summary.csv")
overall_summary_df.to_csv(overall_summary_csv, index=False)
print(f"\nOverall best model summary saved to: {overall_summary_csv}")

BaseModel.print_average_scores()
