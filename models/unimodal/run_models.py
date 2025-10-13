import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_DIR = "data/technical"
REPORT_DIR = "reports/unimodal"
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_END = "2025-05-31"
TEST_START = "2025-06-01"
TEST_END = "2025-09-30"

models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

summary_rows = []


def evaluate_stock(file_path):
    ticker = os.path.basename(file_path).replace("_data.csv", "")
    print(f"\nProcessing {ticker}")

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    features = [c for c in df.columns if c not in ["Date", "Close", "Adj Close"]]
    target = "Close"

    train_df = df[df["Date"] <= TRAIN_END]
    test_df = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)]

    if train_df.empty or test_df.empty:
        print(f"Skipping {ticker} (insufficient data)")
        return

    X_train, y_train = train_df[features].copy(), train_df[target].copy()
    X_test, y_test = test_df[features].copy(), test_df[target].copy()

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train = X_train.ffill().fillna(0)
    X_test = X_test.ffill().fillna(0)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_rmse = float("inf")
    best_preds = None
    best_name = None
    best_mape = None

    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            rmse = root_mean_squared_error(y_test, preds)
            mape = mean_absolute_percentage_error(y_test, preds)

            print(f"  {name}: RMSE={rmse:.2f}, MAPE={mape:.2%}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_mape = mape
                best_preds = preds
                best_name = name
        except Exception as e:
            print(f"  Skipped {name} due to error: {e}")

    # Save plot
    plt.figure(figsize=(10, 5))
    plt.plot(test_df["Date"], y_test, label="Actual", color="black")
    plt.plot(test_df["Date"], best_preds, label=f"Predicted ({best_name})", linestyle="--")
    plt.title(f"{ticker}: Actual vs Predicted Closing Price (Jun–Sep 2025)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"{ticker}_prediction.png"))
    plt.close()

    summary_rows.append([ticker, best_name, best_rmse, best_mape])


if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_data.csv"))

    for file_path in csv_files:
        evaluate_stock(file_path)

    # Save summary
    summary_df = pd.DataFrame(summary_rows, columns=["Ticker", "BestModel", "RMSE", "MAPE"])
    summary_df.to_csv(os.path.join(REPORT_DIR, "summary.csv"), index=False)
    print("\nBenchmark Report Generated")
    print(summary_df)
