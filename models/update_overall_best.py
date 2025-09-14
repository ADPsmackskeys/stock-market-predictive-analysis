import pandas as pd
import os

STAGED_RUN_FILE = 'staged_run.csv'
SUMMARY_FILE = 'overall_best_model_summary.csv'

if not os.path.exists(STAGED_RUN_FILE):
    raise FileNotFoundError("Run staged_run.csv first!")

staged_df = pd.read_csv(STAGED_RUN_FILE)
summary = staged_df.groupby('ticker').apply(lambda x: x.loc[x[['LSTM_MSE','CNN_MSE']].min(axis=1).idxmin()])

# Keep only required columns
summary = summary[['ticker','LSTM_MSE','CNN_MSE','best_model']]

# Save/update overall summary
if os.path.exists(SUMMARY_FILE):
    summary.to_csv(SUMMARY_FILE, mode='a', index=False, header=False)
else:
    summary.to_csv(SUMMARY_FILE, index=False)

print("Overall best model summary updated in", SUMMARY_FILE)
