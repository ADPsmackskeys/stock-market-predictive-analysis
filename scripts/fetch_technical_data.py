import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import pickle

seq_len = 30
feature_cols = ['Open','High','Low','Close','Volume','SMA_20','SMA_50','EMA_20','EMA_50',
                'RSI_14','MACD','MACD_Signal','MACD_Hist','BB_upper','BB_middle','BB_lower',
                'ATR_14','STOCH_K','STOCH_D','OBV','CCI_20','Williams_%R','VWAP','CMF_20']

all_sequences = []
all_labels = []

csv_files = glob.glob("D:/VIT/Project/stock-market-predictive-analysis/data/technical/*.csv")
for file in csv_files:
    df = pd.read_csv(file)
    df = df.sort_values('Date')
    df[feature_cols] = df[feature_cols].fillna(method='bfill').fillna(method='ffill')

    if len(df) < seq_len + 1:
        continue

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df[:-1]

    for i in range(len(df) - seq_len + 1):
        seq = df[feature_cols].iloc[i:i+seq_len].values
        label = df['Target'].iloc[i+seq_len-1]
        all_sequences.append(seq)
        all_labels.append(label)

# Convert to NumPy arrays
X = np.array(all_sequences, dtype=np.float32)
y = np.array(all_labels, dtype=np.int8)

# Normalize features
n_samples, n_timesteps, n_features = X.shape
scaler = StandardScaler()
X_reshaped = X.reshape(-1, n_features)
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

# Compute class weights
classes = np.unique(y)
weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weights = dict(zip(classes, weights))

# Optional: save to disk to avoid memory issues
with open("X_scaled.pkl", "wb") as f:
    pickle.dump(X_scaled, f)
with open("y.pkl", "wb") as f:
    pickle.dump(y, f)
with open("class_weights.pkl", "wb") as f:
    pickle.dump(class_weights, f)

print("X shape:", X_scaled.shape)
print("y shape:", y.shape)
print("Class weights:", class_weights)
