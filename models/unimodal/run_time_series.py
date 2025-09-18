import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "technical")

# Collect sequences of 20 days in batches of 64
SEQ_LEN = 20
BATCH_SIZE = 64

# Combine OHLC data of each ticker in one csv file along with Ticker Name
# OHLC data in data/technical as files "ticker_name_data.csv"
all_dfs = []
for file in os.listdir(DATA_DIR):
    if file.endswith("_data.csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, file), parse_dates=["Date"])
        df['Ticker'] = file.split('_')[0]
        all_dfs.append(df)
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df = combined_df.sort_values(['Ticker', 'Date'])
combined_df['Label'] = (combined_df.groupby('Ticker')['Close'].shift(-1) > combined_df['Close']).astype(int)

# Encode Ticker Name as it is in String Format
le = LabelEncoder()
combined_df['Ticker_encoded'] = le.fit_transform(combined_df['Ticker'])
NUM_TICKERS = combined_df['Ticker_encoded'].nunique()

FEATURE_COLS = ['Open','High','Low','Close','Volume','SMA_20','SMA_50','EMA_20','EMA_50',
                'RSI_14','MACD','MACD_Signal','MACD_Hist','BB_upper','BB_middle','BB_lower',
                'ATR_14','STOCH_K','STOCH_D','OBV','CCI_20','Williams_%R','VWAP','CMF_20']
TARGET_COL = 'Label'

def sequence_generator(df, feature_cols, target_col, seq_len):
    df = df.sort_values(['Ticker', 'Date'])
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker]
        features = ticker_df[feature_cols].values
        tickers = ticker_df['Ticker_encoded'].values
        targets = ticker_df[target_col].values
        for i in range(len(ticker_df) - seq_len):   # Creating sliding Window
            Xf = features[i:i+seq_len]
            Xt = tickers[i+seq_len]
            y = targets[i+seq_len]
            yield {"price_input": Xf, "ticker_input": np.array([Xt])}, y

# -----------------------------
# Train/Test split (indices)
# -----------------------------
tickers = combined_df['Ticker'].unique()
train_tickers, test_tickers = train_test_split(tickers, test_size=0.1, random_state=42)

train_df = combined_df[combined_df['Ticker'].isin(train_tickers)]
test_df = combined_df[combined_df['Ticker'].isin(test_tickers)]

train_dataset = tf.data.Dataset.from_generator(
    lambda: sequence_generator(train_df, FEATURE_COLS, TARGET_COL, SEQ_LEN),
    output_signature=(
        {
            "price_input": tf.TensorSpec(shape=(SEQ_LEN, len(FEATURE_COLS)), dtype=tf.float32),
            "ticker_input": tf.TensorSpec(shape=(1,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: sequence_generator(test_df, FEATURE_COLS, TARGET_COL, SEQ_LEN),
    output_signature=(
        {
            "price_input": tf.TensorSpec(shape=(SEQ_LEN, len(FEATURE_COLS)), dtype=tf.float32),
            "ticker_input": tf.TensorSpec(shape=(1,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# Model
# -----------------------------
def build_lstm_cnn_model(seq_len, num_features, num_tickers, embed_dim=16):
    X_input = tf.keras.Input(shape=(seq_len, num_features), name='price_input')
    ticker_input = tf.keras.Input(shape=(1,), name='ticker_input')
    
    ticker_embed = tf.keras.layers.Embedding(input_dim=num_tickers, output_dim=embed_dim)(ticker_input)
    ticker_embed = tf.keras.layers.Reshape((embed_dim,))(ticker_embed)  # flatten extra dim
    ticker_embed_repeated = tf.keras.layers.RepeatVector(seq_len)(ticker_embed)

    x = tf.keras.layers.Concatenate()([X_input, ticker_embed_repeated])
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    
    model = tf.keras.Model(inputs=[X_input, ticker_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_lstm_cnn_model(SEQ_LEN, len(FEATURE_COLS), NUM_TICKERS)
model.summary()

# -----------------------------
# Train
# -----------------------------
EPOCHS = 10
history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS)

# -----------------------------
# Evaluate
# -----------------------------
y_true, y_pred = [], []
for X_batch, y_batch in test_dataset:
    preds = (model.predict(X_batch) > 0.5).astype(int).ravel()
    y_pred.extend(preds)
    y_true.extend(y_batch.numpy())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_true, y_pred, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
