import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- Split data ---
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, shuffle=True
)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# --- Build LSTM model ---
model = Sequential([
    LSTM(64, input_shape=X_train.shape[1:], return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Training ---
batch_size = 2048  # large but fits in modern GPUs/CPU memory
epochs = 10  # start small

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights,
    shuffle=True
)

# --- Evaluate ---
val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=batch_size)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
