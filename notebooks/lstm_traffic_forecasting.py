import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# ----------------------------------------------------
# Load traffic dataset
# ----------------------------------------------------
df = pd.read_csv("data/processed/traffic_dataset.csv")

# Use only total traffic column
traffic = df["total_traffic"].values.reshape(-1, 1)

# ----------------------------------------------------
# Normalize traffic values
# ----------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
traffic_scaled = scaler.fit_transform(traffic)

# ----------------------------------------------------
# Create sequences for LSTM
# ----------------------------------------------------
sequence_length = 5

X = []
y = []

for i in range(len(traffic_scaled) - sequence_length):
    X.append(traffic_scaled[i:i + sequence_length])
    y.append(traffic_scaled[i + sequence_length])

X = np.array(X)
y = np.array(y)

# ----------------------------------------------------
# Build LSTM model
# ----------------------------------------------------
model = Sequential([
    Input(shape=(sequence_length, 1)),
    LSTM(16),
    Dense(8, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# ----------------------------------------------------
# Train model
# ----------------------------------------------------
history = model.fit(
    X,
    y,
    epochs=20,
    batch_size=4,
    verbose=1
)

# ----------------------------------------------------
# Predict traffic
# ----------------------------------------------------
predictions = model.predict(X)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y)

# ----------------------------------------------------
# Plot Actual vs Predicted traffic
# ----------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(actual, label="Actual Traffic", linewidth=2)
plt.plot(predictions, label="Predicted Traffic", linewidth=2)

plt.title("UrbanGrid AI - LSTM Traffic Forecasting")
plt.xlabel("Time")
plt.ylabel("Traffic Count")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()

# ----------------------------------------------------
# Save trained model
# ----------------------------------------------------
model.save("models/lstm_traffic_model.h5")

print("LSTM model saved to models/lstm_traffic_model.h5")

# ----------------------------------------------------
# Display next traffic prediction
# ----------------------------------------------------
last_sequence = traffic_scaled[-sequence_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)

next_prediction = model.predict(last_sequence)

next_prediction = scaler.inverse_transform(next_prediction)

print(f"Predicted traffic for next interval: {int(next_prediction[0][0])} vehicles")