import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
# 已废弃
# load the data
file_path = r"C:\Users\LENOVO\Desktop\project\Sequence - complete.txt"
sequence = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            sequence.append(int(line))

print(f"Samples: {len(sequence)}")

X_raw = np.array(sequence[:-1]).reshape(-1, 1)
y_raw = np.array(sequence[1:]).reshape(-1, 1)

# normalize
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)

# get the model
model = Sequential([
    Input(shape=(1,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# train the model
history = model.fit(X, y, epochs=100, batch_size=8, verbose=1)
print(f"\nMAE: {history.history['mae'][-1]:.4f}")

def predict_next(val):
    val_scaled = x_scaler.transform(np.array([[val]]))
    pred_scaled = model.predict(val_scaled, verbose=0)
    pred_original = y_scaler.inverse_transform(pred_scaled)[0][0]
    pred = int(round(pred_original))
    print(f"input {val} → predict: {pred}")

# for val in [3155, 3143, 3, 3127, 4669, 2501]:
#     predict_next(val)

