import numpy as np
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.stats import norm
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR = os.path.join(BASE_DIR, "Sequence")
MODEL_DIR = os.path.join(BASE_DIR, "Model_and_Scaler")

SEQ_FILE = os.path.join(SEQ_DIR, "sequence.txt")
MODEL_FILE = os.path.join(MODEL_DIR, 'sequence_model_7.h5')
X_SCALER_FILE = os.path.join(MODEL_DIR, "x_scaler.pkl")
Y_SCALER_FILE = os.path.join(MODEL_DIR, "y_scaler.pkl")


def train_and_save_model():
    """Train sequence prediction model and save artifacts"""
    # Load data from file
    sequence = []
    with open(SEQ_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sequence.append(int(line))

    print(f"Samples: {len(sequence)}")

    # Prepare training data
    X_raw = np.array(sequence[:-1]).reshape(-1, 1)
    y_raw = np.array(sequence[1:]).reshape(-1, 1)

    # Normalize data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(X_raw)
    y = y_scaler.fit_transform(y_raw)

    # Create neural network model
    model = Sequential([
        Input(shape=(1,)),  # Input layer
        Dense(32, activation='relu'),  # Hidden layer 1
        Dense(32, activation='relu'),  # Hidden layer 2
        Dense(1)  # Output layer (regression)
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  # loss=Huber(delta=0.02),
                  metrics=['mae'])

    # Train model
    split = int(len(X) * 0.9)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    early = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    rlr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        shuffle=False,  # important
        callbacks=[early,rlr],
        verbose=1
    )
    print(f"\nMAE: {history.history['mae'][-1]:.4f}")

    # Save model and scalers for later use
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_FILE)
    joblib.dump(x_scaler, X_SCALER_FILE)
    joblib.dump(y_scaler, Y_SCALER_FILE)

    preds_scaled = model.predict(X, verbose=0)
    preds = y_scaler.inverse_transform(preds_scaled)
    y_original = y_scaler.inverse_transform(y)

    mae_original = np.mean(np.abs(preds - y_original))
    print(f"MAE (original scale): {mae_original:.4f}")
    return 0




def main():
    if os.path.exists(MODEL_FILE) and \
       os.path.exists(X_SCALER_FILE) and \
       os.path.exists(Y_SCALER_FILE):
        print("Found existing model and scalers")
    else:
        print("Model or scalers not found, training new model...")
        train_and_save_model()



if __name__ == "__main__":
    main()