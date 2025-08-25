import numpy as np
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.stats import norm



# ====================
# Part 1: Original Model Training
# ====================

def train_and_save_model():
    """Train sequence prediction model and save artifacts"""
    # Load data from file
    file_path = r"C:\Users\LENOVO\Desktop\project\sequence.txt"
    sequence = []
    with open(file_path, 'r') as f:
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
        Dense(64, activation='relu'),  # Hidden layer 1
        Dense(64, activation='relu'),  # Hidden layer 2
        Dense(1)  # Output layer (regression)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    split = int(len(X) * 0.9)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        shuffle=False,  # important
        verbose=1
    )
    print(f"\nMAE: {history.history['mae'][-1]:.4f}")

    # Save model and scalers for later use
    model.save('sequence_model.h5')
    joblib.dump(x_scaler, 'x_scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')
    print("Model and scalers saved successfully")

    preds_scaled = model.predict(X, verbose=0)
    preds = y_scaler.inverse_transform(preds_scaled)
    y_original = y_scaler.inverse_transform(y)

    mae_original = np.mean(np.abs(preds - y_original))
    print(f"MAE (original scale): {mae_original:.4f}")

    return model, x_scaler, y_scaler, sequence


# ====================
# Part 2: DTMC Conversion (Gaussian-based approach)
# ====================

def calculate_error_distribution(model, x_scaler, y_scaler, sequence):
    """Calculate prediction error statistics"""
    actuals = []
    preds = []

    # Iterate through sequence to collect predictions
    for i in range(len(sequence) - 1):
        # Prepare input
        val = sequence[i]
        val_scaled = x_scaler.transform([[val]])

        # Get prediction
        pred_scaled = model.predict(val_scaled, verbose=0)
        pred = y_scaler.inverse_transform(pred_scaled)[0][0]

        # Record actual next value and prediction
        actuals.append(sequence[i + 1])
        preds.append(pred)

    # Calculate residuals (prediction errors)
    residuals = np.array(actuals) - np.array(preds)

    # Return mean and standard deviation of errors
    return residuals.mean(), residuals.std()


def build_dtmc(model, x_scaler, y_scaler, sequence):
    """Build DTMC transition probability matrix"""
    # Get unique states and create mapping
    states = sorted(set(sequence))
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Calculate prediction error statistics
    residual_mean, residual_std = calculate_error_distribution(
        model, x_scaler, y_scaler, sequence
    )
    print(f"Residual distribution: mean={residual_mean:.4f}, std={residual_std:.4f}")

    # Initialize transition matrix
    transition_matrix = np.zeros((n_states, n_states))

    # Build transition probabilities for each state
    for i, state in enumerate(states):
        # Get neural network prediction (continuous value)
        state_scaled = x_scaler.transform([[state]])
        pred_scaled = model.predict(state_scaled, verbose=0)
        pred_cont = y_scaler.inverse_transform(pred_scaled)[0][0]

        # Calculate probabilities for all possible next states
        probs = []
        for target_state in states:
            # Use Gaussian distribution centered at prediction
            prob = norm.pdf(
                target_state,
                loc=pred_cont + residual_mean,  # Adjust by residual mean
                scale=residual_std  # Use residual std as uncertainty measure
            )
            probs.append(prob)

        # Normalize probabilities to sum to 1
        probs = np.array(probs)
        if probs.sum() == 0:
            # Handle zero-sum case (uniform distribution)
            probs = np.ones_like(probs) / n_states
        else:
            probs = probs / probs.sum()

        transition_matrix[i] = probs

    return transition_matrix, states


def analyze_dtmc(transition_matrix, states):
    """Analyze properties of the DTMC"""
    # Verify transition matrix validity (rows sum to 1)
    row_sums = np.sum(transition_matrix, axis=1)
    print(f"Row sum validation: min={min(row_sums):.4f}, max={max(row_sums):.4f}")

    # Find most probable transitions
    print("\nMost probable transitions:")
    for i, state in enumerate(states[:5]):  # Show first 5 states
        j = np.argmax(transition_matrix[i])
        print(f"State {state} → {states[j]}: {transition_matrix[i][j]:.4f}")

    # Compute stationary distribution (if possible)
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary_dist = np.real(eigenvectors[:, stationary_idx])
        stationary_dist /= stationary_dist.sum()  # Normalize

        # Find most probable state
        top_idx = np.argmax(stationary_dist)
        print(f"\nStationary distribution: Most probable state={states[top_idx]}, prob={stationary_dist[top_idx]:.4f}")
    except Exception as e:
        print(f"\nStationary distribution calculation failed: {str(e)}")


# ====================
# Part 3: Main Program
# ====================

def main():
    # Try to load existing model or train new one
    try:
        # Load pre-trained model and scalers
        model = load_model('sequence_model.h5')
        x_scaler = joblib.load('x_scaler.pkl')
        y_scaler = joblib.load('y_scaler.pkl')

        # Reload sequence data
        file_path = r"C:\Users\LENOVO\Desktop\project\Sequence - complete.txt"
        sequence = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    sequence.append(int(line))
        print("Loaded existing model and data")
    except:
        print("Training new model...")
        model, x_scaler, y_scaler, sequence = train_and_save_model()

    # Convert neural network to DTMC
    print("\nBuilding DTMC transition matrix...")
    transition_matrix, states = build_dtmc(model, x_scaler, y_scaler, sequence)

    # Save DTMC components
    np.save("dtmc_transition_matrix.npy", transition_matrix)
    with open("state_space.txt", "w") as f:
        f.write(",".join(map(str, states)))
    print(f"DTMC saved: {len(states)} states")

    # Analyze DTMC properties
    print("\nAnalyzing DTMC properties...")
    analyze_dtmc(transition_matrix, states)

    # Prediction function (original)
    def predict_next(val):
        """Predict next value in sequence"""
        val_scaled = x_scaler.transform(np.array([[val]]))
        pred_scaled = model.predict(val_scaled, verbose=0)
        pred_original = y_scaler.inverse_transform(pred_scaled)[0][0]
        pred = int(round(pred_original))
        print(f"input {val} → predict: {pred}")

    # # Test predictions
    # print("\nTesting predictions:")
    # test_values = [3155, 3143, 3, 3127, 4669, 2501]
    # for val in test_values:
    #     predict_next(val)


if __name__ == "__main__":
    main()