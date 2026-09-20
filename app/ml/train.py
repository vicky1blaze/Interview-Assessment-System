"""
train.py - Neural Network Training Pipeline.

Orchestrates:
  1. Load 18-feature dataset from CSV (synthetic or real)
  2. 70 / 15 / 15 train / validation / test split
  3. Adapt Normalization layer on training features only
  4. Normalise targets to [0, 1] for sigmoid output training:
         y_norm = (y - SCORE_MIN) / SCORE_RANGE
  5. Build the Keras MLP (sigmoid output)
  6. Train with MSE loss over epochs (gradient descent / backpropagation)
  7. Evaluate on held-out test split (MAE, RMSE, R² in original [5,95] scale)
  8. Save training loss curve -> artifacts/training_loss.png
  9. Save trained model -> models/interview_assessment_model.keras

WHY NORMALISE TARGETS?
  The sigmoid output is ∈ (0, 1). If we fed raw scores (e.g., 72.43) as
  targets the MSE loss would be enormous and the gradient signal would push
  the sigmoid into saturation immediately. Normalising targets to [0, 1]
  lets the network learn the correct shape of the mapping.

  At prediction time, predict.py reverses the normalisation:
      score = SCORE_MIN + SCORE_RANGE * sigmoid(z)
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

try:
    from fusion.feature_vector import FEATURE_NAMES
except ImportError:
    from ml.feature_vector import FEATURE_NAMES

from ml.model import create_mlp_regression_model
from ml.evaluation import evaluate_regression_model
from ml.synthetic_data import generate_synthetic_dataset
from ml.predict import SCORE_MIN, SCORE_RANGE

# Resolve paths relative to this file
_ML_DIR       = Path(__file__).resolve().parent
_APP_DIR      = _ML_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

MODEL_SAVE_PATH = _PROJECT_ROOT / "models" / "interview_assessment_model.keras"
LOSS_PLOT_PATH  = _PROJECT_ROOT / "artifacts" / "training_loss.png"


def load_training_data(csv_path: str = "data/ml_training_data.csv"):
    """
    Load multimodal feature matrix X and target score vector y from CSV.
    Auto-generates synthetic dataset if CSV is missing.
    """
    from core.dataset import resolve_path
    resolved = resolve_path(csv_path)
    if not resolved.exists():
        print(f"Dataset not found at '{resolved}'. Generating synthetic dataset...")
        generate_synthetic_dataset(str(resolved), num_samples=100, seed=42)
        resolved = resolve_path(csv_path)

    X, y = [], []
    with open(resolved, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append([float(row[col]) for col in FEATURE_NAMES])
            y.append(float(row["target_score"]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_data(X: np.ndarray, y: np.ndarray,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               seed: int = 42):
    """
    Split dataset into train / validation / test sets using a fixed random
    shuffle. With 100 samples this gives approximately 70 / 15 / 15.

    Args:
        X:          Feature matrix.
        y:          Target vector.
        val_ratio:  Fraction for validation set.
        test_ratio: Fraction for held-out test set.
        seed:       Random seed for reproducibility.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(seed)
    idx = np.random.permutation(len(X))

    n_test = max(1, int(len(X) * test_ratio))
    n_val  = max(1, int(len(X) * val_ratio))

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])


def train_neural_network(
    csv_path: str = "data/ml_training_data.csv",
    epochs: int = 100,
    batch_size: int = 8
) -> dict:
    """
    Train the sigmoid MLP regression model on the synthetic dataset.

    Targets are normalised to [0, 1] before training so that the sigmoid
    output can represent the full score range without saturation:
        y_norm = (y - SCORE_MIN) / SCORE_RANGE   # 5–95  ->  0–1

    The model is evaluated on the held-out test split in the original
    [5, 95] scale (MAE, RMSE, R²) for human-interpretable metrics.

    Returns:
        dict with keys: metrics, model_path, plot_path, model
    """
    print("\n" + "=" * 60)
    print("      TRAINING MULTIMODAL NEURAL NETWORK REGRESSOR")
    print("=" * 60)

    # 1. Load data
    X, y = load_training_data(csv_path)
    print(f"Loaded {len(X)} records  |  {X.shape[1]} input features  "
          f"|  Score range: {y.min():.1f}–{y.max():.1f}")

    # 2. 70 / 15 / 15 split
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    print(f"Split  —  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # 3. Normalise targets to [0, 1] for sigmoid training
    #    SCORE_MIN = 5, SCORE_RANGE = 90  (from predict.py)
    y_train_norm = (y_train - SCORE_MIN) / SCORE_RANGE
    y_val_norm   = (y_val   - SCORE_MIN) / SCORE_RANGE

    # 4. Adapt Normalization layer on training features only
    print("\nAdapting feature normalization layer...")
    normalizer = layers.Normalization(axis=-1, name="Feature_Normalization")
    normalizer.adapt(X_train)

    # 5. Build model (sigmoid output)
    model = create_mlp_regression_model(
        input_dim=18,
        normalizer=normalizer,
        learning_rate=0.01
    )
    print("\n--- Model Architecture ---")
    model.summary()

    # 6. Train (forward pass -> MSE loss -> backprop -> weight update)
    print(f"\nTraining for {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train_norm,
        validation_data=(X_val, y_val_norm),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 7. Evaluate on test set — convert predictions back to original scale
    y_pred_norm = model.predict(X_test, verbose=0).flatten()
    y_pred      = SCORE_MIN + SCORE_RANGE * y_pred_norm   # rescale to [5, 95]
    metrics = evaluate_regression_model(y_test, y_pred)

    print("\n" + "-" * 44)
    print(f"  Test MAE  (Mean Absolute Error)  : {metrics['MAE']:.2f} pts")
    print(f"  Test RMSE (Root Mean Sq Error)   : {metrics['RMSE']:.2f} pts")
    print(f"  Test R²   (Coefficient of Det.)  : {metrics['R2']:.4f}")
    print("-" * 44)
    print("Note: Evaluated on held-out test split of the synthetic dataset.")

    # 8. Save training loss curve
    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["loss"],
             label="Training Loss (MSE)", color="#2563eb", linewidth=2)
    plt.plot(history.history["val_loss"],
             label="Validation Loss (MSE)", color="#dc2626", linewidth=2, linestyle="--")
    plt.title("Neural Network Training Loss (Sigmoid Regression, Normalised Targets)",
              fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("MSE Loss (on normalised [0–1] targets)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(str(LOSS_PLOT_PATH), dpi=150)
    plt.close()
    print(f"Training loss curve saved to: '{LOSS_PLOT_PATH}'")

    # 9. Save model
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    print(f"Model saved to: '{MODEL_SAVE_PATH}'")
    print("=" * 60 + "\n")

    return {
        "metrics":    metrics,
        "model_path": str(MODEL_SAVE_PATH),
        "plot_path":  str(LOSS_PLOT_PATH),
        "model":      model
    }


if __name__ == "__main__":
    train_neural_network()
