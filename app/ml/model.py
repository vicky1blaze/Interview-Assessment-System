"""
model.py - Keras/TensorFlow Neural Network Regression Model.

Constructs a Multilayer Perceptron (MLP) for continuous interview score
prediction using the 18 fused multimodal features.

OUTPUT DESIGN:
  The output neuron uses a sigmoid activation, which naturally constrains
  the raw output to (0, 1). The final score is then scaled to the
  project-defined [5, 95] range in predict.py:

      score = 5 + 90 * sigmoid(z)

  This means the model CANNOT extrapolate beyond 95 or below 5.
  The bound is architectural, not a post-hoc clamp. This is still
  regression — the output is a continuous value like 72.43, not a class.

NEURAL NETWORK STRUCTURE:
  1. Input: 18-dim multimodal feature vector
  2. Normalization: z-score standardisation ((x - mean) / std) per feature,
     fitted on training data so all features have equal influence regardless
     of magnitude (e.g., word_count ~50 vs filler_ratio ~0.05).
  3. Hidden Layer 1: Dense(16, ReLU)
     Forward pass: Z1 = W1 * X + b1; A1 = max(0, Z1)
     ReLU introduces non-linearity so the network learns feature interactions.
  4. Hidden Layer 2: Dense(8, ReLU)
     Further abstracts the 16 activations into higher-level competency signals.
  5. Output Layer: Dense(1, sigmoid)
     Sigmoid f(z) = 1 / (1 + e^-z) squashes output to (0, 1).
     Scaled externally to [5, 95] for human-readable scores.

TRAINING:
  Loss: Mean Squared Error (MSE) computed on normalised targets in [0, 1].
  Optimiser: Adam (adaptive learning rate, momentum-based gradient descent).
  Backpropagation computes dLoss/dW via chain rule and updates weights
  across epochs to minimise prediction error.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_mlp_regression_model(
    input_dim: int = 18,
    normalizer: layers.Normalization = None,
    learning_rate: float = 0.01,
    optimizer_type: str = "adam"
) -> keras.Model:
    """
    Build the Interview Assessment MLP regression model.

    Args:
        input_dim:      Number of input features (must be 18).
        normalizer:     Pre-adapted Normalization layer from training data.
        learning_rate:  Step size for gradient descent weight updates.
        optimizer_type: "adam" (default) or "sgd".

    Returns:
        Compiled Keras Sequential model ready for training.
    """
    model = keras.Sequential(name="Interview_Assessment_MLP_Regressor")

    # --- Step 1: Input ---
    model.add(keras.Input(shape=(input_dim,), name="Multimodal_18D_Input"))

    # --- Step 2: Feature Normalisation ---
    # Standardises each feature to zero mean and unit variance using
    # statistics computed from the training set. This prevents high-magnitude
    # features (e.g., word_count ≈ 60) from dominating low-magnitude ones
    # (e.g., filler_ratio ≈ 0.05).
    if normalizer is not None:
        model.add(normalizer)
    else:
        norm_layer = layers.Normalization(axis=-1, name="Feature_Normalization")
        model.add(norm_layer)

    # --- Step 3: Hidden Layer 1 — Dense(16, ReLU) ---
    # Learns weighted combinations of 18 input features.
    # ReLU: A = max(0, Z) introduces non-linearity and avoids vanishing gradients.
    model.add(layers.Dense(16, activation="relu", name="Hidden_Layer_1_Dense16"))

    # --- Step 4: Hidden Layer 2 — Dense(8, ReLU) ---
    # Further compresses the 16 representations into 8 abstract competency signals.
    model.add(layers.Dense(8, activation="relu", name="Hidden_Layer_2_Dense8"))

    # --- Step 5: Output Layer — Dense(1, sigmoid) ---
    # Sigmoid output ∈ (0, 1) — architecturally bounded.
    # Scaled to [5, 95] in predict.py: score = 5 + 90 * sigmoid(z)
    # This is regression (continuous output), not classification.
    model.add(layers.Dense(1, activation="sigmoid", name="Output_Sigmoid_Bounded"))

    # --- Step 6: Compile ---
    # MSE loss operates on normalised targets [0, 1] during training.
    # Adam: adaptive moment estimation with per-parameter learning rates,
    # computes first-order (momentum) and second-order (variance) gradient moments.
    if optimizer_type.lower() == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )

    return model
