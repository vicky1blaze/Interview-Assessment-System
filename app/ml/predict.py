"""
predict.py - Model Loading and Interview Assessment Score Prediction.

Loads the serialized Keras regression model, feeds the 18-dimensional
multimodal feature vector through forward propagation, and outputs a
continuous assessment score in the range [5, 95].

SCORE SCALING:
  The output neuron uses sigmoid activation — output ∈ (0, 1).
  Scores are recovered via:

      score = 5 + 90 * sigmoid_output

  This maps:
    sigmoid → 0.0  =>  score ≈  5  (very poor)
    sigmoid → 0.5  =>  score ≈ 50  (average)
    sigmoid → 1.0  =>  score ≈ 95  (excellent)

  The bound is structural — the model cannot predict > 95 or < 5.
  A small safety clamp [5, 95] is retained as a guard against
  floating-point edge cases only.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Resolve project root relative to this file (app/ml/predict.py)
_ML_DIR      = Path(__file__).resolve().parent
_APP_DIR     = _ML_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "interview_assessment_model.keras"

# Module-level cache so the model is only loaded once per process
_loaded_model = None

# Score range constants — single source of truth
SCORE_MIN = 5.0
SCORE_MAX = 95.0
SCORE_RANGE = SCORE_MAX - SCORE_MIN  # 90.0


def load_trained_model(model_path: str = None) -> keras.Model:
    """
    Load the trained neural network from disk (cached after first load).
    Automatically triggers training if the model file is missing.

    Args:
        model_path: Optional path override. Defaults to models/ directory.

    Returns:
        Loaded (or freshly trained) Keras model.
    """
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not path.exists():
        print(f"\n[AI Model] Model not found at '{path}'.")
        print("[AI Model] Initiating initial training from synthetic dataset...")
        from ml.train import train_neural_network
        res = train_neural_network()
        _loaded_model = res["model"]
        return _loaded_model

    try:
        _loaded_model = keras.models.load_model(str(path))
        return _loaded_model
    except Exception as e:
        print(f"[Error] Failed to load model from '{path}': {e}")
        print("[AI Model] Retraining model...")
        from ml.train import train_neural_network
        res = train_neural_network()
        _loaded_model = res["model"]
        return _loaded_model


def predict_score(model: keras.Model, feature_vector: np.ndarray) -> float:
    """
    Predict a continuous interview assessment score from an 18-dim feature vector.

    The model outputs a sigmoid value in (0, 1). This is scaled to [5, 95]:
        score = SCORE_MIN + SCORE_RANGE * sigmoid_output

    This is regression — the output is a continuous value (e.g., 72.43),
    not a class label.

    Args:
        model:          Trained Keras sigmoid regression model.
        feature_vector: 1D NumPy array of length 18.

    Returns:
        Float score in [5.0, 95.0], rounded to 2 decimal places.
    """
    vec = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    if vec.shape[1] != 18:
        raise ValueError(f"Expected 18 features, got {vec.shape[1]}")

    # Forward propagation: Normalization -> Dense(16,relu) -> Dense(8,relu) -> Dense(1,sigmoid)
    sigmoid_output = float(model.predict(vec, verbose=0)[0][0])

    # Scale from (0, 1) to [SCORE_MIN, SCORE_MAX] = [5, 95]
    score = SCORE_MIN + SCORE_RANGE * sigmoid_output

    # Safety guard against floating-point edge cases only
    score = max(SCORE_MIN, min(SCORE_MAX, score))

    return round(score, 2)


def get_score_level(score: float) -> str:
    """
    Map a continuous [5, 95] score to a qualitative performance band.

    Bands are designed for the [5, 95] output range:
      80–95  Excellent
      65–79  Good
      50–64  Average
      5–49   Needs Improvement
    """
    if score >= 80.0:
        return "Excellent"
    elif score >= 65.0:
        return "Good"
    elif score >= 50.0:
        return "Average"
    else:
        return "Needs Improvement"


def compute_overall_score(question_scores: list) -> float:
    """
    Compute overall interview score as arithmetic mean of 5 question scores.
    Clamped to [SCORE_MIN, SCORE_MAX] = [5, 95].

    Args:
        question_scores: List of float scores, one per question.

    Returns:
        Mean score rounded to 2 decimal places.
    """
    if not question_scores:
        return SCORE_MIN
    avg = float(np.mean(question_scores))
    return round(max(SCORE_MIN, min(SCORE_MAX, avg)), 2)
