"""
feature_vector.py - Central Multimodal Feature Vector Definition and Feature Fusion.

This module is the SINGLE SOURCE OF TRUTH for the 18-dimensional feature vector.
Defines the exact feature order and implements explicit feature-level fusion
concatenating text NLP features with speech acoustic features.
"""

import numpy as np

# Exact 12 Text Modality Features
TEXT_FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "average_sentence_length",
    "vocabulary_ratio",
    "filler_ratio",
    "stopword_ratio",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_compound",
    "keyword_coverage",
    "tfidf_cosine_similarity"
]

# Exact 6 Speech Modality Features (5 acoustic + 1 modality availability flag)
SPEECH_FEATURE_NAMES = [
    "duration_seconds",
    "speech_rate_wpm",
    "rms_energy",
    "zero_crossing_rate",
    "silence_ratio",
    "speech_available"
]

# Complete 18-Feature Vector in strict canonical order
FEATURE_NAMES = TEXT_FEATURE_NAMES + SPEECH_FEATURE_NAMES
assert len(FEATURE_NAMES) == 18, f"Expected 18 features, got {len(FEATURE_NAMES)}"

def fuse_features(text_features: dict, speech_features: dict) -> np.ndarray:
    """
    Perform explicit feature-level multimodal fusion:
    [text_vector (12)] + [speech_vector (6)] -> [fused_vector (18)]

    Args:
        text_features: Dictionary containing the 12 text NLP metrics.
        speech_features: Dictionary containing the 6 speech acoustic metrics.

    Returns:
        1D NumPy float32 array of shape (18,).
    """
    # 1. Construct Text Feature Vector (12 features)
    text_vector = np.array(
        [float(text_features.get(k, 0.0)) for k in TEXT_FEATURE_NAMES],
        dtype=np.float32
    )

    # 2. Construct Speech Feature Vector (6 features)
    speech_vector = np.array(
        [float(speech_features.get(k, 0.0)) for k in SPEECH_FEATURE_NAMES],
        dtype=np.float32
    )

    # 3. Explicit Feature-Level Multimodal Fusion (Concatenation)
    fused_vector = np.concatenate([text_vector, speech_vector], axis=0)

    assert fused_vector.shape == (18,), f"Fused vector shape mismatch: {fused_vector.shape}"
    return fused_vector

def vector_to_dict(vector: np.ndarray) -> dict:
    """
    Convert a 1D 18-element NumPy vector to a named feature dictionary.
    """
    vec_flat = np.array(vector, dtype=np.float32).flatten()
    return {name: float(val) for name, val in zip(FEATURE_NAMES, vec_flat)}

def dict_to_vector(feature_dict: dict) -> np.ndarray:
    """
    Extract a 1D 18-element NumPy vector from a feature dictionary using FEATURE_NAMES order.
    """
    return np.array(
        [float(feature_dict.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float32
    )
