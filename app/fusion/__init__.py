"""
app.fusion - Multimodal feature fusion package.
"""
from fusion.feature_vector import (
    TEXT_FEATURE_NAMES,
    SPEECH_FEATURE_NAMES,
    FEATURE_NAMES,
    fuse_features,
    vector_to_dict,
    dict_to_vector
)

__all__ = [
    "TEXT_FEATURE_NAMES",
    "SPEECH_FEATURE_NAMES",
    "FEATURE_NAMES",
    "fuse_features",
    "vector_to_dict",
    "dict_to_vector"
]
