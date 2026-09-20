"""
acoustic_features.py - Lightweight Speech & Audio Feature Extraction.

Extracts 5 acoustic features from candidate audio files using only NumPy
and Soundfile (no librosa or large dependencies):
1. duration_seconds
2. speech_rate_wpm
3. rms_energy
4. zero_crossing_rate
5. silence_ratio
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf

def extract_acoustic_features(audio_path: str, transcript: str = "", silence_threshold: float = 0.01) -> dict:
    """
    Extract acoustic features from an audio file.
    Handles missing files, zero-duration audio, and corrupted audio gracefully.
    
    Returns:
        dict with:
            duration_seconds (float)
            speech_rate_wpm (float)
            rms_energy (float)
            zero_crossing_rate (float)
            silence_ratio (float)
            speech_available (float: 1.0 if audio processed, 0.0 otherwise)
    """
    default_features = {
        "duration_seconds": 0.0,
        "speech_rate_wpm": 0.0,
        "rms_energy": 0.0,
        "zero_crossing_rate": 0.0,
        "silence_ratio": 0.0,
        "speech_available": 0.0
    }

    if not audio_path or not os.path.exists(audio_path):
        return default_features

    try:
        data, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception as e:
        print(f"Warning: Unable to read audio file '{audio_path}': {e}")
        return default_features

    # Convert multi-channel (stereo) to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    sample_count = len(data)
    if sample_count == 0 or sample_rate <= 0:
        return default_features

    # 1. duration_seconds = audio sample count / sample rate
    duration_seconds = float(sample_count / sample_rate)

    # 2. speech_rate_wpm = transcribed word count / duration in minutes
    word_count = len(transcript.strip().split()) if transcript else 0
    duration_minutes = duration_seconds / 60.0
    speech_rate_wpm = float(word_count / duration_minutes) if duration_minutes > 0 else 0.0

    # 3. rms_energy = sqrt(mean(audio^2))
    rms_energy = float(np.sqrt(np.mean(data ** 2)))

    # 4. zero_crossing_rate = rate of sign changes in the waveform
    # Replace zeros to ensure exact sign transitions
    sign_changes = np.abs(np.diff(np.sign(data))) > 0
    zero_crossing_rate = float(np.sum(sign_changes) / (sample_count - 1)) if sample_count > 1 else 0.0

    # 5. silence_ratio = percentage of samples below amplitude threshold
    silent_samples = np.sum(np.abs(data) < silence_threshold)
    silence_ratio = float(silent_samples / sample_count)

    return {
        "duration_seconds": round(duration_seconds, 3),
        "speech_rate_wpm": round(speech_rate_wpm, 2),
        "rms_energy": round(rms_energy, 4),
        "zero_crossing_rate": round(zero_crossing_rate, 4),
        "silence_ratio": round(silence_ratio, 4),
        "speech_available": 1.0
    }
