# Multi-Modal Interview Assessment System

A lightweight, genuine neural-network-based interview assessment system using
TensorFlow/Keras continuous regression over two active modalities: **Text** and **Speech**.

---

## Pipeline Overview

```
Question -> Candidate Response (Audio + Text)
         -> Feature Extraction (12 Text NLP + 6 Speech Acoustic)
         -> Explicit Feature Fusion (18-D vector)
         -> Trained Keras MLP Regressor (18 -> 16 -> 8 -> 1 with Sigmoid Output)
         -> Question & Overall Assessment Score (5-95)
         -> Explainable Feedback Report
```

---

## Running the System

```bash
python app/main.py
```

**Main Menu Options:**
1. **Dummy Data Prediction** — Fast demo with representative candidate profiles loaded from `data/dummy_profiles.json`. No microphone needed.
2. **Manual Interview** — Live speech (Faster-Whisper transcription + acoustic features) or typed text fallback.
3. **Train / Retrain Neural Network** — Regenerate 100-sample synthetic dataset and retrain the Keras MLP with 70/15/15 split.
4. **Exit**

---

## Running the Automated Test Suite

```bash
python tests/test_pipeline.py
```

Validates all system requirements:
- 5 canonical questions with reference metadata and NLP
- 12 text NLP features (statistics, sentiment, BoW, shared IDF cosine similarity, keyword coverage)
- 6 speech acoustic features (duration, WPM, RMS energy, ZCR, silence ratio, speech_available)
- 18-dim feature vector and explicit feature-level fusion (`app/fusion/feature_vector.py`)
- Shared IDF basis across question reference and candidate answers
- Synthetic dataset generation (100 samples spanning 30-95 score range with low scores <50 present)
- Keras MLP training with normalized targets and 70/15/15 train/val/test split
- Output layer architecture: `Dense(1, activation="sigmoid")` scaled to project-defined `[5.0, 95.0]` range
- Critical 4-Profile behavioral test (Strong > Average > Weak > Very Poor) with anti-saturation confirmation
- Explainable feedback engine (`app/services/feedback.py`)
- Canonical JSON schema with preserved video modality
- Non-destructive test execution

---

## Architecture & Project Structure

| Component | Description |
|---|---|
| `app/main.py` | Slim CLI application — main menu loop and mode orchestration |
| `app/core/dataset.py` | Central JSON I/O utilities with path resolution |
| `app/core/questions.py` | 5 canonical interview questions + reference NLP (lemmas, BoW, TF-IDF) |
| `app/fusion/feature_vector.py` | Single source of truth for 18-dim feature definition and multimodal fusion |
| `app/ml/model.py` | Keras MLP: Input(18) → Normalization → 16(ReLU) → 8(ReLU) → 1(Sigmoid) |
| `app/ml/train.py` | Training pipeline: 70/15/15 split, target normalization, evaluation, loss plot |
| `app/ml/synthetic_data.py` | 100-sample proof-of-concept training dataset generator across 7 profiles |
| `app/ml/predict.py` | Model loading, `5.0 + 90.0 * sigmoid` score prediction, qualitative levels |
| `app/ml/evaluation.py` | MAE, RMSE, R² regression metrics |
| `app/services/feedback.py` | Rule-based explainable feedback from neural network score + features |
| `app/text/preprocessing.py` | NLTK tokenization, stop-word removal, lemmatization |
| `app/text/filler_words.py` | Curated verbal hesitations list (um, uh, er, erm, ah, eh, hmm, like) |
| `app/text/feature_extraction.py` | 12 text NLP features, shared-IDF TF-IDF cosine similarity, keyword coverage |
| `app/speech/acoustic_features.py` | 5 acoustic features from audio files (NumPy + SoundFile) |
| `app/speech/speech_to_text.py` | Microphone recording + Faster-Whisper transcription |
| `tests/test_pipeline.py` | Automated end-to-end verification and behavioral test suite |
| `requirements.txt` | Core pinned dependencies |

---

## Data Files

| File | Contents |
|---|---|
| `data/candidates.json` | Candidate interview responses (text, speech paths, video reserved) |
| `data/corpus.json` | NLP processing results, feature vectors, neural network evaluations |
| `data/questions.json` | 5 canonical questions with metadata and reference NLP vectors |
| `data/dummy_profiles.json` | Predefined multi-question profiles (Strong, Hesitant, Mid-Level) |
| `data/ml_training_data.csv` | Synthetic training dataset (100 rows × 18 features + target_score) |
| `models/interview_assessment_model.keras` | Trained Keras model (serialized) |
| `artifacts/training_loss.png` | Training vs. validation MSE loss curve |

---

## Active Modalities

| Modality | Status | Features |
|---|---|---|
| **Text** | ✅ Active | 12 NLP features (statistics, sentiment, BoW, shared-IDF TF-IDF, keyword coverage) |
| **Speech** | ✅ Active | 6 acoustic features + Faster-Whisper transcription |
| **Video** | ⏸ Reserved | Preserved in JSON schema; untouched |

---

## Score Interpretation

The system produces continuous regression scores within the architectural range of **[5.0, 95.0]**:

| Score Band | Performance Level | Description |
|---|---|---|
| **80.0 – 95.0** | Excellent | Highly relevant, articulate, optimal pace, minimal fillers, rich content |
| **68.0 – 79.9** | Good | Relevant, solid vocabulary, clear delivery, occasional minor pauses |
| **55.0 – 67.9** | Competent | Acceptable relevance and vocabulary, moderate hesitations |
| **40.0 – 54.9** | Developing | Noticeable gaps in coverage, high filler ratio, or hesitant delivery |
| **5.0 – 39.9** | Needs Improvement | Very brief, off-topic, or disjointed response |