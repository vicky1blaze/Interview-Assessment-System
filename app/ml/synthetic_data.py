"""
synthetic_data.py - Proof-of-Concept Synthetic Training Data Generator.

Generates realistic proof-of-concept training records across the 5 interview
questions. Maps the 18 multimodal features to a target score (5-95) using a
structured, explainable rubric (content relevance, vocabulary, filler penalties,
speaking pace, acoustic quality, rambling penalties) plus controlled noise.

Key Design Properties:
- Total samples: 100
- Score range: ~30 to ~95 with ample low (<50) and moderate (50-70) samples
- Realistic feature ranges matching real microphones, Faster-Whisper, and text
- Weak influence for sentiment and raw acoustic noise
- Anti-verbosity penalty preventing rambling answers from scoring high
"""

import os
import csv
from pathlib import Path
import numpy as np

try:
    from fusion.feature_vector import FEATURE_NAMES
except ImportError:
    from ml.feature_vector import FEATURE_NAMES

try:
    from core.dataset import resolve_path
except ImportError:
    from dataset import resolve_path

QUESTIONS_LIST = ["qid_1", "qid_2", "qid_3", "qid_4", "qid_5"]

def compute_synthetic_target_score(feat: dict, rng: np.random.RandomState) -> float:
    """
    Calculate a synthetic assessment score (5-95) based on an explainable
    rubric incorporating all text and speech features plus slight random noise.

    Design constraints:
    - Base score anchor: 35.0
    - Content relevance (keyword coverage + TF-IDF similarity): dominant factor
    - Length: optimal in 35-90 words; penalized if very short or rambling off-topic
    - Sentiment & raw acoustics: minor/weak influence
    - Natural score ceiling: ~92-94 for stellar answers; floor: ~25-35 for very poor
    """
    # Baseline anchor
    score = 35.0

    # 1. Content Relevance (Dominant text factor)
    # Keyword coverage adds up to +20 points
    score += feat["keyword_coverage"] * 20.0
    # TF-IDF cosine similarity adds up to +16 points
    score += feat["tfidf_cosine_similarity"] * 16.0

    # 2. Text Lexical Quality & Length
    wc = feat["word_count"]
    if wc < 15:
        # Sharp penalty for one-line or un-elaborated answers
        score -= (15 - wc) * 1.0
    elif 35 <= wc <= 90:
        # Optimal interview response length
        score += 6.0
    elif 90 < wc <= 150:
        score += 4.0
    else:
        # Verbosity guard: long answer with low relevance is penalized
        if feat["keyword_coverage"] < 0.45:
            score -= 3.0
        else:
            score += 2.0

    # Vocabulary diversity bonus (up to +6)
    score += min(feat["vocabulary_ratio"], 1.0) * 6.0

    # Filler word penalty (e.g. 10% fillers = -3.0 pts, 20% fillers = -6.0 pts)
    score -= feat["filler_ratio"] * 30.0

    # 3. Sentiment & Tone (Weak influence per prompt requirements)
    score += feat["sentiment_pos"] * 2.5
    score -= feat["sentiment_neg"] * 5.0
    score += feat["sentiment_compound"] * 1.5

    # 4. Speech Delivery & Acoustic Features (when speech is present)
    if feat["speech_available"] > 0.5:
        wpm = feat["speech_rate_wpm"]
        if 120.0 <= wpm <= 165.0:
            score += 7.0  # Optimal conversational pace
        elif (95.0 <= wpm < 120.0) or (165.0 < wpm <= 190.0):
            score += 2.5  # Slightly slow or slightly brisk
        else:
            score -= 4.0  # Noticeably hesitant or rushed

        # Silence & pause ratio penalty (high hesitation = penalty)
        score -= feat["silence_ratio"] * 14.0

        # Vocal energy (audible projection bonus, weak influence)
        if 0.01 <= feat["rms_energy"] <= 0.12:
            score += 2.0

        # Zero crossing rate (weak acoustic marker)
        if 0.04 <= feat["zero_crossing_rate"] <= 0.18:
            score += 1.0

    # 5. Small Gaussian noise
    noise = rng.normal(0.0, 1.0)
    score += noise

    # Bound within [5.0, 95.0]
    return float(np.clip(round(score, 2), 5.0, 95.0))

def generate_synthetic_dataset(output_path: str = "data/ml_training_data.csv", num_samples: int = 100, seed: int = 42):
    """
    Generate synthetic multimodal interview training samples.
    Produces ~100 samples spanning the 30-95 score range across 7 candidate profiles
    with realistic distributions for word count, sentence length, speech acoustics, etc.
    """
    rng = np.random.RandomState(seed)
    records = []

    # Profiles representing diverse performance bands across [30, 95]
    # (name, cov_range, sim_range, wc_range, fil_range, wpm_range, sil_range, sp_prob)
    profiles = [
        # Very poor / off-topic (target score ~30-45)
        ("very_poor_offtopic", (0.05, 0.22), (0.05, 0.20), (5, 22),   (0.10, 0.22), (60, 95),   (0.45, 0.78), 0.85),
        # Weak / hesitant (target score ~42-55)
        ("weak_hesitant",      (0.18, 0.38), (0.16, 0.36), (18, 45),  (0.08, 0.18), (80, 115),  (0.32, 0.58), 0.90),
        # Solid average / competent (target score ~58-72)
        ("average_competent",  (0.42, 0.66), (0.38, 0.62), (35, 75),  (0.02, 0.06), (120, 148), (0.14, 0.28), 0.85),
        # Verbose rambler: high word count, lower relevance (target score ~48-64)
        ("verbose_rambler",    (0.28, 0.52), (0.25, 0.48), (140, 228), (0.04, 0.10), (165, 210), (0.10, 0.24), 0.85),
        # Strong articulate (target score ~78-88)
        ("strong_articulate",  (0.68, 0.88), (0.64, 0.84), (50, 95),  (0.00, 0.03), (130, 160), (0.08, 0.18), 0.90),
        # Exceptional multi-modal (target score ~88-94)
        ("exceptional",        (0.85, 0.98), (0.80, 0.95), (60, 110), (0.00, 0.015),(135, 155), (0.05, 0.12), 0.95),
        # Text-only mixed quality (target score ~38-82, speech_available = 0.0)
        ("text_only_mixed",    (0.20, 0.85), (0.18, 0.80), (20, 100), (0.01, 0.08), (0, 0),     (0.0, 0.0),   0.0),
    ]

    samples_per_profile = int(np.ceil(num_samples / len(profiles)))
    sample_idx = 1

    for prof_name, cov_r, sim_r, wc_r, fil_r, wpm_r, sil_r, sp_prob in profiles:
        for _ in range(samples_per_profile):
            if len(records) >= num_samples:
                break

            qid = QUESTIONS_LIST[(sample_idx - 1) % len(QUESTIONS_LIST)]
            has_speech = 1.0 if (rng.rand() < sp_prob and prof_name != "text_only_mixed") else 0.0

            word_count = int(rng.randint(wc_r[0], wc_r[1] + 1))
            sentence_count = max(1, int(np.ceil(word_count / rng.uniform(7.0, 18.0))))
            avg_sent_len = round(word_count / sentence_count, 2)
            vocab_ratio = round(float(rng.uniform(0.40, 0.92)), 4)
            filler_ratio = round(float(rng.uniform(fil_r[0], fil_r[1])), 4)
            stopword_ratio = round(float(rng.uniform(0.28, 0.55)), 4)

            pos_s = round(float(rng.uniform(0.02, 0.30)), 4)
            neg_s = round(float(rng.uniform(0.00, 0.08)), 4)
            neu_s = round(max(0.0, 1.0 - (pos_s + neg_s)), 4)
            compound_s = round(float(rng.uniform(0.15, 0.85) if pos_s > neg_s else rng.uniform(-0.25, 0.15)), 4)

            coverage = round(float(rng.uniform(cov_r[0], cov_r[1])), 4)
            similarity = round(float(rng.uniform(sim_r[0], sim_r[1])), 4)

            if has_speech > 0.5:
                wpm = round(float(rng.uniform(wpm_r[0], wpm_r[1])), 2)
                # Realistic audio duration: speaking time + pauses
                speaking_sec = word_count / (wpm / 60.0)
                silence = round(float(rng.uniform(sil_r[0], sil_r[1])), 4)
                duration = round(float(speaking_sec / max(0.2, (1.0 - silence)) + rng.uniform(0.5, 2.5)), 2)
                duration = min(duration, 130.0)
                rms = round(float(rng.uniform(0.005, 0.085)), 4)
                zcr = round(float(rng.uniform(0.03, 0.20)), 4)
            else:
                wpm = 0.0
                duration = 0.0
                rms = 0.0
                zcr = 0.0
                silence = 0.0

            feat = {
                "word_count": float(word_count),
                "sentence_count": float(sentence_count),
                "average_sentence_length": float(avg_sent_len),
                "vocabulary_ratio": float(vocab_ratio),
                "filler_ratio": float(filler_ratio),
                "stopword_ratio": float(stopword_ratio),
                "sentiment_pos": float(pos_s),
                "sentiment_neg": float(neg_s),
                "sentiment_neu": float(neu_s),
                "sentiment_compound": float(compound_s),
                "keyword_coverage": float(coverage),
                "tfidf_cosine_similarity": float(similarity),
                "duration_seconds": float(duration),
                "speech_rate_wpm": float(wpm),
                "rms_energy": float(rms),
                "zero_crossing_rate": float(zcr),
                "silence_ratio": float(silence),
                "speech_available": float(has_speech)
            }

            target_score = compute_synthetic_target_score(feat, rng)

            row = {
                "sample_id": f"sample_{sample_idx:03d}",
                "qid": qid,
                **feat,
                "target_score": target_score
            }
            records.append(row)
            sample_idx += 1

    # Write to CSV
    target_file = resolve_path(output_path, for_writing=True)
    target_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sample_id", "qid"] + FEATURE_NAMES + ["target_score"]
    with open(target_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} synthetic training records at '{target_file}'.")
    return target_file

if __name__ == "__main__":
    generate_synthetic_dataset(num_samples=100)
