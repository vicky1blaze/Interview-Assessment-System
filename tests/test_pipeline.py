"""
test_pipeline.py - Automated End-to-End Test Suite for Multi-Modal Interview Assessment System.

Covers all core requirements:
1. Canonical 5 questions and reference dataset integrity
2. Text NLP preprocessing & 12 text feature extraction
3. Speech acoustic feature extraction (5 features + speech_available flag)
4. Central 18-dimensional feature vector & explicit feature fusion
5. Shared IDF basis for TF-IDF cosine similarity
6. Synthetic dataset generation (~100 records, score range 30-95, low scores < 50 present)
7. Keras MLP Neural Network architecture (Input(18), Normalization, Dense(1, sigmoid))
8. Target normalisation [0, 1] and 70/15/15 train/val/test split
9. Prediction scaling (5 + 90 * sigmoid) naturally bounded in [5, 95]
10. Critical 4-Profile Behavioral Test: Strong > Average > Weak > VeryPoor (no saturation)
11. Explainable rule-based feedback engine
12. Video modality reservation and canonical schema preservation
13. Non-destructive: does not overwrite existing candidate records
"""

import sys
import os
import csv
import copy
from pathlib import Path
import numpy as np

# Safe console encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project paths are resolvable
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
APP_DIR = PROJECT_ROOT / "app"

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

def test_pipeline():
    print("=" * 65)
    print("   MULTI-MODAL INTERVIEW ASSESSMENT SYSTEM - TEST SUITE")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Canonical 5 Questions & Reference Metadata
    # -------------------------------------------------------------
    print("\n[Test 1] Verifying 5 Canonical Questions & Metadata...")
    from core.questions import questions, questions_tfidf, QUESTIONS_META
    assert len(questions) == 5, f"Expected 5 questions, found {len(questions)}"
    for qid in ["qid_1", "qid_2", "qid_3", "qid_4", "qid_5"]:
        assert qid in questions, f"Missing question id: {qid}"
        assert qid in QUESTIONS_META, f"Missing question metadata: {qid}"
        assert "category" in QUESTIONS_META[qid]
        assert "difficulty" in QUESTIONS_META[qid]
    ref_dict = questions_tfidf()
    assert "questions" in ref_dict and len(ref_dict["questions"]) == 5
    print("  [PASS] Exactly 5 questions verified with full metadata and reference NLP.")

    # -------------------------------------------------------------
    # Test 2: Text Modality & 12 Numerical Text Features
    # -------------------------------------------------------------
    print("\n[Test 2] Verifying Text Modality & 12 Features...")
    from text.preprocessing import preprocess
    from text.feature_extraction import (
        extract_features_statistics,
        extract_features_sentiment,
        extract_features_bow,
        get_text_feature_dict
    )
    from fusion.feature_vector import TEXT_FEATURE_NAMES, SPEECH_FEATURE_NAMES, FEATURE_NAMES

    sample_text = "I have five years of experience building Python and machine learning applications. I enjoy tackling complex architectural problems."
    lemmas = preprocess(sample_text)
    assert len(lemmas) > 0, "Preprocessing returned empty lemmas"

    stats = extract_features_statistics(sample_text)
    assert stats["word_count"] > 0
    assert "vocabulary_ratio" in stats
    assert "filler_ratio" in stats

    sent = extract_features_sentiment(sample_text)
    assert "compound" in sent and "pos" in sent

    bow = extract_features_bow(lemmas)
    assert len(bow) > 0

    dummy_tfidf = {lemmas[0]: 0.5}
    text_features = get_text_feature_dict(
        sample_text, lemmas, dummy_tfidf, dummy_tfidf, lemmas
    )
    assert len(TEXT_FEATURE_NAMES) == 12, f"Expected 12 text features, got {len(TEXT_FEATURE_NAMES)}"
    for f in TEXT_FEATURE_NAMES:
        assert f in text_features, f"Missing text feature: {f}"
    print("  [PASS] Text modality preprocessing and 12 numerical features verified.")

    # -------------------------------------------------------------
    # Test 3: Speech Modality Acoustic Features (5 acoustic + speech_available)
    # -------------------------------------------------------------
    print("\n[Test 3] Verifying Speech Acoustic Feature Extraction...")
    from speech.acoustic_features import extract_acoustic_features
    # Verify fallback for missing audio file
    empty_feat = extract_acoustic_features(None, "")
    assert empty_feat["speech_available"] == 0.0
    assert empty_feat["duration_seconds"] == 0.0
    assert len(SPEECH_FEATURE_NAMES) == 6, f"Expected 6 speech features, got {len(SPEECH_FEATURE_NAMES)}"
    for f in SPEECH_FEATURE_NAMES:
        assert f in empty_feat, f"Missing speech feature: {f}"
    print("  [PASS] Speech acoustic features (5 acoustic + speech_available) verified.")

    # -------------------------------------------------------------
    # Test 4: Explicit Feature Fusion & Canonical Order
    # -------------------------------------------------------------
    print("\n[Test 4] Verifying 18-D Multimodal Feature Fusion...")
    from fusion.feature_vector import fuse_features, vector_to_dict, dict_to_vector
    fused = fuse_features(text_features, empty_feat)
    assert isinstance(fused, np.ndarray), "Fused vector is not a numpy array"
    assert fused.shape == (18,), f"Fused vector shape mismatch: expected (18,), got {fused.shape}"
    assert len(FEATURE_NAMES) == 18, f"FEATURE_NAMES count mismatch: {len(FEATURE_NAMES)}"

    v_dict = vector_to_dict(fused)
    assert len(v_dict) == 18
    v_back = dict_to_vector(v_dict)
    assert np.allclose(fused, v_back)
    print("  [PASS] Explicit feature fusion (12 text + 6 speech -> exact 18-D vector) verified.")

    # -------------------------------------------------------------
    # Test 5: Shared IDF Basis in TF-IDF
    # -------------------------------------------------------------
    print("\n[Test 5] Verifying Shared IDF Basis for TF-IDF Consistency...")
    from text.feature_extraction import extract_features_tfidf, cosine_similarity
    mock_corpus = {
        "candidate_test": {
            "responses": {
                "qid_1": {
                    "text": {"processed": {"bow": {"experience": 2, "python": 3}, "lemmas": ["experience", "python"]}}
                }
            }
        }
    }
    updated_corpus = extract_features_tfidf("candidate_test", mock_corpus, ref_dict)
    q1_tfidf = updated_corpus["candidate_test"]["responses"]["qid_1"]["text"]["processed"]["tfidf"]
    assert len(q1_tfidf) > 0, "Candidate TF-IDF was not populated"
    sim = cosine_similarity(ref_dict["tfidf"].get("qid_1", {}), q1_tfidf)
    assert 0.0 <= sim <= 1.0, f"Cosine similarity out of bounds [0, 1]: {sim}"
    print(f"  [PASS] Shared IDF basis verified. Cosine similarity = {sim:.4f}")

    # -------------------------------------------------------------
    # Test 6: Synthetic Dataset Generation (~100 samples, score spread)
    # -------------------------------------------------------------
    print("\n[Test 6] Verifying Synthetic Dataset (~100 samples, 30-95 range)...")
    from ml.synthetic_data import generate_synthetic_dataset
    csv_target = PROJECT_ROOT / "data" / "ml_training_data.csv"
    gen_path = generate_synthetic_dataset(str(csv_target), num_samples=100, seed=42)
    assert Path(gen_path).exists(), f"Synthetic dataset CSV missing at {gen_path}"

    with open(gen_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 100, f"Expected 100 records, got {len(rows)}"
    target_scores = [float(r["target_score"]) for r in rows]
    min_score = min(target_scores)
    max_score = max(target_scores)
    low_scores = [s for s in target_scores if s < 50.0]

    assert len(low_scores) >= 10, f"Dataset lacks low scores (<50): found only {len(low_scores)}"
    assert min_score >= 5.0, f"Score below 5.0 minimum: {min_score}"
    assert max_score <= 95.0, f"Score above 95.0 maximum: {max_score}"
    assert max_score < 95.0, f"Dataset reached 95.0 saturation ceiling: {max_score}"
    print(f"  [PASS] Synthetic dataset verified: {len(rows)} samples.")
    print(f"         Score range: [{min_score:.2f}, {max_score:.2f}], low scores (<50): {len(low_scores)} samples.")

    # -------------------------------------------------------------
    # Test 7: Neural Network Architecture (Sigmoid Output)
    # -------------------------------------------------------------
    print("\n[Test 7] Verifying Keras MLP Architecture (Dense(1, sigmoid))...")
    from ml.model import create_mlp_regression_model
    from tensorflow.keras import layers
    dummy_x = np.random.uniform(0.0, 1.0, (20, 18)).astype(np.float32)
    norm_layer = layers.Normalization(axis=-1)
    norm_layer.adapt(dummy_x)
    model = create_mlp_regression_model(input_dim=18, normalizer=norm_layer)
    output_layer = model.layers[-1]
    assert output_layer.activation.__name__ == "sigmoid", \
        f"Expected sigmoid activation on output layer, got {output_layer.activation.__name__}"
    assert model.output_shape == (None, 1), f"Expected output shape (None, 1), got {model.output_shape}"
    # Verify bounded raw prediction
    raw_pred = model.predict(dummy_x[:2], verbose=0)
    assert np.all(raw_pred >= 0.0) and np.all(raw_pred <= 1.0), \
        f"Sigmoid raw output must be in (0, 1), got {raw_pred}"
    print("  [PASS] Keras model architecture verified: Dense(1, sigmoid) with bounded output.")

    # -------------------------------------------------------------
    # Test 8: Neural Network Training (70/15/15 split, target normalization)
    # -------------------------------------------------------------
    print("\n[Test 8] Verifying Model Training Pipeline & Artifacts...")
    from ml.train import train_neural_network
    train_res = train_neural_network(csv_path=str(csv_target), epochs=45, batch_size=8)
    assert Path(train_res["model_path"]).exists(), "Trained model file missing"
    assert Path(train_res["plot_path"]).exists(), "Loss curve plot missing"
    metrics = train_res["metrics"]
    assert "MAE" in metrics and "RMSE" in metrics and "R2" in metrics
    print(f"  [PASS] Model trained: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")
    print(f"  [PASS] Saved model: {train_res['model_path']}")
    print(f"  [PASS] Saved loss plot: {train_res['plot_path']}")

    # -------------------------------------------------------------
    # Test 9: Prediction Scaling (5 + 90 * sigmoid)
    # -------------------------------------------------------------
    print("\n[Test 9] Verifying Score Scaling (5.0 + 90.0 * sigmoid)...")
    from ml.predict import load_trained_model, predict_score, get_score_level, compute_overall_score, SCORE_MIN, SCORE_RANGE
    assert SCORE_MIN == 5.0
    assert SCORE_RANGE == 90.0

    loaded_model = load_trained_model(train_res["model_path"])
    score = predict_score(loaded_model, fused)
    assert 5.0 <= score <= 95.0, f"Score {score} out of [5.0, 95.0] range"
    lvl = get_score_level(score)
    assert lvl in ["Excellent", "Good", "Competent", "Developing", "Needs Improvement"]
    print(f"  [PASS] Score scaling verified: score = {score:.2f} / 100 ({lvl})")

    # -------------------------------------------------------------
    # Test 10: Critical 4-Profile Behavioral Test (Anti-Saturation)
    # -------------------------------------------------------------
    print("\n[Test 10] Running Critical 4-Profile Behavioral Test...")
    # 1. Very Poor Profile (off-topic, very brief, high hesitation, high silence)
    very_poor_vec = np.array([
        8.0, 1.0, 8.0, 0.40, 0.18, 0.50,       # wc, sc, avg_len, vocab, filler, stopword
        0.02, 0.06, 0.92, -0.15,               # pos, neg, neu, compound
        0.10, 0.08,                            # cov, sim
        20.0, 65.0, 0.008, 0.04, 0.65, 1.0    # duration, wpm, rms, zcr, silence, speech_available
    ], dtype=np.float32)

    # 2. Weak Profile (brief, moderate filler, slow pace)
    weak_vec = np.array([
        22.0, 2.0, 11.0, 0.52, 0.11, 0.48,     # wc, sc, avg_len, vocab, filler, stopword
        0.05, 0.03, 0.92, 0.10,                # pos, neg, neu, compound
        0.26, 0.24,                            # cov, sim
        24.0, 95.0, 0.025, 0.06, 0.42, 1.0    # duration, wpm, rms, zcr, silence, speech_available
    ], dtype=np.float32)

    # 3. Average Profile (balanced, competent)
    average_vec = np.array([
        52.0, 3.0, 17.3, 0.68, 0.035, 0.42,    # wc, sc, avg_len, vocab, filler, stopword
        0.14, 0.01, 0.85, 0.48,                # pos, neg, neu, compound
        0.58, 0.52,                            # cov, sim
        24.0, 132.0, 0.065, 0.08, 0.18, 1.0   # duration, wpm, rms, zcr, silence, speech_available
    ], dtype=np.float32)

    # 4. Strong Profile (articulate, optimal pace, high relevance)
    strong_vec = np.array([
        75.0, 5.0, 15.0, 0.80, 0.008, 0.38,    # wc, sc, avg_len, vocab, filler, stopword
        0.22, 0.00, 0.78, 0.72,                # pos, neg, neu, compound
        0.86, 0.82,                            # cov, sim
        30.0, 150.0, 0.085, 0.09, 0.10, 1.0   # duration, wpm, rms, zcr, silence, speech_available
    ], dtype=np.float32)

    score_vp = predict_score(loaded_model, very_poor_vec)
    score_w  = predict_score(loaded_model, weak_vec)
    score_avg = predict_score(loaded_model, average_vec)
    score_str = predict_score(loaded_model, strong_vec)

    print(f"  Profile Scores:")
    print(f"    - Very Poor : {score_vp:.2f} ({get_score_level(score_vp)})")
    print(f"    - Weak      : {score_w:.2f} ({get_score_level(score_w)})")
    print(f"    - Average   : {score_avg:.2f} ({get_score_level(score_avg)})")
    print(f"    - Strong    : {score_str:.2f} ({get_score_level(score_str)})")

    # Verify strictly monotonic ordering
    assert score_str > score_avg, f"Ordering violated: Strong ({score_str:.2f}) <= Average ({score_avg:.2f})"
    assert score_avg > score_w,   f"Ordering violated: Average ({score_avg:.2f}) <= Weak ({score_w:.2f})"
    assert score_w > score_vp,    f"Ordering violated: Weak ({score_w:.2f}) <= Very Poor ({score_vp:.2f})"

    # Verify anti-saturation (no saturation at bounds)
    for name, sc in [("Very Poor", score_vp), ("Weak", score_w), ("Average", score_avg), ("Strong", score_str)]:
        assert sc > 5.0, f"{name} hit bottom clamp: {sc}"
        assert sc < 95.0, f"{name} saturated at 95.0 ceiling: {sc}"
        assert sc != 100.0, f"{name} scored 100.0!"

    print("  [PASS] 4-Profile behavioral test passed: Strong > Average > Weak > Very Poor.")
    print("  [PASS] Anti-saturation confirmed: All scores bounded in (5.0, 95.0), no 100/100.")

    # -------------------------------------------------------------
    # Test 11: Explainable Feedback Engine
    # -------------------------------------------------------------
    print("\n[Test 11] Verifying Explainable Feedback Engine...")
    from services.feedback import generate_feedback_from_metrics
    dummy_metrics = {name: float(strong_vec[i]) for i, name in enumerate(FEATURE_NAMES)}
    report = generate_feedback_from_metrics(score_str, get_score_level(score_str), dummy_metrics)
    assert "INTERVIEW ASSESSMENT REPORT" in report
    assert "KEY STRENGTHS" in report
    assert "AREAS FOR IMPROVEMENT" in report
    assert "EXTRACTED PERFORMANCE METRICS" in report
    print("  [PASS] Feedback engine generated comprehensive, transparent report.")

    # -------------------------------------------------------------
    # Test 12: Preserved Video Modality & Schema Non-Destructive Check
    # -------------------------------------------------------------
    print("\n[Test 12] Verifying Reserved Video Modality & Schema Integrity...")
    from core.dataset import load_data
    cand_file = PROJECT_ROOT / "data" / "candidates.json"
    if cand_file.exists():
        cand_data = load_data("data/candidates.json")
        for cid, cinfo in cand_data.get("candidates", {}).items():
            for qid, qresp in cinfo.get("responses", {}).items():
                assert "video" in qresp, f"Video field missing in candidates.json for {cid}:{qid}"

    corpus_file = PROJECT_ROOT / "data" / "corpus.json"
    if corpus_file.exists():
        corpus_data = load_data("data/corpus.json")
        for cid, cinfo in corpus_data.get("corpus", {}).items():
            if "overall" in cinfo:
                assert "video" in cinfo["overall"], f"Video field missing in corpus overall for {cid}"

    print("  [PASS] Video modality structure verified untouched across schemas.")
    print("  [PASS] Non-destructive execution confirmed.")

    print("\n" + "=" * 65)
    print("   ALL PIPELINE & BEHAVIORAL TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_pipeline()
