"""
main.py - Multi-Modal Interview Assessment System.

Interactive CLI application supporting:
1. Dummy Data Prediction (fast, deterministic neural-network regression demo)
2. Manual Interview (Speech modality with Faster-Whisper + Text modality fallback)
3. Train / Retrain Neural Network Regressor
4. Exit

Academic Pipeline:
Question -> Candidate Response (Audio + Text) -> Feature Extraction (12 Text + 6 Speech)
         -> Explicit Feature Fusion (18-D vector) -> Trained Keras Neural Network Regressor
         -> Question & Overall Assessment Scores (5-95) -> Explainable Feedback Engine
"""

import sys
import os
from pathlib import Path
import numpy as np

# Ensure app directory is on python search path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from core.dataset import load_data, save_data, resolve_path
    from core.questions import questions, questions_tfidf
except ImportError:
    from dataset import load_data, save_data, resolve_path
    from questions import questions, questions_tfidf

from text.preprocessing import preprocess
from text.feature_extraction import (
    extract_features_statistics,
    extract_features_sentiment,
    extract_features_bow,
    extract_features_tfidf,
    evaluate_candidate,
    get_text_feature_dict
)
from speech.acoustic_features import extract_acoustic_features
from speech.speech_to_text import record, transcribe_audio, speech_to_text

try:
    from fusion.feature_vector import FEATURE_NAMES, fuse_features, vector_to_dict
except ImportError:
    from ml.feature_vector import FEATURE_NAMES, fuse_features, vector_to_dict

from ml.predict import load_trained_model, predict_score, get_score_level, compute_overall_score

try:
    from services.feedback import feedback, generate_feedback_from_metrics
except ImportError:
    from feedback import feedback, generate_feedback_from_metrics

def run_dummy_prediction():
    """
    Option 1: Fast, deterministic neural-network regression demo.
    Evaluates representative candidate profiles across all 5 interview questions
    using the trained neural network model. Does not require microphone or Faster-Whisper.
    """
    print("\n" + "=" * 60)
    print("      OPTION 1: DUMMY DATA NEURAL NETWORK PREDICTION")
    print("=" * 60)

    # Ensure model is available
    try:
        model = load_trained_model()
    except Exception as e:
        print(f"[Error] Could not load or train neural network: {e}")
        return

    # Load candidate profiles from external canonical dataset
    try:
        dummy_profiles = load_data("data/dummy_profiles.json")
    except Exception as e:
        print(f"[Error] Could not load dummy profiles from data/dummy_profiles.json: {e}")
        return

    print("Select a Demo Profile to test:")
    print("  1. Candidate D01 - Strong Multi-Modal Profile")
    print("  2. Candidate D02 - Hesitant Profile (High Fillers & Pauses)")
    print("  3. Candidate D03 - Balanced Mid-Level Profile")
    choice = input("Enter choice (1-3, default 1): ").strip()
    profile_key = "D01"
    if choice == "2":
        profile_key = "D02"
    elif choice == "3":
        profile_key = "D03"

    profile = dummy_profiles[profile_key]
    print(f"\n--- Running Prediction for: {profile['title']} ---\n")

    question_keys = ["qid_1", "qid_2", "qid_3", "qid_4", "qid_5"]
    question_scores = []
    aggregated_metrics = {k: 0.0 for k in FEATURE_NAMES}

    for idx, (qid, feat_dict) in enumerate(zip(question_keys, profile["questions"]), start=1):
        q_text = questions.get(qid, f"Question {idx}")
        # Build exact 18-element vector in canonical order
        vec = np.array([float(feat_dict[name]) for name in FEATURE_NAMES], dtype=np.float32)

        # Predict score using trained Keras neural network
        q_score = predict_score(model, vec)
        q_level = get_score_level(q_score)
        question_scores.append(q_score)

        for name in FEATURE_NAMES:
            aggregated_metrics[name] += float(feat_dict[name])

        print(f"Q{idx}: {q_text}")
        print(f"    ↳ Neural Network Predicted Score: {q_score:.2f} / 100  (Level: {q_level})\n")

    # Average metrics
    for name in FEATURE_NAMES:
        aggregated_metrics[name] /= len(question_keys)

    overall_score = compute_overall_score(question_scores)
    overall_level = get_score_level(overall_score)

    print("=" * 60)
    print(f"DEMO SUMMARY FOR CANDIDATE [{profile_key}]")
    print(f"  Q1 Score : {question_scores[0]:.2f}")
    print(f"  Q2 Score : {question_scores[1]:.2f}")
    print(f"  Q3 Score : {question_scores[2]:.2f}")
    print(f"  Q4 Score : {question_scores[3]:.2f}")
    print(f"  Q5 Score : {question_scores[4]:.2f}")
    print("-" * 40)
    print(f"  OVERALL NEURAL NETWORK SCORE: {overall_score:.2f} / 100  ({overall_level})")
    print("=" * 60)

    # Generate and display explainable feedback report
    report = generate_feedback_from_metrics(overall_score, overall_level, aggregated_metrics)
    print(report)

def run_manual_interview():
    """
    Option 2: Live Manual Interview.
    Collects audio from microphone (or typed text fallback),
    extracts 12 text NLP features + 5 acoustic features + speech_available,
    fuses into 18-D vector, predicts score via trained neural network,
    and saves to candidates.json and corpus.json.
    """
    print("\n" + "=" * 60)
    print("              OPTION 2: LIVE MANUAL INTERVIEW")
    print("=" * 60)

    # Load Neural Network Model
    try:
        model = load_trained_model()
    except Exception as e:
        print(f"[Error] Neural network unavailable: {e}")
        return

    # Load canonical datasets
    candidates_data = load_data("data/candidates.json")
    corpus_data = load_data("data/corpus.json")

    # Ensure root objects exist
    if "candidates" not in candidates_data:
        candidates_data = {"candidates": {}}
    if "corpus" not in corpus_data:
        corpus_data = {"corpus": {}}

    # Load question references
    ref_dict = questions_tfidf()

    candidate_id = input("\nEnter Candidate ID (e.g., '101' or Enter -1 to return): ").strip()
    if candidate_id == "-1" or not candidate_id:
        return

    print("\nSelect Response Modality:")
    print("  1. Speech Modality (Microphone -> Audio -> Faster-Whisper + Acoustics)")
    print("  2. Text Modality Fallback (Typed Text)")
    mode_choice = input("Enter choice (1/2, default 1): ").strip()
    speech_mode = (mode_choice != "2")

    if speech_mode:
        print("\n[Modality] Speech Modality Active: Capturing raw audio and Whisper transcript.")
    else:
        print("\n[Modality] Text Modality Fallback Active: Capturing typed text.")

    # Initialize Candidate in candidates.json
    if candidate_id not in candidates_data["candidates"]:
        candidates_data["candidates"][candidate_id] = {
            "metadata": {"candidate_id": candidate_id, "created_at": None},
            "responses": {}
        }

    # Initialize Candidate in corpus.json
    if candidate_id not in corpus_data["corpus"]:
        corpus_data["corpus"][candidate_id] = {
            "metadata": {"candidate_id": candidate_id},
            "responses": {},
            "overall": {
                "text": {"features": {"statistics": {}, "sentiment": {}}},
                "speech": {"features": {}},
                "video": {"features": {}},  # Reserved video modality preserved
                "evaluation": {},
                "fusion": {}
            }
        }

    question_scores = []
    combined_answer_text = ""
    extension = ".wav"

    # Iterate through exactly the 5 canonical interview questions
    for q_idx, (qid, q_text) in enumerate(questions.items(), start=1):
        print("\n" + "-" * 50)
        print(f"Question {q_idx} of 5: {q_text}")
        print("-" * 50)

        audio_path = None
        candidate_answer = ""
        speech_features = {
            "duration_seconds": 0.0,
            "speech_rate_wpm": 0.0,
            "rms_energy": 0.0,
            "zero_crossing_rate": 0.0,
            "silence_ratio": 0.0,
            "speech_available": 0.0
        }

        if speech_mode:
            audio_path = f"speech/audio/cid_{candidate_id}_{qid}{extension}"
            rec_ok = record(audio_path)

            if rec_ok:
                transcript = transcribe_audio(audio_path)
                candidate_answer = transcript.strip()
                if not candidate_answer:
                    print("[Note] Spoken response was silent or unclear. You may enter text manually:")
                    candidate_answer = input("Answer text: ").strip()

                # Extract 5 acoustic features from recorded audio
                speech_features = extract_acoustic_features(audio_path, candidate_answer)
            else:
                print("[Fallback] Microphone unavailable. Please type your answer:")
                candidate_answer = input("Answer text: ").strip()
        else:
            candidate_answer = input("\nType your answer: ").strip()
            if not candidate_answer:
                candidate_answer = "No answer provided."

        # Accumulate text for overall metrics
        space = " " if combined_answer_text.endswith(".") else ". "
        combined_answer_text += space + candidate_answer

        # Store in candidates.json safely
        candidates_data["candidates"][candidate_id]["responses"][qid] = {
            "text": {"answer": candidate_answer},
            "speech": {"audio_path": audio_path, "transcript": candidate_answer if speech_mode else None},
            "video": {"video_path": None}  # Preserve video field
        }

        # Text Modality Processing
        lemmas = preprocess(candidate_answer)
        bow = extract_features_bow(lemmas)
        stats = extract_features_statistics(candidate_answer)
        sentiment = extract_features_sentiment(candidate_answer)

        # Store in corpus.json response structure
        corpus_data["corpus"][candidate_id]["responses"][qid] = {
            "text": {
                "processed": {
                    "lemmas": lemmas,
                    "bow": bow,
                    "tfidf": {}
                },
                "features": {
                    "statistics": stats,
                    "sentiment": sentiment
                }
            },
            "speech": {
                "transcript": candidate_answer if speech_mode else "",
                "features": speech_features
            },
            "video": {
                "features": {}  # Preserve video field
            },
            "evaluation": {},
            "fusion": {}
        }

    # Compute TF-IDF across corpus for this candidate with shared IDF
    corpus_data["corpus"] = extract_features_tfidf(candidate_id, corpus_data["corpus"], ref_dict)

    # Compute diagnostic evaluations (TF-IDF similarity, keyword coverage, rule-based metric)
    corpus_data["corpus"] = evaluate_candidate(corpus_data["corpus"], ref_dict, candidate_id)

    # Question-level Neural Network Predictions & Feature Fusion
    print("\n" + "=" * 50)
    print("      COMPUTING NEURAL NETWORK ASSESSMENTS")
    print("=" * 50)

    for q_idx, qid in enumerate(questions.keys(), start=1):
        q_resp = corpus_data["corpus"][candidate_id]["responses"][qid]
        cand_tfidf = q_resp["text"]["processed"]["tfidf"]
        cand_lemmas = q_resp["text"]["processed"]["lemmas"]
        cand_raw_text = candidates_data["candidates"][candidate_id]["responses"][qid]["text"]["answer"]

        q_ref_tfidf = ref_dict["tfidf"].get(qid, {})
        q_ref_keywords = ref_dict["lemmas"].get(qid, [])

        # 1. Extract 12 Text Features
        text_feat_dict = get_text_feature_dict(
            cand_raw_text, cand_lemmas, cand_tfidf, q_ref_tfidf, q_ref_keywords
        )

        # 2. Extract Speech Features
        sp_feat_dict = q_resp["speech"]["features"]

        # 3. Explicit Feature Fusion (12 Text + 6 Speech = 18 Features)
        fused_vector = fuse_features(text_feat_dict, sp_feat_dict)

        # 4. Neural Network Regression Prediction
        q_score = predict_score(model, fused_vector)
        q_level = get_score_level(q_score)
        question_scores.append(q_score)

        # Store question-level fusion and neural network evaluation in corpus.json
        q_resp["fusion"] = {
            "feature_names": FEATURE_NAMES,
            "feature_vector": [round(float(v), 4) for v in fused_vector]
        }
        q_resp["evaluation"]["neural_network"] = {
            "score": q_score,
            "level": q_level
        }

        print(f"Q{q_idx} ({questions[qid]}):")
        print(f"   ↳ Neural Network Score: {q_score:.2f} / 100  ({q_level})")

    # Overall Score Calculation
    overall_score = compute_overall_score(question_scores)
    overall_level = get_score_level(overall_score)

    # Store overall evaluations in corpus.json
    overall_stats = extract_features_statistics(combined_answer_text)
    overall_sentiment = extract_features_sentiment(combined_answer_text)

    cand_corpus = corpus_data["corpus"][candidate_id]
    cand_corpus["overall"]["text"]["features"]["statistics"] = overall_stats
    cand_corpus["overall"]["text"]["features"]["sentiment"] = overall_sentiment

    # Overall speech aggregates
    speech_durations = [r["speech"]["features"].get("duration_seconds", 0.0) for r in cand_corpus["responses"].values()]
    speech_wpms = [r["speech"]["features"].get("speech_rate_wpm", 0.0) for r in cand_corpus["responses"].values()]
    speech_silences = [r["speech"]["features"].get("silence_ratio", 0.0) for r in cand_corpus["responses"].values()]
    speech_rms = [r["speech"]["features"].get("rms_energy", 0.0) for r in cand_corpus["responses"].values()]

    cand_corpus["overall"]["speech"]["features"] = {
        "duration_seconds": round(float(sum(speech_durations)), 2),
        "speech_rate_wpm": round(float(np.mean(speech_wpms)), 2) if speech_wpms else 0.0,
        "silence_ratio": round(float(np.mean(speech_silences)), 4) if speech_silences else 0.0,
        "rms_energy": round(float(np.mean(speech_rms)), 4) if speech_rms else 0.0,
        "speech_available": 1.0 if speech_mode else 0.0
    }

    # Store overall neural network assessment
    cand_corpus["overall"]["evaluation"]["neural_network"] = {
        "score": overall_score,
        "level": overall_level
    }

    # Save synchronized JSON files to disk
    save_data("data/candidates.json", candidates_data, "Saved Candidate Response Data")
    save_data("data/corpus.json", corpus_data, "Saved Candidate Corpus & Evaluation")

    print("\n" + "=" * 60)
    print(f"CANDIDATE {candidate_id} ASSESSMENT COMPLETE")
    print(f"Overall Neural Network Score: {overall_score:.2f} / 100 ({overall_level})")
    print("=" * 60)

    # Generate and display explainable feedback report
    feedback(candidate_id, corpus_data)

def run_train_pipeline():
    """
    Option 3: Regenerate synthetic dataset and train the Keras neural network.
    """
    print("\n" + "=" * 60)
    print("      OPTION 3: TRAIN / RETRAIN NEURAL NETWORK REGRESSOR")
    print("=" * 60)
    from ml.train import train_neural_network
    from ml.synthetic_data import generate_synthetic_dataset

    regen = input("Regenerate synthetic training dataset? (y/n, default: y): ").strip().lower()
    if regen != "n":
        generate_synthetic_dataset("data/ml_training_data.csv", num_samples=100, seed=42)

    epochs_input = input("Enter training epochs (default: 80): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 80

    res = train_neural_network(csv_path="data/ml_training_data.csv", epochs=epochs, batch_size=8)
    print("\n[Done] Model trained and saved successfully.")
    print(f"Artifacts: \n  Model: {res['model_path']}\n  Loss Plot: {res['plot_path']}")

def main():
    """Main CLI Menu Loop."""
    print("█" * 64)
    print("  MULTI-MODAL INTERVIEW ASSESSMENT SYSTEM")
    print("  Text & Speech Modalities | Keras Neural Network Regression")
    print("█" * 64)

    # Initialize reference questions on start
    try:
        questions_tfidf()
    except Exception as e:
        print(f"[Warning] Question reference initialization: {e}")

    while True:
        print("\n" + "=" * 50)
        print("                   MAIN MENU")
        print("=" * 50)
        print("  1. Dummy Data Prediction (Fast Demo, No Mic Needed)")
        print("  2. Manual Interview (Live Speech / Text Fallback)")
        print("  3. Train / Retrain Neural Network")
        print("  4. Exit")
        print("=" * 50)

        choice = input("Enter Choice (1-4): ").strip()

        if choice == "1":
            run_dummy_prediction()
        elif choice == "2":
            run_manual_interview()
        elif choice == "3":
            run_train_pipeline()
        elif choice == "4" or choice == "-1":
            print("\nExiting Multi-Modal Interview Assessment System. Goodbye!\n")
            break
        else:
            print("[Warning] Invalid selection. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()