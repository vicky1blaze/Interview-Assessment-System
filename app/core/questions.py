"""
questions.py - Interview question management and reference NLP processing.

Maintains the exact 5 canonical interview questions and computes reference
lemmas, Bag-of-Words (BoW), and TF-IDF vectors used as baseline features
for evaluating candidate responses.
"""

from collections import Counter
import math
from text.preprocessing import preprocess

try:
    from core.dataset import load_data, save_data
except ImportError:
    from dataset import load_data, save_data

# The exact 5 canonical interview questions and their metadata
QUESTIONS_META = {
    "qid_1": {
        "question": "Tell me about yourself and your background.",
        "category": "Introduction",
        "difficulty": "Easy"
    },
    "qid_2": {
        "question": "What are your key strengths?",
        "category": "Self-Assessment",
        "difficulty": "Easy"
    },
    "qid_3": {
        "question": "Describe a technical problem you solved and how you solved it.",
        "category": "Technical",
        "difficulty": "Medium"
    },
    "qid_4": {
        "question": "Why are you interested in this role?",
        "category": "Motivation",
        "difficulty": "Easy"
    },
    "qid_5": {
        "question": "How do you handle pressure or a tight deadline?",
        "category": "Behavioral",
        "difficulty": "Medium"
    }
}

# Standard dictionary mapping question_id -> question string
questions = {qid: data["question"] for qid, data in QUESTIONS_META.items()}

def questions_bow(token_list):
    """
    Compute Bag-of-Words word frequency distribution for reference tokens.
    """
    return dict(Counter(token_list))

def compute_tf_idf(tokens):
    """
    Compute Term Frequency - Inverse Document Frequency (TF-IDF) across
    all 5 reference interview questions.
    """
    tf = {}
    df = {}
    tokens["tfidf"] = {}
    doc_count = len(tokens["questions"].keys())

    # 1. Compute Local Term Frequency (TF)
    for qid, bow in tokens["bow"].items():
        question_length = sum(bow.values())
        tf[qid] = {}
        for term, freq in bow.items():
            tf[qid][term] = (freq / question_length) if question_length > 0 else 0.0
            df[term] = df.get(term, 0) + 1

    # 2. Compute Global Inverse Document Frequency (IDF) and TF-IDF
    idf = {}
    for term, freq in df.items():
        # Standard smoothed IDF
        idf[term] = math.log10(doc_count / freq) if freq > 0 else 0.0

    for qid in tf.keys():
        tokens["tfidf"][qid] = {}
        for term, tf_val in tf[qid].items():
            tokens["tfidf"][qid][term] = round(tf_val * idf.get(term, 0.0), 4)

    return tokens

def questions_tfidf():
    """
    Build data/questions.json in the canonical hierarchy:
    {
      "questions": {
        "qid_1": {
          "metadata": {
            "question_id": "qid_1",
            "category": "Introduction",
            "difficulty": "Easy"
          },
          "question": "...",
          "reference": {
            "lemmas": [...],
            "bow": {...},
            "tfidf": {...},
            "keywords": [...]
          }
        },
        ...
      }
    }
    """
    flat_tokens = {
        "questions": {},
        "lemmas": {},
        "bow": {},
        "tfidf": {}
    }

    for qid, q_text in questions.items():
        flat_tokens["questions"][qid] = q_text
        lemmas = preprocess(q_text)
        flat_tokens["lemmas"][qid] = lemmas
        flat_tokens["bow"][qid] = questions_bow(lemmas)
        flat_tokens["tfidf"][qid] = {}

    flat_tokens = compute_tf_idf(flat_tokens)

    canonical_questions = {
        "questions": {}
    }

    for qid, q_text in questions.items():
        meta = QUESTIONS_META[qid]
        lemmas = flat_tokens["lemmas"][qid]
        canonical_questions["questions"][qid] = {
            "metadata": {
                "question_id": qid,
                "category": meta["category"],
                "difficulty": meta["difficulty"]
            },
            "question": q_text,
            "reference": {
                "lemmas": lemmas,
                "bow": flat_tokens["bow"][qid],
                "tfidf": flat_tokens["tfidf"][qid],
                "keywords": lemmas
            }
        }

    save_data("data/questions.json", canonical_questions, "Canonical 5-question dataset loaded")

    return {
        "questions": canonical_questions["questions"],
        "lemmas": flat_tokens["lemmas"],
        "bow": flat_tokens["bow"],
        "tfidf": flat_tokens["tfidf"]
    }
