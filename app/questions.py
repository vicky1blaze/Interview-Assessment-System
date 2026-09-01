from text.preprocessing import preprocess
from collections import Counter
import math
from dataset import load_data, save_data

questions = {
    "qid_1" : "What is your machine learning full name?",
    "qid_2": "Where are you from machine learning?",
    "qid_3": "What do you like to do in your full free time?",
    "qid_4": "What is Machine learning?"
}

def questions_bow(token):
    bow = Counter(token)

    return dict(bow) 

def compute_tf_idf(tokens):
    tf = {}
    df = {}
    idf = {}
    tokens_tf_idf = {}
    doc_count = len(tokens["questions"].keys())

    # Compute TF (Locally) =========================================================

    for qid, question in tokens["bow"].items():
        question_length = len(question)

        if qid not in tf:
            tf[qid] = {}

        for term, freq in question.items():
            tf[qid][term] = (freq / question_length if question_length else 0)

            df[term] = df.get(term, 0) + 1

    # Compute IDF (Globally) =======================================================

    for term, freq in df.items():
        idf[term] = math.log10(doc_count / freq)

    for qid in tf.keys():
        for term, freq in tf[qid].items():
            
            tokens["tfidf"][qid][term] = round(tf[qid][term] * idf[term], 4)

    return tokens

def questions_tfidf():
    """
    Build questions.json in the new canonical hierarchy:
    {
      "questions": {
        "qid_1": {
          "metadata": {...},
          "question": "...",
          "reference": {
            "lemmas": [...],
            "bow": {...},
            "tfidf": {...},
            "keywords": [...]
          }
        }
      }
    }
    """
    # First, build the flat structure for TF-IDF computation
    flat_tokens = {
        "questions": {},
        "lemmas": {},
        "bow": {},
        "tfidf": {} 
    }

    for qid, question in questions.items():
        flat_tokens["questions"][qid] = question
        flat_tokens["lemmas"][qid] = preprocess(question)
        flat_tokens["bow"][qid] = questions_bow(flat_tokens["lemmas"][qid])
        flat_tokens["tfidf"][qid] = {}

    # Compute TF-IDF
    flat_tokens = compute_tf_idf(flat_tokens)

    # Now transform into the new canonical hierarchy
    canonical_questions = {
        "questions": {}
    }

    for qid, question in questions.items():
        canonical_questions["questions"][qid] = {
            "metadata": {
                "question_id": qid,
                "category": None,
                "difficulty": None
            },
            "question": question,
            "reference": {
                "lemmas": flat_tokens["lemmas"][qid],
                "bow": flat_tokens["bow"][qid],
                "tfidf": flat_tokens["tfidf"][qid],
                "keywords": flat_tokens["lemmas"][qid]  # keywords are the processed lemmas
            }
        }

    save_data("data/questions.json", canonical_questions, "Question set loaded")

    # Return a structure that maintains backward compatibility for internal use
    return {
        "questions": canonical_questions["questions"],
        "lemmas": flat_tokens["lemmas"],
        "bow": flat_tokens["bow"],
        "tfidf": flat_tokens["tfidf"]
    }