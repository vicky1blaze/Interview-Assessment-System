"""
feature_extraction.py - NLP Feature Extraction and Diagnostic Evaluation.

Extracts text statistics, lexical metrics, sentiment scores, Bag-of-Words,
TF-IDF vectors, TF-IDF cosine similarity, and keyword coverage.
Provides the 12 numerical text features required by the neural network.
"""

from collections import Counter
import math
from string import punctuation
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    from text.preprocessing import remove_punctuation, stop_words, preprocess
except ImportError:
    from preprocessing import remove_punctuation, stop_words, preprocess

try:
    from text.filler_words import filler_words
except ImportError:
    from .filler_words import filler_words

# Ensure VADER lexicon is downloaded
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

def extract_features_statistics(text):
    """
    Extract lexical statistics from raw text.
    Handles empty/short text gracefully without division by zero.
    """
    if not text or not isinstance(text, str):
        return {
            "word_count": 0,
            "sentence_count": 0,
            "average_sentence_length": 0.0,
            "stop_words_count": 0,
            "filler_words_count": 0,
            "stopword_ratio": 0.0,
            "filler_ratio": 0.0,
            "vocabulary_count": 0,
            "vocabulary_ratio": 0.0,
            "vocabulary_score": 0.0,
            "no_punctuation_tokens": [],
            "no_punctuation_tokens_count": 0
        }

    text_lower = text.lower()
    sent_tokens = [s for s in sent_tokenize(text_lower) if s.strip()]
    word_tokens = word_tokenize(text_lower)

    # Filter out punctuation-only tokens
    clean_tokens = remove_punctuation(word_tokens)
    word_count = len(clean_tokens)
    sentence_count = max(1, len(sent_tokens)) if word_count > 0 else 0

    average_sentence_length = (word_count / sentence_count) if sentence_count > 0 else 0.0

    # Vocabulary diversity
    vocabulary = set(clean_tokens)
    vocabulary_count = len(vocabulary)
    vocabulary_ratio = (vocabulary_count / word_count) if word_count > 0 else 0.0

    # Filler and stop words
    filler_set = set(filler_words)
    filler_words_count = sum(1 for t in clean_tokens if t in filler_set)
    stop_words_count = sum(1 for t in clean_tokens if t in stop_words)

    filler_ratio = (filler_words_count / word_count) if word_count > 0 else 0.0
    stopword_ratio = (stop_words_count / word_count) if word_count > 0 else 0.0
    vocabulary_score = min(vocabulary_ratio, 1.0)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": round(average_sentence_length, 4),
        "stop_words_count": stop_words_count,
        "filler_words_count": filler_words_count,
        "stopword_ratio": round(stopword_ratio, 4),
        "filler_ratio": round(filler_ratio, 4),
        "vocabulary_count": vocabulary_count,
        "vocabulary_ratio": round(vocabulary_ratio, 4),
        "vocabulary_score": round(vocabulary_score, 4),
        "no_punctuation_tokens": clean_tokens,
        "no_punctuation_tokens_count": word_count
    }

def extract_features_sentiment(raw_text):
    """
    Extract VADER sentiment polarity scores:
    pos, neg, neu, compound.
    """
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    scores = sia.polarity_scores(raw_text)
    return {
        "neg": round(scores.get("neg", 0.0), 4),
        "neu": round(scores.get("neu", 1.0), 4),
        "pos": round(scores.get("pos", 0.0), 4),
        "compound": round(scores.get("compound", 0.0), 4)
    }

def extract_features_bow(lemma_list):
    """
    Compute Bag-of-Words frequency distribution from lemma list.
    """
    if not lemma_list:
        return {}
    return dict(Counter(lemma_list))

def extract_features_tfidf(candidate_id, corpus, questions=None):
    """
    Compute TF-IDF vectors for all responses of a candidate.
    Term Frequency (TF) is computed locally per response.
    Document Frequency (DF) and Inverse Document Frequency (IDF)
    are calculated on a shared document corpus encompassing both reference
    question texts and candidate responses, guaranteeing a consistent IDF baseline.
    """
    tf = {}
    df = {}
    doc_overall_count = 0

    candidate_responses = corpus.get(candidate_id, {}).get("responses", {})

    # Compute TF locally for this candidate's responses
    for qid, q_data in candidate_responses.items():
        processed = q_data.get("text", {}).get("processed", {})
        bow = processed.get("bow", {})
        total_terms = sum(bow.values())
        tf[qid] = {}
        if total_terms > 0:
            for term, count in bow.items():
                tf[qid][term] = count / total_terms

    # 1. Include reference questions in document frequency count
    q_lemmas_map = {}
    if questions is not None:
        if "lemmas" in questions:
            q_lemmas_map = questions["lemmas"]
        elif "questions" in questions and isinstance(questions["questions"], dict):
            for qk, qv in questions["questions"].items():
                if isinstance(qv, dict) and "reference" in qv:
                    q_lemmas_map[qk] = qv["reference"].get("lemmas", [])
                elif isinstance(qv, str):
                    q_lemmas_map[qk] = preprocess(qv)
        elif isinstance(questions, dict):
            for qk, qv in questions.items():
                if isinstance(qv, str):
                    q_lemmas_map[qk] = preprocess(qv)

    # Fallback to standard canonical questions if none provided
    if not q_lemmas_map:
        try:
            from core.questions import QUESTIONS_META
            for qk, qv in QUESTIONS_META.items():
                q_lemmas_map[qk] = preprocess(qv["question"])
        except Exception:
            try:
                from questions import QUESTIONS_META
                for qk, qv in QUESTIONS_META.items():
                    q_lemmas_map[qk] = preprocess(qv["question"])
            except Exception:
                pass

    for q_lemmas in q_lemmas_map.values():
        if q_lemmas:
            doc_overall_count += 1
            for token in set(q_lemmas):
                df[token] = df.get(token, 0) + 1

    # 2. Include all candidate responses across corpus in DF count
    for c_id, c_data in corpus.items():
        for resp in c_data.get("responses", {}).values():
            lemmas = resp.get("text", {}).get("processed", {}).get("lemmas", [])
            if lemmas:
                doc_overall_count += 1
                for token in set(lemmas):
                    df[token] = df.get(token, 0) + 1

    # Safe document count baseline
    doc_overall_count = max(doc_overall_count, 1)

    # Compute shared IDF
    idf = {}
    for term, term_df in df.items():
        idf[term] = math.log10(doc_overall_count / term_df) if term_df > 0 else 0.0

    # Compute candidate TF-IDF using shared IDF
    for qid in tf.keys():
        if "text" not in candidate_responses[qid]:
            candidate_responses[qid]["text"] = {"processed": {}, "features": {}}
        if "processed" not in candidate_responses[qid]["text"]:
            candidate_responses[qid]["text"]["processed"] = {}

        candidate_responses[qid]["text"]["processed"]["tfidf"] = {}
        for term, term_tf in tf[qid].items():
            term_idf = idf.get(term, 0.0)
            candidate_responses[qid]["text"]["processed"]["tfidf"][term] = round(term_tf * term_idf, 4)

    # If reference questions dict was passed with a 'tfidf' dict, update question tfidf with shared IDF too
    if questions is not None and isinstance(questions, dict) and "tfidf" in questions:
        for qk, q_lemmas in q_lemmas_map.items():
            if q_lemmas:
                q_bow = Counter(q_lemmas)
                q_total = sum(q_bow.values())
                questions["tfidf"][qk] = {}
                for term, cnt in q_bow.items():
                    q_tf = cnt / q_total if q_total > 0 else 0.0
                    questions["tfidf"][qk][term] = round(q_tf * idf.get(term, 0.0), 4)

    return corpus

def cosine_similarity(vec1, vec2):
    """
    Compute TF-IDF cosine similarity between two word-weight dictionaries.
    """
    if not vec1 or not vec2:
        return 0.0

    all_words = set(vec1.keys()).union(set(vec2.keys()))
    v1 = [vec1.get(w, 0.0) for w in all_words]
    v2 = [vec2.get(w, 0.0) for w in all_words]

    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    sim = dot_product / (mag1 * mag2)
    return max(0.0, min(1.0, round(sim, 4)))

def compute_coverage(question_tokens, answer_tokens):
    """
    Compute keyword coverage (lexical overlap ratio) between
    question/reference keywords and candidate response lemmas.
    """
    if not question_tokens:
        return {"score": 0.0, "matched": [], "missing": []}

    q_set = set(question_tokens)
    a_set = set(answer_tokens) if answer_tokens else set()

    matched = q_set.intersection(a_set)
    missing = q_set - a_set
    score = len(matched) / len(q_set) if len(q_set) > 0 else 0.0

    return {
        "score": round(score, 4),
        "matched": sorted(list(matched)),
        "missing": sorted(list(missing))
    }

def get_level(score):
    """Categorize a 0-1 scale score into Low / Moderate / High."""
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Moderate"
    else:
        return "Low"

def compute_final_score(similarity, coverage, w_sim=0.6, w_cov=0.4):
    """
    Compute the legacy rule-based diagnostic score (0.6*sim + 0.4*cov).
    Preserved strictly for explainability and diagnostics, NOT used as final NN score.
    """
    final = (w_sim * similarity) + (w_cov * coverage)
    return {
        "score": round(final, 4),
        "level": get_level(final)
    }

def evaluate_candidate(corpus, questions, candidate_id, w_sim=0.6, w_cov=0.4):
    """
    Evaluate candidate responses against reference questions using:
    - TF-IDF Cosine Similarity
    - Lexical Keyword Coverage
    - Rule-based Combined Score (diagnostic)

    Stores diagnostics in:
    corpus[candidate_id]["responses"][qid]["evaluation"]["semantic|lexical|rule_based"]
    """
    sim_scores = []
    cov_scores = []
    final_scores = []

    candidate_data = corpus.get(candidate_id, {})
    responses = candidate_data.get("responses", {})

    # Ensure candidate responses and reference questions share the same IDF basis
    extract_features_tfidf(candidate_id, corpus, questions)

    for qid in questions.get("tfidf", {}).keys():
        if qid not in responses:
            responses[qid] = {
                "text": {"answer": "", "processed": {}, "features": {}},
                "speech": {"audio_path": None, "transcript": None, "features": {}},
                "video": {"video_path": None, "features": {}},
                "evaluation": {}
            }

        q_resp = responses[qid]
        if "evaluation" not in q_resp:
            q_resp["evaluation"] = {}

        answer_vec = q_resp.get("text", {}).get("processed", {}).get("tfidf", {})
        question_vec = questions["tfidf"].get(qid, {})

        answer_tokens = q_resp.get("text", {}).get("processed", {}).get("lemmas", [])
        question_tokens = questions.get("lemmas", {}).get(qid, [])

        # 1. TF-IDF Cosine Similarity
        sim = cosine_similarity(question_vec, answer_vec)
        sim_scores.append(sim)
        q_resp["evaluation"]["semantic"] = {
            "similarity_score": {
                "score": sim,
                "level": get_level(sim)
            }
        }

        # 2. Keyword Coverage
        cov_data = compute_coverage(question_tokens, answer_tokens)
        cov_scores.append(cov_data["score"])
        q_resp["evaluation"]["lexical"] = {
            "keyword_coverage": cov_data
        }

        # 3. Rule-based Diagnostic Metric
        diag_score = compute_final_score(sim, cov_data["score"], w_sim, w_cov)
        final_scores.append(diag_score["score"])
        q_resp["evaluation"]["rule_based"] = {
            "combined_score": diag_score
        }

    def safe_avg(arr):
        return round(sum(arr) / len(arr), 4) if arr else 0.0

    # Store overall diagnostic aggregates
    if "overall" not in candidate_data:
        candidate_data["overall"] = {
            "text": {"features": {"statistics": {}, "sentiment": {}}},
            "speech": {"features": {}},
            "video": {"features": {}},
            "evaluation": {},
            "fusion": {}
        }

    candidate_data["overall"]["evaluation"]["semantic"] = {
        "similarity_score": {"score": safe_avg(sim_scores), "level": get_level(safe_avg(sim_scores))}
    }
    candidate_data["overall"]["evaluation"]["lexical"] = {
        "keyword_coverage": {"score": safe_avg(cov_scores), "level": get_level(safe_avg(cov_scores))}
    }
    candidate_data["overall"]["evaluation"]["rule_based"] = {
        "combined_score": {"score": safe_avg(final_scores), "level": get_level(safe_avg(final_scores))}
    }

    return corpus

def get_text_feature_dict(raw_text, lemmas, candidate_tfidf, question_tfidf, question_keywords):
    """
    Extract the 12 numerical text features required for neural network regression:
    1. word_count
    2. sentence_count
    3. average_sentence_length
    4. vocabulary_ratio
    5. filler_ratio
    6. stopword_ratio
    7. sentiment_pos
    8. sentiment_neg
    9. sentiment_neu
    10. sentiment_compound
    11. keyword_coverage
    12. tfidf_cosine_similarity
    """
    stats = extract_features_statistics(raw_text)
    sentiment = extract_features_sentiment(raw_text)
    coverage = compute_coverage(question_keywords, lemmas)
    similarity = cosine_similarity(question_tfidf, candidate_tfidf)

    return {
        "word_count": float(stats["word_count"]),
        "sentence_count": float(stats["sentence_count"]),
        "average_sentence_length": float(stats["average_sentence_length"]),
        "vocabulary_ratio": float(stats["vocabulary_ratio"]),
        "filler_ratio": float(stats["filler_ratio"]),
        "stopword_ratio": float(stats["stopword_ratio"]),
        "sentiment_pos": float(sentiment["pos"]),
        "sentiment_neg": float(sentiment["neg"]),
        "sentiment_neu": float(sentiment["neu"]),
        "sentiment_compound": float(sentiment["compound"]),
        "keyword_coverage": float(coverage["score"]),
        "tfidf_cosine_similarity": float(similarity)
    }
