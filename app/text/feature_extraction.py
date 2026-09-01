from nltk import word_tokenize, sent_tokenize
from text.preprocessing import remove_punctuation, stop_words
from filler_words import filler_words
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import math, numpy as np

sia = SentimentIntensityAnalyzer()

def extract_features_statistics(text):
    """
    Extract statistics from raw text.
    Returns a dictionary of statistics features.
    """
    text = text.lower()

    sent_tokens = sent_tokenize(text)
    word_tokens = word_tokenize(text)

    word_count = len(word_tokens)
    sentence_count = len(sent_tokens)

    average_sentence_length = (word_count / sentence_count if sentence_count else 0)

    # Vocabulary

    no_punctuation_tokens = remove_punctuation(word_tokens)
    no_punctuation_tokens_count = len(no_punctuation_tokens)

    vocabulary = set(no_punctuation_tokens)
    vocabulary_count = len(vocabulary)

    vocabulary_ratio = (vocabulary_count / no_punctuation_tokens_count if no_punctuation_tokens_count else 0)

    # Filler and stop words count

    filler_words_count = 0
    stop_words_count = 0

    for token in word_tokens:
        if token in filler_words:
            filler_words_count += 1
            
        if token in stop_words:
            stop_words_count += 1

    filler_ratio = (filler_words_count / word_count if word_count else 0) 
    stopword_ratio = (stop_words_count / word_count if word_count else 0)

    # Vocabulary score: combination of unique vocabulary and word count
    vocabulary_score = min(vocabulary_ratio, 1.0)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "stop_words_count": stop_words_count,
        "filler_words_count": filler_words_count,
        "stopword_ratio": stopword_ratio,
        "filler_ratio": filler_ratio,
        "vocabulary_count": vocabulary_count,
        "vocabulary_ratio": vocabulary_ratio,
        "vocabulary_score": vocabulary_score,
        "no_punctuation_tokens": no_punctuation_tokens,
        "no_punctuation_tokens_count": no_punctuation_tokens_count
    }

def extract_features_sentiment(raw_text):
    sentiment_score = sia.polarity_scores(raw_text)

    return sentiment_score

def extract_features_bow(lemma):
    bow = Counter(lemma)

    return dict(bow)

def extract_features_tfidf(candidate_id, corpus):
    """
    Compute TF-IDF for all answers of a candidate.
    Stores results in: corpus[candidate_id]["responses"][qid]["text"]["processed"]["tfidf"]
    """
    tf = {}
    idf = {}
    df = {}
    doc_overall_count = 0

    # Collect all lemmas for this candidate
    candidate_lemmas = {}
    for qid in corpus[candidate_id]["responses"].keys():
        if "text" in corpus[candidate_id]["responses"][qid]:
            if "processed" in corpus[candidate_id]["responses"][qid]["text"]:
                if "lemmas" in corpus[candidate_id]["responses"][qid]["text"]["processed"]:
                    candidate_lemmas[qid] = corpus[candidate_id]["responses"][qid]["text"]["processed"]["lemmas"]

    # Compute TF (Locally) ===========================================================

    for qid, lemmas in candidate_lemmas.items():
        bow = corpus[candidate_id]["responses"][qid]["text"]["processed"]["bow"]
        total_terms = sum(bow.values())

        if qid not in tf:
            tf[qid] = {}

        for term, term_frequency in bow.items():
            tf_of_term = term_frequency / total_terms
            tf[qid][term] = tf_of_term

    # Compute IDF (Globally) ========================================================

    for candidate in corpus.values():
        if "responses" in candidate:
            for qid_data in candidate["responses"].values():
                if "text" in qid_data and "processed" in qid_data["text"]:
                    if "lemmas" in qid_data["text"]["processed"]:
                        doc_overall_count += 1
                        unique_tokens = set(qid_data["text"]["processed"]["lemmas"])
                        for token in unique_tokens:
                            df[token] = df.get(token, 0) + 1

    for token, freq in df.items():
        idf_score = round(math.log10(doc_overall_count / freq) if freq > 0 else 0, 4)
        idf[token] = idf.get(token, 0) + idf_score

    # Compute TF-IDF ================================================================

    for qid in tf.keys():
        if qid not in corpus[candidate_id]["responses"]:
            continue
        if "text" not in corpus[candidate_id]["responses"][qid]:
            continue
        if "processed" not in corpus[candidate_id]["responses"][qid]["text"]:
            corpus[candidate_id]["responses"][qid]["text"]["processed"] = {}
        
        corpus[candidate_id]["responses"][qid]["text"]["processed"]["tfidf"] = {}
        
        for term, term_tf in tf[qid].items():
            tfidf = round(term_tf * idf.get(term, 0), 4)
            corpus[candidate_id]["responses"][qid]["text"]["processed"]["tfidf"][term] = tfidf

    return corpus

def cosine_similarity(vec1, vec2):

    # Step 1: Get all unique words
    all_words = set(vec1.keys()).union(set(vec2.keys()))
    
    # Step 2: Create aligned vectors
    v1 = []
    v2 = []
    
    for word in all_words:
        v1.append(vec1.get(word, 0))
        v2.append(vec2.get(word, 0))
    
    # Step 3: Dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Step 4: Magnitudes
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    
    # Step 5: Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Step 6: Cosine similarity
    return dot_product / (mag1 * mag2)


def compute_coverage(question_tokens, answer_tokens):
    if not question_tokens:
        return {
            "score": 0.0,
            "matched": [],
            "missing": []
        }

    q_set = set(question_tokens)
    a_set = set(answer_tokens)

    matched = q_set.intersection(a_set)
    missing = q_set - a_set

    score = len(matched) / len(q_set)

    return {
        "score": round(score, 4),
        "matched": list(matched),
        "missing": list(missing)
    }

def get_level(score):
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Moderate"
    else:
        return "Low"

def compute_final_score(similarity, coverage, w_sim=0.6, w_cov=0.4):
    final = (w_sim * similarity) + (w_cov * coverage)

    return {
        "score": round(final, 4),
        "level": get_level(final)
    }

def evaluate_candidate(corpus, questions, candidate_id, w_sim=0.6, w_cov=0.4):
    """
    Evaluate a candidate's responses against questions using:
    - Semantic similarity (cosine similarity)
    - Lexical coverage (keyword matching)
    - Rule-based combined score (weighted average)

    Stores results in the new hierarchy:
    corpus[candidate_id]["responses"][qid]["evaluation"]["semantic"]["similarity_score"]
    corpus[candidate_id]["responses"][qid]["evaluation"]["lexical"]["keyword_coverage"]
    corpus[candidate_id]["responses"][qid]["evaluation"]["rule_based"]["combined_score"]

    Also stores overall aggregates in:
    corpus[candidate_id]["overall"]["evaluation"]["semantic|lexical|rule_based"]
    """

    sim_scores = []
    cov_scores = []
    final_scores = []

    # ===== Loop through each question in the question bank =====
    for qid in questions["tfidf"].keys():
        # Initialize evaluation structure for this QID if not exists
        if qid not in corpus[candidate_id]["responses"]:
            corpus[candidate_id]["responses"][qid] = {
                "text": {"answer": "", "processed": {}, "features": {}},
                "speech": {"audio_path": None, "transcript": None, "features": {}},
                "video": {"video_path": None, "features": {}},
                "evaluation": {}
            }

        if "evaluation" not in corpus[candidate_id]["responses"][qid]:
            corpus[candidate_id]["responses"][qid]["evaluation"] = {}

        # ---- Fetch vectors ----
        answer_vec = {}
        if "text" in corpus[candidate_id]["responses"][qid]:
            if "processed" in corpus[candidate_id]["responses"][qid]["text"]:
                answer_vec = corpus[candidate_id]["responses"][qid]["text"]["processed"].get("tfidf", {})

        question_vec = questions["tfidf"][qid]

        # ---- Fetch tokens (IMPORTANT) ----
        answer_tokens = []
        if "text" in corpus[candidate_id]["responses"][qid]:
            if "processed" in corpus[candidate_id]["responses"][qid]["text"]:
                answer_tokens = corpus[candidate_id]["responses"][qid]["text"]["processed"].get("lemmas", [])

        question_tokens = questions["lemmas"][qid]

        # ===== 1. Semantic: Cosine Similarity =====
        sim = cosine_similarity(question_vec, answer_vec)
        sim_scores.append(sim)

        if "semantic" not in corpus[candidate_id]["responses"][qid]["evaluation"]:
            corpus[candidate_id]["responses"][qid]["evaluation"]["semantic"] = {}

        corpus[candidate_id]["responses"][qid]["evaluation"]["semantic"]["similarity_score"] = {
            "score": round(sim, 4),
            "level": get_level(sim)
        }

        # ===== 2. Lexical: Keyword Coverage =====
        coverage_data = compute_coverage(question_tokens, answer_tokens)
        cov = coverage_data["score"]
        cov_scores.append(cov)

        if "lexical" not in corpus[candidate_id]["responses"][qid]["evaluation"]:
            corpus[candidate_id]["responses"][qid]["evaluation"]["lexical"] = {}

        corpus[candidate_id]["responses"][qid]["evaluation"]["lexical"]["keyword_coverage"] = coverage_data

        # ===== 3. Rule-based: Combined Score =====
        final_data = compute_final_score(sim, cov, w_sim, w_cov)
        final_scores.append(final_data["score"])

        if "rule_based" not in corpus[candidate_id]["responses"][qid]["evaluation"]:
            corpus[candidate_id]["responses"][qid]["evaluation"]["rule_based"] = {}

        corpus[candidate_id]["responses"][qid]["evaluation"]["rule_based"]["combined_score"] = final_data

    # ===== OVERALL CALCULATIONS =====

    def safe_avg(arr):
        return sum(arr) / len(arr) if arr else 0.0

    avg_sim = safe_avg(sim_scores)
    avg_cov = safe_avg(cov_scores)
    avg_final = safe_avg(final_scores)

    # Initialize overall evaluation structure
    if "overall" not in corpus[candidate_id]:
        corpus[candidate_id]["overall"] = {
            "text": {"features": {"statistics": {}, "sentiment": {}}},
            "speech": {"features": {}},
            "video": {"features": {}},
            "evaluation": {"semantic": {}, "lexical": {}, "rule_based": {}},
            "fusion": {"features": {}}
        }

    if "evaluation" not in corpus[candidate_id]["overall"]:
        corpus[candidate_id]["overall"]["evaluation"] = {
            "semantic": {},
            "lexical": {},
            "rule_based": {}
        }

    # Store overall evaluation scores
    corpus[candidate_id]["overall"]["evaluation"]["semantic"] = {
        "similarity_score": {
            "score": round(avg_sim, 4),
            "level": get_level(avg_sim)
        }
    }

    corpus[candidate_id]["overall"]["evaluation"]["lexical"] = {
        "keyword_coverage": {
            "score": round(avg_cov, 4),
            "level": get_level(avg_cov)
        }
    }

    corpus[candidate_id]["overall"]["evaluation"]["rule_based"] = {
        "combined_score": {
            "score": round(avg_final, 4),
            "level": get_level(avg_final)
        }
    }

    return corpus


