from nltk import word_tokenize, sent_tokenize
from text.preprocessing import remove_punctuation, stop_words
from filler_words import filler_words
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import math, numpy as np

sia = SentimentIntensityAnalyzer()

def extract_features_statistics(candidates_data, candidate_id):
    text = candidates_data[candidate_id]["text"]["combined_answer"].lower()

    sent_tokens = sent_tokenize(text)
    word_tokens = word_tokenize(text)

    word_count = len(word_tokens)
    sentence_count = len(sent_tokens)

    average_sentence_length = (word_count / sentence_count if sentence_count else 0)

    #Vocubalry

    no_punctuation_tokens = remove_punctuation(word_tokens)
    no_punctuation_tokens_count = len(no_punctuation_tokens)

    vocabulary = set(no_punctuation_tokens)
    vocabulary_count = len(vocabulary)

    vocabulary_ratio = (vocabulary_count / no_punctuation_tokens_count if no_punctuation_tokens_count else 0)

    #Filler and stop words count

    filler_words_count = 0
    stop_words_count = 0

    for token in word_tokens:
        if token in filler_words:
            filler_words_count += 1
            
        if token in stop_words:
            stop_words_count += 1

    filler_ratio = (filler_words_count / word_count if word_count else 0) 
    stopword_ratio = (stop_words_count / word_count if word_count else 0)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "stop_words_count": stop_words_count,
        "filler_words_count": filler_words_count,
        "stopword_ratio": stopword_ratio,
        "filler_ratio": filler_ratio,
        # "vocabulary": vocabulary,
        "vocabulary_count": vocabulary_count,
        "vocabulary_ratio": vocabulary_ratio,
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
    tf = {}
    idf = {}
    df = {}
    doc_overall_count = 0

    # Comput TF (Locally) ===========================================================

    for doc in corpus[candidate_id]["bow"].keys():
        total_terms = sum(corpus[candidate_id]["bow"][doc].values())

        if doc not in tf:
            tf[doc] =  {}

        for term, term_frequency in corpus[candidate_id]["bow"][doc].items():
            tf_of_term = term_frequency / total_terms
            tf[doc][term] = tf_of_term

    # Compute IDF (Globally) ========================================================

    for candidate in corpus.values():
        for doc_tokens in candidate["lemmas"].values():
            doc_overall_count += 1

            unique_tokens = set(doc_tokens)

            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

    for token, freq in df.items():
        idf_score = round(math.log10(doc_overall_count / freq), 4)

        idf[token] = idf.get(token, 0) + idf_score

    # Compute TF-IDF ================================================================

    for doc in tf.keys():
        for term, term_tf in tf[doc].items():
            tfidf = round(term_tf * idf[term], 4)

            if doc not in corpus[candidate_id]["tfidf"]:
                corpus[candidate_id]["tfidf"][doc] = {}
                
            corpus[candidate_id]["tfidf"][doc][term] = tfidf

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

    # ===== Initialize structure =====
    corpus[candidate_id]["lexical_semantic_score"] = {
        "similarity": {
            "question_wise": {},
            "overall": {}
        },
        "coverage": {
            "question_wise": {},
            "overall": {}
        },
        "final": {
            "question_wise": {},
            "overall": {}
        }
    }

    sim_scores = []
    cov_scores = []
    final_scores = []

    # ===== Loop through each question =====
    for qid in questions["tfidf"].keys():

        # ---- Fetch vectors ----
        answer_vec = corpus[candidate_id]["tfidf"].get(qid, {})
        question_vec = questions["tfidf"][qid]

        # ---- Fetch tokens (IMPORTANT) ----
        answer_tokens = corpus[candidate_id]["lemmas"].get(qid, [])
        question_tokens = questions["lemmas"][qid]

        # ===== 1. Cosine Similarity =====
        sim = cosine_similarity(question_vec, answer_vec)
        sim_scores.append(sim)

        corpus[candidate_id]["lexical_semantic_score"]["similarity"]["question_wise"][qid] = {
            "score": round(sim, 4),
            "level": get_level(sim)
        }

        # ===== 2. Coverage =====
        coverage_data = compute_coverage(question_tokens, answer_tokens)
        cov = coverage_data["score"]
        cov_scores.append(cov)

        corpus[candidate_id]["lexical_semantic_score"]["coverage"]["question_wise"][qid] = coverage_data

        # ===== 3. Final Score =====
        final_data = compute_final_score(sim, cov, w_sim, w_cov)
        final_scores.append(final_data["score"])

        corpus[candidate_id]["lexical_semantic_score"]["final"]["question_wise"][qid] = final_data

    # ===== OVERALL CALCULATIONS =====

    def safe_avg(arr):
        return sum(arr) / len(arr) if arr else 0.0

    avg_sim = safe_avg(sim_scores)
    avg_cov = safe_avg(cov_scores)
    avg_final = safe_avg(final_scores)

    corpus[candidate_id]["lexical_semantic_score"]["similarity"]["overall"] = {
        "score": round(avg_sim, 4),
        "level": get_level(avg_sim)
    }

    corpus[candidate_id]["lexical_semantic_score"]["coverage"]["overall"] = {
        "score": round(avg_cov, 4),
        "level": get_level(avg_cov)
    }

    corpus[candidate_id]["lexical_semantic_score"]["final"]["overall"] = {
        "score": round(avg_final, 4),
        "level": get_level(avg_final)
    }

    return corpus


