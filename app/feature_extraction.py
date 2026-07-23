from nltk import word_tokenize, sent_tokenize
from preprocessing import remove_punctuation, stop_words
from filler_words import filler_words
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import math

sia = SentimentIntensityAnalyzer()

def extract_features_statistics(raw_text):
    text = raw_text.lower()

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
        "vocabulary": vocabulary,
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
    doc_count = len(corpus[candidate_id]["bow"].keys())
    tf = {}
    terms_counts = {}
    idf = {}

    for doc in corpus[candidate_id]["bow"].keys():
        total_terms = sum(corpus[candidate_id]["bow"][doc].values())

        if doc not in tf:
            tf[doc] =  {}

        for term, term_frequency in corpus[candidate_id]["bow"][doc].items():
            tf_of_term = term_frequency / total_terms
            tf[doc][term] = tf_of_term

            terms_counts[term] = terms_counts.get(term, 0) + 1

    for term, term_frequency in terms_counts.items():
        idf_score = math.log10(doc_count / term_frequency)

        idf[term] = idf_score

    for doc in tf.keys():
        for term, term_tf in tf[doc].items():
            tfidf = round(term_tf * idf[term], 4)

            if doc not in corpus[candidate_id]["tfidf"]:
                corpus[candidate_id]["tfidf"][doc] = {}
                
            corpus[candidate_id]["tfidf"][doc][term] = tfidf

    return corpus