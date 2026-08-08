from text.preprocessing import preprocess
from collections import Counter
import math

questions = {
    "qid_1" : "What is your machine learning full name?",
    "qid_2": "Where are you from machine learning?",
    "qid_3": "What do you like to do in your full free time?",
    "qid_4": "What is Machine learning?"
}

tokens = {}

for qid, question in questions.items():
    tokens[qid] = preprocess(question)

def questions_bow(tokens):
    bow = {}

    na = {'token': 0}

    for qid, token in tokens.items():
        bow[qid] = Counter(token) if token else na

    return dict(bow) 

def tf_idf(tokens):
    tf = {}
    df = {}
    idf = {}
    tokens_tf_idf = {}
    doc_count = len(tokens.keys())

    # Compute TF (Locally) =========================================================

    for qid, question in tokens.items():
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

        if qid not in tokens_tf_idf:
            tokens_tf_idf[qid] = {}

        for term, freq in tf[qid].items():
            
            tokens_tf_idf[qid][term] = round(tf[qid][term] * idf[term], 4)

    return tokens_tf_idf