from preprocessing import preprocess
from feedback import feedback
from feature_extraction import extract_features_statistics, extract_features_sentiment, extract_features_bow

raw_text = input("Enter the answer: ")

lemma = preprocess(raw_text)

features = extract_features_statistics(raw_text)

sentiment_score = extract_features_sentiment(raw_text)

bow = extract_features_bow(lemma)

feedback(lemma, features, sentiment_score, bow)
