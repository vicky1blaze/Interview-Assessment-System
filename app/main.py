from preprocessing import preprocess
from feedback import feedback
from feature_extraction import extract_features_statistics

raw_text = input("Enter the answer: ")

lemma = preprocess(raw_text)

features = extract_features_statistics(raw_text)

feedback(lemma, features)
