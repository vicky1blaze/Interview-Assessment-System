from preprocessing import preprocess
from feedback import feedback

raw_text = input("Enter the answer: ")

lemma = preprocess(raw_text)

feedback(lemma)
