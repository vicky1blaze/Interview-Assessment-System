from preprocessing import preprocess
from feedback import feedback

raw_text = input("Enter the answer: ")

sent_tok, tokens, vocabulary, filler_ratio, stop_ratio, vocab_ratio, clean_list = preprocess(raw_text)

feedback(sent_tok, tokens, vocab_ratio, filler_ratio, stop_ratio)
