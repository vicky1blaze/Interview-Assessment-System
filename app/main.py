from preprocessing import preprocess
from feedback import feedback
from score import score_vocab

answers = input("Enter the answer: ").lower()

sent_tok, tokens, vocabulary = preprocess(answers)

score_vocabs = score_vocab(tokens, vocabulary)

feedback(sent_tok, tokens, vocabulary, score_vocabs)
