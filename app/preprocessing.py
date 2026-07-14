from nltk.tokenize import word_tokenize, sent_tokenize

def preprocess(answers):
    sent_tok = sent_tokenize(answers)
    tokens = word_tokenize(answers)
    vocabs = set(tokens)

    return sent_tok, tokens, vocabs