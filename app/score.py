# Vocbaluary score
def score_vocab(tokens, vocabulary):
    tok_count = len(tokens)
    vocab_count = len(vocabulary)

    if vocab_count > 0.40 * tok_count:
        return "Good"
    else:
        return "Weak"