# Tokenize
def feedback(sent_tok, tokens, vocab_ratio, filler_ratio, stop_ratio):
    print(f"\n\nWord Tokenize: {tokens} \n")
    print(f"Sentence Tokenize: {sent_tok} \n")

    # print(f"Unique words: {vocabulary} \n")
    # print(f"Total unique words: {len(vocabulary)} \n")
    # print(f"Vocabulary Strength: {score_vocabs} \n")
    # print(f"Filler word weightage: {filler_percent}% \n\n")

    print("============= Statistics ==============\n")
    print(f"Vocabulary Ratio: {int(vocab_ratio)}%")
    print(f"Filler words ratio: {int(filler_ratio)}%")
    print(f"Stop words ratio: {int(stop_ratio)}%")
    print("\n")