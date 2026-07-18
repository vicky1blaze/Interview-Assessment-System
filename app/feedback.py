# Tokenize
def feedback(lemma, features):
    # print(f"\n\nWord Tokenize: {tokens} \n")
    # print(f"Sentence Tokenize: {sent_tok} \n")
    # print(f"Clean list: {cleanes_list}\n")

    # print(f"Unique words: {vocabulary} \n")
    # print(f"Total unique words: {len(vocabulary)} \n")
    # print(f"Vocabulary Strength: {score_vocabs} \n")
    # print(f"Filler word weightage: {filler_percent}% \n\n")

    print("============= Statistics ==============\n")
    print(f"Vocabulary Ratio: {int(features["vocabulary_ratio"] * 100)}%")
    print(f"Filler words ratio: {int(features["filler_ratio"] * 100)}%")
    print(f"Stop words ratio: {int(features["stopword_ratio"] * 100)}%")
    print("\n")

    print(f"\nLemmitized: {lemma}\n")