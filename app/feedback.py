# Tokenize
def feedback(candidate_id, corpus):
    # print(f"\n\nWord Tokenize: {tokens} \n")
    # print(f"Sentence Tokenize: {sent_tok} \n")
    # print(f"Clean list: {cleanes_list}\n")

    # print(f"Unique words: {vocabulary} \n")
    # print(f"Total unique words: {len(vocabulary)} \n")
    # print(f"Vocabulary Strength: {score_vocabs} \n")
    # print(f"Filler word weightage: {filler_percent}% \n\n")

    # print("============= Statistics ==============\n")
    for feature, doc in corpus[candidate_id].items():
        doc_counter += 1
        doc_key = f"doc_{doc_counter}"

        print(f"Vocabulary Ratio: {int(doc["statistics"]["vocabulary_ratio"] * 100)}%")
        print(f"Filler words ratio: {int(corpus[candidate_id]["statistics"]["filler_ratio"] * 100)}%")
        print(f"Stop words ratio: {int(corpus[candidate_id]["statistics"]["stopword_ratio"] * 100)}%")
    
    print("============= Sentiment Score ==============\n")
    print(f"Positive: {corpus[candidate_id]["sentiment"]["pos"]}")
    print(f"Negative: {sentiment_score["neg"]}")
    print(f"Neutral: {sentiment_score["neu"]}")
    print(f"Compound: {sentiment_score["compound"]}")

    print(f"BoW: {bow}")
    print(f"\nLemmitized: {lemma}\n")