def feedback(candidate_id, corpus_data):
    """
    Display feedback for a candidate from the new corpus hierarchy.
    Accesses: corpus_data["corpus"][candidate_id]["overall"]["text"]["features"]
    """
    if candidate_id not in corpus_data["corpus"]:
        print(f"Candidate {candidate_id} not found in corpus.")
        return

    candidate_corpus = corpus_data["corpus"][candidate_id]
    
    # Access overall statistics from the new hierarchy
    if "overall" in candidate_corpus and "text" in candidate_corpus["overall"]:
        if "features" in candidate_corpus["overall"]["text"]:
            if "statistics" in candidate_corpus["overall"]["text"]["features"]:
                stats = candidate_corpus["overall"]["text"]["features"]["statistics"]
                print(f"Vocabulary Score: {int(stats.get('vocabulary_ratio', 0) * 100)}%")
                print(f"Filler words ratio: {int(stats.get('filler_ratio', 0) * 100)}%")
                print(f"Stop words ratio: {int(stats.get('stopword_ratio', 0) * 100)}%")