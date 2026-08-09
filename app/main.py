from text.preprocessing import preprocess
from feedback import feedback
from text.feature_extraction import extract_features_statistics, extract_features_sentiment, extract_features_bow, extract_features_tfidf, evaluate_candidate
from answers import candidates_data
from questions import questions, questions_tfidf
from dataset import load_data, save_data
from speech.speech_to_text import record, speech_to_text

def main():
    global corpus, candidate_id
    # extension = ".wav"

    qid_counter = 0

    corpus = load_data("data/corpus.json")
    candidates_data = load_data("data/candidates.json")
    questions_statistics = questions_tfidf()

    while True:
        candidate_id = input("\nEnter the Candidate ID (Enter -1 to exit): ").strip()
    
        if candidate_id == "-1":
            break

        if candidate_id not in candidates_data:
            candidates_data[candidate_id] = {
                "text": {
                     "answers": {},
                     "combined_answer": ""
                },
                "speech": {
                     "answers": {},
                     "combined_answer": ""
                }
            }

        if candidate_id not in corpus:
                corpus[candidate_id] = {
                    "lemmas": {},
                    "bow": {},
                    "tfidf":{},
                    "statistics": {},
                    "sentiment": {},
                    "lexical_semantic_score": {
                        "similarity": {},
                        "coverage": {},
                        "final": {}
                    }
                }

        Q = 1

        for question_id, question in questions.items():
            print(f"\nQ{Q}: {question}")
            candidate_answer = input("Answer: ")

            Q += 1

            # =========================================================================================
            # Speech Module: If speech enable use this
            # =========================================================================================

            # audio_path = "speech/audio/cid_" + candidate_id + "_" + question_id + extension    

            # record(audio_path) 
            # candidates_data = speech_to_text(audio_path, candidates_data, candidate_id, question_id)
            # 
            # =========================================================================================    

            space = " " if candidates_data[candidate_id]["text"]["combined_answer"].endswith(".") else ". "
            candidates_data[candidate_id]["text"]["combined_answer"] += space + candidate_answer

            qid_counter += 1
            qid_key = f"qid_{qid_counter}"

            candidates_data[candidate_id]["text"]["answers"][question_id] = candidate_answer

            lemma = preprocess(candidate_answer)
            corpus[candidate_id]["lemmas"][qid_key] = lemma
            
            sentiment_score = extract_features_sentiment(candidate_answer) # Re-think on Sentiment
            corpus[candidate_id]["sentiment"][qid_key] = sentiment_score

            bow = extract_features_bow(lemma)
            corpus[candidate_id]["bow"][qid_key] = bow

        corpus = extract_features_tfidf(candidate_id, corpus)

        corpus = evaluate_candidate(corpus, questions_statistics, candidate_id)

        features = extract_features_statistics(candidates_data, candidate_id)
        corpus[candidate_id]["statistics"] = features

    save_data("data/candidates.json", candidates_data, "Saved Candidate Data")
    save_data("data/corpus.json", corpus, "Saved Candidate Stats")

if __name__ == "__main__":
    main()

# # feedback(candidate_id, corpus)