from text.preprocessing import preprocess
from feedback import feedback
from text.feature_extraction import extract_features_statistics, extract_features_sentiment, extract_features_bow, extract_features_tfidf
from answers import candidates_data
from questions import questions
from dataset import load_data, save_data

def main():
    global corpus, candidate_id

    doc_counter = 0

    candidates_data, corpus = load_data()

    while True:
        temp_id = input("\nEnter the ID (Enter -1 to exit):").strip()
    
        if temp_id == "-1":
            break

        candidate_id = temp_id

        if candidate_id not in candidates_data:
            candidates_data[candidate_id] = {
                "answers": {},
                "combined_answer": ""
            }

        if candidate_id not in corpus:
                corpus[candidate_id] = {
                    "lemmas": {},
                    "bow": {},
                    "tfidf":{},
                    "statistics": {},
                    "sentiment": {}
                }

        for question_id, question in questions.items():
            print(f"\n{question}")
            candidate_answer = input("Answer: ")

            space = " " if candidates_data[candidate_id]["combined_answer"].endswith(".") else ". "
            candidates_data[candidate_id]["combined_answer"] += space + candidate_answer

            doc_counter += 1
            doc_key = f"doc_{doc_counter}"

            candidates_data[candidate_id]["answers"][question_id] = candidate_answer

            lemma = preprocess(candidate_answer)
            corpus[candidate_id]["lemmas"][doc_key] = lemma
            
            sentiment_score = extract_features_sentiment(candidate_answer)
            corpus[candidate_id]["sentiment"][doc_key] = sentiment_score

            bow = extract_features_bow(lemma)
            corpus[candidate_id]["bow"][doc_key] = bow

            corpus = extract_features_tfidf(candidate_id, corpus)

    if candidate_id != "-1":
        features = extract_features_statistics(candidates_data, candidate_id)
        corpus[candidate_id]["statistics"] = features

    save_data(candidates_data, corpus)

if __name__ == "__main__":
    main()

# feedback(candidate_id, corpus)