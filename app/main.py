from preprocessing import preprocess
from feedback import feedback
from feature_extraction import extract_features_statistics, extract_features_sentiment, extract_features_bow, extract_features_tfidf
from answers import candidates_data
from questions import questions
from dataset import load_data, save_data

def main():
    global candidate_id, candidate_answer

    doc_counter = 0

    candidates_data, corpus = load_data()

    while True:
        candidate_id = input("\nEnter the ID (Enter -1 to exit):").strip()
    
        if candidate_id == "-1":
            break

        if candidate_id not in candidates_data:
            candidates_data[candidate_id] = {
                "answers": {}
            }

        if candidate_id not in corpus:
                corpus[candidate_id] = {
                    "lemmas": {},
                    "bow": {},
                    "tfidf":{}
                }

        for question_id, question in questions.items():
            print(f"\n{question}")
            candidate_answer = input("Answer: ")

            doc_counter += 1
            doc_key = f"doc_{doc_counter}"

            lemma = preprocess(candidate_answer)
            corpus[candidate_id]["lemmas"][doc_key] = lemma

            bow = extract_features_bow(lemma)
            corpus[candidate_id]["bow"][doc_key] = bow

            corpus = extract_features_tfidf(candidate_id, corpus)
            
            candidates_data[candidate_id]["answers"][question_id] = candidate_answer

    save_data(candidates_data, corpus)

if __name__ == "__main__":
    main()

# lemma = preprocess()

# candidates_data(raw_text)

# features = extract_features_statistics(raw_text)

# sentiment_score = extract_features_sentiment(raw_text)

# bow = extract_features_bow(lemma)

# feedback(lemma, features, sentiment_score, bow)