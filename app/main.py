from text.preprocessing import preprocess
from feedback import feedback
from text.feature_extraction import extract_features_statistics, extract_features_sentiment, extract_features_bow, extract_features_tfidf, evaluate_candidate
from answers import candidates_data
from questions import questions, questions_tfidf
from dataset import load_data, save_data
from speech.speech_to_text import record, speech_to_text

def main():
    global corpus, candidate_id
    extension = ".wav"

    corpus_data = load_data("data/corpus.json")
    candidates_data = load_data("data/candidates.json")
    questions_statistics = questions_tfidf()

    # Ensure root-level structure
    if "corpus" not in corpus_data:
        corpus_data = {"corpus": {}}
    if "candidates" not in candidates_data:
        candidates_data = {"candidates": {}}

    print("█"*60)
    print("\t\tInterview Assesment System")
    print("█"*60, "\n")

    while True:
        candidate_id = input("\nEnter the Candidate ID (Enter -1 to exit): ").strip()
    
        if candidate_id == "-1":
            break

        # Initialize candidate in new hierarchy if not exists
        if candidate_id not in candidates_data["candidates"]:
            candidates_data["candidates"][candidate_id] = {
                "metadata": {
                    "candidate_id": candidate_id,
                    "created_at": None
                },
                "responses": {}
            }

        if candidate_id not in corpus_data["corpus"]:
            corpus_data["corpus"][candidate_id] = {
                "metadata": {
                    "candidate_id": candidate_id
                },
                "responses": {},
                "overall": {
                    "text": {"features": {"statistics": {}, "sentiment": {}}},
                    "speech": {"features": {}},
                    "video": {"features": {}},
                    "evaluation": {"semantic": {}, "lexical": {}, "rule_based": {}},
                    "fusion": {"features": {}}
                }
            }

        Q = 1

        combined_answer_text = ""

        # Select answer mode: Text/Speech

        mode_of_answer = int(input("Select the mode of answer: \n 1. Text \n 2. Speech \nEnter Choice: "))

        if mode_of_answer == 1:
            text_mode = True
            print("\nLog: Initializing text mode")
        else:
            text_mode = False
            print("\nLog: Initializing speech mode")

        for question_id, question in questions.items():
            print(f"\nQ{Q}: {question}")

            if text_mode:
                candidate_answer = input("Answer: ")
            else:
                
            # =========================================================================================
            # Speech Module
            # =========================================================================================

                audio_path = "speech/audio/cid_" + candidate_id + "_" + question_id + extension    
                
                record(audio_path) 
                candidates_data, corpus_data = speech_to_text(audio_path, candidates_data, corpus_data, candidate_id, question_id)
                candidate_answer = candidates_data["candidates"][candidate_id]["responses"][question_id]["speech"]["transcript"]

            Q += 1    

            # Add to combined answer for overall statistics
            space = " " if combined_answer_text.endswith(".") else ". "
            combined_answer_text += space + candidate_answer

            # Store answer in candidates.json under new hierarchy
            candidates_data["candidates"][candidate_id]["responses"][question_id] = {
                "text": {
                    "answer": candidate_answer
                },
                "speech": {
                    "audio_path": None,
                    "transcript": None
                },
                "video": {
                    "video_path": None
                }
            }

            # Process answer and store in corpus under new hierarchy
            lemma = preprocess(candidate_answer)
            
            corpus_data["corpus"][candidate_id]["responses"][question_id] = {
                "text": {
                    "processed": {
                        "lemmas": lemma,
                        "bow": extract_features_bow(lemma),
                        "tfidf": {}
                    },
                    "features": {
                        "statistics": extract_features_statistics(candidate_answer),
                        "sentiment": extract_features_sentiment(candidate_answer)
                    }
                },
                "speech": {
                    "transcript": "",
                    "features": {}
                },
                "video": {
                    "features": {}
                },
                "evaluation": {}
            }

        # Compute TF-IDF for all responses
        corpus_data["corpus"] = extract_features_tfidf(candidate_id, corpus_data["corpus"])

        # Evaluate candidate against questions
        corpus_data["corpus"] = evaluate_candidate(corpus_data["corpus"], questions_statistics, candidate_id)

        # Compute overall statistics from combined answer
        overall_stats = extract_features_statistics(combined_answer_text)
        corpus_data["corpus"][candidate_id]["overall"]["text"]["features"]["statistics"] = overall_stats
        
        # Compute overall sentiment from combined answer
        overall_sentiment = extract_features_sentiment(combined_answer_text)
        corpus_data["corpus"][candidate_id]["overall"]["text"]["features"]["sentiment"] = overall_sentiment

    # Save with new structure
    save_data("data/candidates.json", candidates_data, "Saved Candidate Data")
    save_data("data/corpus.json", corpus_data, "Saved Candidate Stats")
    
    # feedback(candidate_id, corpus_data)

if __name__ == "__main__":
    main()