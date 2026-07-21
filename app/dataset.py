import json

def load_data():
    with open("data/candidates.json", "r") as file:
        data = json.load(file)

    with open("data/corpus.json", "r") as file:
        corpus = json.load(file)

    return data, corpus

def save_data(data, corpus):

    with open("data/candidates.json", "w") as file:
        json.dump(data, file, indent=4)

    with open("data/corpus.json", "w") as file:
        json.dump(corpus, file, indent=4)

    print("Saved successfully\n")