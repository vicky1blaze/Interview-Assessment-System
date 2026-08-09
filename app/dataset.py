import json

def load_data(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)

    return file

def save_data(file_path, file):

    with open(file_path, "w") as f:
        json.dump(file, f, indent=4)

    print("Saved successfully\n")