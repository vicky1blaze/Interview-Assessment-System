from nltk.tokenize import word_tokenize, sent_tokenize
from filler_words import filler_words
from nltk.corpus import stopwords
from string import punctuation
from nltk import pos_tag
from nltk import WordNetLemmatizer

# Initialization
lemmetizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

pos_mapping = {
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",

    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",

    "JJ": "a",
    "JJR": "a",
    "JJS": "a",

    "RB": "r",
    "RBR": "r",
    "RBS": "r"
}

# Functions

def tokenize(text):
    tokens = word_tokenize(text)

    return tokens

def remove_punctuation(tokens):
    tokens = [token for token in tokens if token not in punctuation]

    return tokens

def remove_filler(tokens):
    tokens = [token for token in tokens if token not in filler_words]

    return tokens

def remove_stop_words(tokens):
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

def pos_tag_lemmatize(tokens):
    pos_tags = pos_tag(tokens)

    lemma = []
    
    for word, pos in pos_tags:
        wordNet_pos = pos_mapping.get(pos, "n")
        lemma.append(lemmetizer.lemmatize(word, wordNet_pos))

    return lemma

# Pre-processing Pipeline

def preprocess(raw_text):
    text = raw_text.lower()
 
    # Tokenization 
    tokens = tokenize(text)

    # Punctation removal
    tokens = remove_punctuation(tokens)

    # Filler word removal
    tokens = remove_filler(tokens)

    # Stop word removal        
    tokens = remove_stop_words(tokens)

    # POS Tagging and Lemmitization
    lemma = pos_tag_lemmatize(tokens)

    return lemma