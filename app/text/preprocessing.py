"""
preprocessing.py - NLP text preprocessing pipeline.

Implements tokenization, punctuation removal, filler-word filtering,
POS-tagging, lemmatization, and stop-word filtering using NLTK.
"""

from string import punctuation
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
try:
    from text.filler_words import filler_words
except ImportError:
    from .filler_words import filler_words

# Ensure required NLTK resources are available
for resource in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

pos_mapping = {
    "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n",
    "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v",
    "JJ": "a", "JJR": "a", "JJS": "a",
    "RB": "r", "RBR": "r", "RBS": "r"
}

def tokenize(text):
    """Tokenize text into words."""
    if not text:
        return []
    return word_tokenize(text)

def remove_punctuation(tokens):
    """Filter out punctuation tokens."""
    return [token for token in tokens if token not in punctuation and any(c.isalnum() for c in token)]

def remove_filler(tokens):
    """Filter out filler words."""
    filler_set = set(filler_words)
    return [token for token in tokens if token.lower() not in filler_set]

def remove_stop_words(tokens):
    """Filter out English stopwords."""
    return [token for token in tokens if token.lower() not in stop_words]

def pos_tag_lemmatize(tokens):
    """
    Apply Part-of-Speech tagging and lemmatize words based on their POS tag.
    """
    if not tokens:
        return []
    pos_tags = pos_tag(tokens)
    lemmas = []
    for word, pos in pos_tags:
        wordnet_pos = pos_mapping.get(pos, "n")
        lemmas.append(lemmatizer.lemmatize(word, wordnet_pos))
    return lemmas

def preprocess(raw_text):
    """
    Execute full text preprocessing pipeline:
    raw_text -> lowercase -> tokenize -> remove punct -> remove filler
             -> POS tag & lemmatize -> remove stopwords -> lemmas list
    """
    if not raw_text or not isinstance(raw_text, str):
        return []

    text = raw_text.lower().strip()
    if not text:
        return []

    tokens = tokenize(text)
    tokens = remove_punctuation(tokens)
    tokens = remove_filler(tokens)
    tokens = pos_tag_lemmatize(tokens)
    lemmas = remove_stop_words(tokens)

    return lemmas