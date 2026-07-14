from nltk.tokenize import word_tokenize, sent_tokenize
from filler_words import filler_words
from nltk.corpus import stopwords
from string import punctuation
from nltk import pos_tag
from nltk import WordNetLemmatizer

def preprocess(raw_text):
    answers = raw_text.lower()
    non_filler_token = []
    cleanest_list = []
 
    # Tokenization 

    sent_tok = sent_tokenize(answers)
    tokens = word_tokenize(answers)

    # Punctation removal & vocabulary analysis

    non_punct_tokens = [token for token in tokens if token not in punctuation]

    vocabulary = set(non_punct_tokens)
    tokens_count = len(non_punct_tokens)
    vocab_count = len(vocabulary)

    vocab_ratio = (vocab_count / tokens_count) * 100

    # Filler word analysis -- Pending BoW

    filler_word_count = 0
    
    for word in non_punct_tokens:
        if word in filler_words:
            filler_word_count += 1
        else:
            non_filler_token.append(word)

    filler_ratio = (filler_word_count / tokens_count) * 100

    # Stop word analysis        
    stop_word_count = 0
    stop_words = set(stopwords.words("english"))
    
    for word in non_filler_token:
        if word in stop_words:
            stop_word_count += 1
        else:
            cleanest_list.append(word)

    stop_ratio = (stop_word_count / tokens_count) * 100

    # POS Tagging and Lemmitization

    pos_tagg = pos_tag(cleanest_list)

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
    
    lemmetizer = WordNetLemmatizer()
    lemma = []
    
    for word, pos in pos_tagg:
        wordNet_pos = pos_mapping.get(pos, "n")
        lemma.append(lemmetizer.lemmatize(word, wordNet_pos))

    return sent_tok, tokens, vocabulary, filler_ratio, stop_ratio, vocab_ratio, cleanest_list, lemma