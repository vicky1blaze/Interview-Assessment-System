from nltk.tokenize import word_tokenize, sent_tokenize
from filler_words import filler_words
from nltk.corpus import stopwords

def preprocess(raw_text):
    answers = raw_text.lower()
 
    # Tokenization 

    sent_tok = sent_tokenize(answers)
    tokens = word_tokenize(answers)
    vocabulary = set(tokens) #unique words

    tok_count = len(tokens)
    vocab_count = len(vocabulary)
    clean_list = []

    # Filler word analysis -- Pending BoW

    filler_word_count = 0
    for word in tokens:
        if word not in filler_words:
            clean_list.append(word)
        else:
            filler_word_count += 1

    filler_ratio = (filler_word_count / tok_count) * 100

    # Vocabulary ratio

    vocab_ratio = (vocab_count / tok_count) * 100

    # Stop word analysis        
    stop_word_count = 0
    stop_words = set(stopwords.words("english"))
    
    for word in tokens:
        if word in stop_words:
            clean_list.remove(word)
            stop_word_count += 1

    stop_ratio = (stop_word_count / tok_count) * 100

    return sent_tok, tokens, vocabulary, filler_ratio, stop_ratio, vocab_ratio, clean_list