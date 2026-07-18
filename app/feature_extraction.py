from nltk import word_tokenize, sent_tokenize
from preprocessing import remove_punctuation, stop_words
from filler_words import filler_words

def extract_features_statistics(raw_text):
    text = raw_text.lower()

    sent_tokens = sent_tokenize(text)
    word_tokens = word_tokenize(text)

    word_count = len(word_tokens)
    sentence_count = len(sent_tokens)

    average_sentence_length = (word_count / sentence_count)

    #Vocubalry

    no_punctuation_tokens = remove_punctuation(word_tokens)
    no_punctuation_tokens_count = len(no_punctuation_tokens)

    vocabulary = set(no_punctuation_tokens)
    vocabulary_count = len(vocabulary)

    vocabulary_ratio = (vocabulary_count / no_punctuation_tokens_count)

    #Filler and stop words count

    filler_words_count = 0
    stop_words_count = 0

    for token in word_tokens:
        if token in filler_words:
            filler_words_count += 1
            
        if token in stop_words:
            stop_words_count += 1

    filler_ratio = (filler_words_count / word_count)
    stopword_ratio = (stop_words_count / word_count)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "stop_words_count": stop_words_count,
        "filler_words_count": filler_words_count,
        "stopword_ratio": stopword_ratio,
        "filler_ratio": filler_ratio,
        "vocabulary": vocabulary,
        "vocabulary_count": vocabulary_count,
        "vocabulary_ratio": vocabulary_ratio,
        "no_punctuation_tokens": no_punctuation_tokens,
        "no_punctuation_tokens_count": no_punctuation_tokens_count,
        
    }


    


