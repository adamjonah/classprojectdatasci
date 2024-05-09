import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub("\\W", " ", text)  # remove special chars

    # stem words
    words = re.split("\\s+", text)
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]

    # lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatize_words = [lemmatizer.lemmatize(word=word) for word in stemmed_words]
    clean_text = " ".join(lemmatize_words)

    return clean_text


def preprocess_text_no_stem_no_lem(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub("\\W", " ", text)  # remove special chars
    return text


class RevertVocabulary:
    """This function maps the post Lemmatized and Stemmed words back to original"""

    def __init__(self, words, vocabulary, lemmatized=True, stemmed=True, unstemmed_words=None):
        self.words = words
        self.vocabulary = vocabulary
        self.lemmatized = lemmatized
        self.stemmed = stemmed
        self.original_words = self._revert_words()

    def _revert_words(self):
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r"\d+", "", text)  # remove numbers
            words = re.split("\\s+", text)  # remove special chars
            return " ".join(words)

        vectorizer = CountVectorizer(strip_accents="ascii", preprocessor=preprocess_text)
        X = vectorizer.fit_transform(self.vocabulary)
        vocab = set(vectorizer.get_feature_names_out())  # all possible words that can appear

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # map stemmed words to words that occured in the original text so d will
        # Example: d['seriou'] = {'serious', 'seriously'}
        d = defaultdict(set)
        for v in vocab:
            word = v
            if self.stemmed:
                word = stemmer.stem(word)
            if self.lemmatized:
                word = lemmatizer.lemmatize(word)
            d[word].add(v)

        # for each stemmed/lemmatized word, find the equivalent original in the dictionary
        unstemmed_words = []
        for word in self.words:
            try:
                unstemmed_words.append(next(iter(d[word])))
            except:
                print(word, d[word])

        return unstemmed_words
