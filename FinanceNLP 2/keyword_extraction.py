# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet_ic')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    nltk.download('omw-1.4')

import pke
import summa
from nltk.stem import PorterStemmer
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import _stop_words
from yake import KeywordExtractor

# Obligatory path fix
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath("")))))
sys.path.insert(0, os.path.split(os.getcwd())[0])

from ProcessText import preprocess_text, RevertVocabulary, preprocess_text_no_stem_no_lem

class KeywordExtraction:

    def __init__(self, ad, n_keywords=10):
        self.n = n_keywords
        self.all_data = ad
        self.all_vocab_idf = pd.DataFrame({'A' : []})

    def tf_idf_fit(self, data, stop_words):
        stop_words = [PorterStemmer().stem(word=word) for word in stop_words]
        data = np.array(data)
        vectorizer = TfidfVectorizer(stop_words=stop_words, strip_accents="ascii", preprocessor=preprocess_text)
        X = vectorizer.fit_transform(data)

        return X, vectorizer

    def idf(self, group_data):
        """For a universe of documents and a group of documents extract the keywords"""

        stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
        stop_words = [PorterStemmer().stem(word=word) for word in stop_words]  # words to be removed

        if self.all_vocab_idf.empty:
            X, vectorizer = self.tf_idf_fit(self.all_data, stop_words)  # calculate the idf for all words in all docs
            all_document_vocabulary = vectorizer.get_feature_names_out()
            all_document_idf = vectorizer.idf_
            self.all_vocab_idf = pd.DataFrame({"vocabulary": all_document_vocabulary, "idf": all_document_idf})

        X, vectorizer = self.tf_idf_fit(group_data, stop_words)  # calculate the idf for all words in group
        group_document_vocabulary = vectorizer.get_feature_names_out()
        group_document_idf = vectorizer.idf_
        group_vocab_idf = pd.DataFrame({"vocabulary": group_document_vocabulary, "idf": group_document_idf})

        # Keywords will have a high idf score for all the documents and a low score in the group of documents.
        group_vocab = group_vocab_idf.merge(self.all_vocab_idf, on="vocabulary", how="left", suffixes=("_group", "_all"))
        group_vocab["idf_all/idf_group"] = group_vocab["idf_all"].div(group_vocab["idf_group"])

        u = RevertVocabulary(group_vocab.vocabulary.values, self.all_data)  # reverts data to original format
        group_vocab["original_vocabulary"] = u.original_words

        group_vocab.sort_values(by=["idf_all/idf_group"], ascending=False, inplace=True)
        keywords = group_vocab.original_vocabulary[:self.n].values

        return keywords

    def tf_idf(self, group_data):
        """
        * TF := Term Frequency. How often does a word appear in a document.
        * IDF : How imporant is that word in the context of all the documents.
        """

        stop_words = list(_stop_words.ENGLISH_STOP_WORDS)

        if self.all_vocab_idf.empty:
            print("Generating keywords for all documents.")
            X, vectorizer = self.tf_idf_fit(self.all_data, stop_words)
            all_document_vocabulary = vectorizer.get_feature_names_out()
            all_document_idf = vectorizer.idf_
            self.all_vocab_idf = pd.DataFrame({"vocabulary": all_document_vocabulary, "idf": all_document_idf})

        clean_data = [preprocess_text(text) for text in list(group_data)]  # clean each text

        vectorizer = CountVectorizer(stop_words="english")
        tf = vectorizer.fit_transform(clean_data)  # term frequency for each document
        tf = np.squeeze(np.array(tf.sum(axis=0)))  # sum the frequencies across

        subgroub_calc = pd.DataFrame({"vocabulary": vectorizer.get_feature_names_out(), "tf": tf})
        subgroub_calc = pd.merge(subgroub_calc, self.all_vocab_idf, on="vocabulary")
        subgroub_calc["tf*idf"] = subgroub_calc["tf"] * subgroub_calc["idf"]  # TODO: think about if this is correct

        u = RevertVocabulary(subgroub_calc.vocabulary.values, self.all_data)  # reverts data to original format
        subgroub_calc["original_vocabulary"] = u.original_words

        subgroub_calc.sort_values("tf*idf", ascending=False, inplace=True)
        keywords = subgroub_calc.original_vocabulary[:self.n].values
        return keywords

    def YAKE(self, group_data):
        """https://github.com/LIAAD/yake"""

        clean_text = [preprocess_text_no_stem_no_lem(text) for text in list(group_data)]  # TODO: is this how we preprocess the text
        clean_concat_text = ".".join(clean_text)  # TODO: is this correct

        kw_extractor = KeywordExtractor(lan="en", n=1, top=self.n)
        keywords = kw_extractor.extract_keywords(clean_concat_text)
        keywords = [key[0] for key in keywords]
        return keywords

    def RAKE(self, group_data):
        """
        RAKE (Rapid Automatic Keyword Extraction) -- for keyphrases
        https://csurfer.github.io/rake-nltk/_build/html/index.html
        """

        # clean text
        clean_text = [preprocess_text(text) for text in list(group_data)]  # TODO: is this how we preprocess the text

        stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
        r = Rake(stopwords=stop_words, max_length=4, include_repeated_phrases=False)  # TODO: is a good method?

        r.extract_keywords_from_sentences(clean_text)
        keywords = [key[1] for key in r.get_ranked_phrases_with_scores()]
        return keywords[:self.n]

    def TextRank(self, group_data):
        """https://github.com/summanlp/textrank"""

        # clean text
        clean_text = [preprocess_text_no_stem_no_lem(text) for text in list(group_data)]  # TODO: is this how we preprocess the text
        clean_concat_text = "".join(clean_text)  # TODO: is this the correct form of the input?

        keywords = summa.keywords.keywords(clean_concat_text)  # TODO: are they ranked?
        keywords = keywords.split("\n")  # this is a string
        return keywords[:self.n]

    def TopicRank(self, group_data):
        """
        To use, please install:
        !pip install git+https://github.com/boudinfl/pke.git
        !python -m spacy download en
        """

        # clean text
        clean_text = [preprocess_text_no_stem_no_lem(text) for text in list(group_data)]  # TODO: is this how we preprocess the text?
        clean_concat_text = ".".join(clean_text)  # TODO: is this the correct form of the input?

        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(input=clean_concat_text, language="en")

        # TODO: Are keywords ranked?#
        # TODO: Is this the correct format of the input?
        # TODO: Unstem themif possible, you might need to create a new function in preprocess_text.py

        keywords = []
        for i, sentence in enumerate(extractor.sentences):
            keywords.extend(sentence.words[:self.n])
        return keywords

    def KeyBert(self, group_data):
        """
        To use, please install:
        !pip install git+https://github.com/boudinfl/pke.git
        !python -m spacy download en
        """

        # clean text
        clean_text = [preprocess_text_no_stem_no_lem(text) for text in list(group_data)]  # TODO: is this how we preprocess the text?
        clean_concat_text = ".".join(clean_text)

        stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(clean_concat_text, top_n=self.n, stop_words=stop_words)
        keywords = [key[0] for key in keywords]
        return keywords[:self.n]

    def KP_Miner(self, group_data):
        """
        To use, please install:
        !pip install git+https://github.com/boudinfl/pke.git
        !python -m spacy download en
        """

        # clean text
        clean_text = [preprocess_text_no_stem_no_lem(text) for text in list(group_data)]  # TODO: is this how we preprocess the text?
        clean_concat_text = ".".join(clean_text)  # TODO: is this the correct form of the input?

        # TODO: Are keyphrases ranked?
        # TODO: Can we make them into keywords?

        extractor = pke.unsupervised.KPMiner()

        extractor.load_document(input=clean_concat_text, language="en", normalization=None)  # TODO: are these hyper parameters correct
        extractor.candidate_selection()
        extractor.candidate_weighting()  # alpha=alpha, sigma=sigma)
        keyphrases = extractor.get_n_best(n=self.n)
        keyphrases = [key[0] for key in keyphrases]
        return keyphrases[:self.n]