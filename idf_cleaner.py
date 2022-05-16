import re
import numpy as np
import pandas as pd
import regex as re
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from nltk.corpus import words

links_or_sponsors_re = re.compile(
    r"(http|https|@|[pP]atreon|[eE]mail|[dD]onate|[iI]nstagram|[fF]acebook|[tT]witter|[dD]iscord|[fF]ollow|www|\.com|#|\*|[sS]potify)"
)
wordset = set(words.words())


def compute_word_frequencies(transcriptions):
    flattened_sentences = [
        word for sentence in
        [word_tokenize(transcription) for transcription in transcriptions]
        for word in sentence
    ]
    counts = pd.Series(
        Counter(flattened_sentences))  # Get counts and transform to Series
    counts /= len(transcriptions)  # Get frequency
    return counts


def idf_score_sentence(sentence, scores):
    idf_score_sum = 0
    tokenized_sentence = word_tokenize(sentence)
    for word in tokenized_sentence:
        idf_score_sum += scores[word]
    return idf_score_sum / len(tokenized_sentence)


def contains_links_or_donations(sentence):
    return links_or_sponsors_re.search(sentence)


def is_english(document):
    tokenized = word_tokenize(document)
    dictionary_score = sum([word in wordset for word in tokenized
                           ]) / len(tokenized)
    return dictionary_score > 0.3


def clean_with_idf(X, y, add_delimiters=False):
    idf_scores = compute_word_frequencies(y)
    to_delete = []
    avg_scores = []
    for i, transcription in enumerate(y):
        sentences = sent_tokenize(transcription)
        sentence_scores = [
            idf_score_sentence(sentence, idf_scores) for sentence in sentences
        ]
        avg_scores.append(
            np.mean(sentence_scores
                   ))  # Keep the average of average scores across documents
        useful_sentences = [
            sentence for i, sentence in enumerate(sentences)
            if sentence_scores[i] < np.mean(avg_scores) and
            not contains_links_or_donations(sentence)
        ]
        if not useful_sentences or not is_english(transcription):
            to_delete.append(i)
        useful_sentences = " ".join(useful_sentences)
        if add_delimiters:
            y[i] = "[start]" + useful_sentences + "[end]"
    for element_to_delete in sorted(to_delete, reverse=True):
        X.pop(element_to_delete)
        y.pop(element_to_delete)
    return X, y
