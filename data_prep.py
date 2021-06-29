#!/bin/sh python
###################################################################################################
# This script preprocesses the spam input message to convert into data types that are compatible
# with SMPC. Because SMPC can't be used to process strings directly, only numerical values.
#

# Numpy for accelerated string processing
import numpy as np
# Pandas for CSV processing of the sample msgs
import pandas as pd
# RegEx for text filtering
import re

# Stopwords that could be removed from the msgs
from nltk.corpus import stopwords

# Unset the stop words temporarily, may set it back if we need it
STOP_WORDS = set([])  # set(stopwords.words("english")


def clean_text(text):
    '''Clean text by removing all punctuations/symbols and stop words'''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text


def word_tokens(text, word_index):
    '''Convert words into numbered tokens'''
    tokens = []
    for word in text.split():
        tokens.append(word_index[word])
    return tokens


def pad_n_trunc(msgs, max_len):
    '''Pad and truncate the message to fit within a data set size'''
    features = np.zeros((len(msgs), max_len), dtype=int)
    for i, text in enumerate(msgs):
        if len(text):
            features[i, -len(text):] = text[:max_len]
    return features


def main():
    # Import spam data set into dataframe and clean it
    data = pd.read_csv("spamsample.csv", sep='\t')
    data.text = data.text.apply(clean_text)

    # Convert words into tokens
    words = set((' '.join(data.text)).split())
    word_index = {word: i for i, word in enumerate(words, 1)}
    tokens = data.text.apply(lambda x: word_tokens(x, word_index))

    # Resize dataset to fit within a fixed size of 30 elements
    texts = pad_n_trunc(tokens, 30)
    # Data that is spam, will be set to 1, else 0
    types = np.array((data.type == 'spam').astype(int))

    # Save both message types and text messages
    np.save("data/msgtypes.npy", types)
    np.save("data/msgtexts.npy", texts)


if __name__ == '__main__':
    main()
