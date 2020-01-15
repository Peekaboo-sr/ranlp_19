#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim
"""

import sys
import re
import numpy as np
from collections import defaultdict
from random import shuffle
from unidecode import unidecode
import xml.etree.ElementTree as ET
from nltk.tokenize import TweetTokenizer

TOKENIZER = TweetTokenizer(reduce_len=True)
WORD_INDEX_SPECIAL_MARKS = {"<PAD>": 0, "<START>": 1, "<UNK>": 2, "<UNUSED>": 3}


def get_comment_text_from_descendants(node, text):
    if node.tag == 'comment':
        node_text = node.text
        if node_text:
            text.append(TOKENIZER.tokenize((node_text.strip())))
    for child in node:
        text = get_comment_text_from_descendants(child, text)
    return text


def read_reddit_xml_corpus_texts(path):
    text = []
    sys.stderr.write('Parsing reddit corpus...\n')
    xml_corpus = ET.parse(path).getroot()
    sys.stderr.write(' ... done.\n')
    sys.stderr.write('Selecting comments from corpus...\n')
    text = get_comment_text_from_descendants(xml_corpus, text)
    sys.stderr.write(' ... done.\n')
    return tuple(text)


def read_reddit_xml_corpus(path):
    sys.stderr.write('Parsing reddit corpus...\n')
    xml_corpus = ET.parse(path).getroot()
    sys.stderr.write(' ... done.\n')
    return xml_corpus


def read_olid_training_data(path):
    binary_annotated_instances = []
    for line in open(path, encoding='utf-8'):
        line = line.split('\t')
        if line[0] == 'id':
            continue

        tweet = unidecode(line[1])

        # twitter-specific: remove user name mentions (@-marked)
        tweet = re.sub(r'@[A-Za-z0-9_]{1,15}([^A-Za-z0-9_]|$)', r'\1', tweet)

        # twitter-specific: process keywords (remove #-mark)
        tweet = tweet.replace('#', '')

        # tokenization
        tweet = TOKENIZER.tokenize(tweet)

        # normalization: lower casing and replacing all numbers by "2"
        tweet = [word.lower() for word in tweet]
        tweet = [re.sub(r'[0-9]+([.,][0-9]+)?', '2', word) for word in tweet]

        label = line[2] == 'OFF'
        binary_annotated_instances.append((tweet, label))
    shuffle(binary_annotated_instances)
    return tuple(binary_annotated_instances)


def get_word_index(data, max_vocab_size=10000):
    """Pre-compute a word-index based on a set of words of a given dataset.

    :param data: dataset used to compute the word index/vocabulary
    :param max_vocab_size: maximum size limit of the word index/vocabulary
    :return: word index (dictionary with key=word, value=index)
    """
    vocabulary = defaultdict(int)
    for instance in data:
        for word in instance:
            vocabulary[word.lower()] += 1

    # filter out words with frequency = 1
    vocabulary = {k: v for k, v in vocabulary.items() if v > 1}
    vocabulary = sorted(vocabulary.keys(), key=lambda k: vocabulary[k], reverse=True)[:max_vocab_size-4]
    sys.stderr.write('vocab size: %d\n' % len(vocabulary))

    # special pre-defined embedding values
    word_index = {"<PAD>": 0, "<START>": 1, "<UNK>": 2, "<UNUSED>": 3}
    for word in vocabulary:
        index = len(word_index)
        word_index[word] = index

    return word_index


def read_word_embeddings_random(path, data, max_vocab_size=10000):
    """Pre-compute a word-index and embedding-matrix based on a set of words of a given dataset.

    :param path: filename of pre-trained word embeddings
    :param data: dataset used to compute the word index/vocabulary
    :param max_vocab_size: maximum size limit of the word index/vocabulary
    :return: tuple of word index (dictionary with key=word, value=index) and embedding matrix (numpy array in form of a
             matrix where the index corresponds to the word index and the vector of each index to the word vector)
    """
    vocabulary = defaultdict(int)
    for instance in data:
        for word in instance:
            vocabulary[word.lower()] += 1

    # filter out words with frequency = 1
    vocabulary = {k: v for k, v in vocabulary.items() if v > 1}
    vocabulary = sorted(vocabulary.keys(), key=lambda k: vocabulary[k], reverse=True)[:max_vocab_size-4]
    sys.stderr.write('vocab size: %d\n' % len(vocabulary))

    # special pre-defined embedding values
    word_index = {k: v for k, v in WORD_INDEX_SPECIAL_MARKS.items()}
    word_embeddings = {v: np.random.rand(200) for v in word_index.values()}
    for word in vocabulary:
        index = len(word_index)
        word_index[word] = index
        word_embeddings[index] = np.random.rand(200)

    sys.stderr.write('Reading word embeddings from file "%s".\n' % path)
    vocab_keys = set(vocabulary)
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        line = line.split(' ')
        if len(line) < 3:
            continue
        word = line[0].lower()
        if word in vocabulary:
            index = word_index[word]
            vector = np.asarray(line[1:], dtype=np.float32)
            word_embeddings[index] = vector
            vocab_keys.remove(word)
    sys.stderr.write('Word types not found in pre-trained word embeddings: %d\n' % len(vocab_keys))

    embedding_matrix = np.random.random((len(word_index) + 1, 200))
    for index, vector in word_embeddings.items():
        embedding_matrix[index] = vector
    return word_index, embedding_matrix
