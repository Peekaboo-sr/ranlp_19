#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim

Functions to load the OLID dataset (https://scholar.harvard.edu/malmasi/olid)
"""

import sys
import pickle
from tensorflow import keras
from data import read
from olpred import nn_functions as nn

NN_CONFIG = {
             'class_weights_smooth_factor': 1,
             'early_stopping': True,
             'early_stopping_config': (('min_delta', 0.0005), ('patience', 5), ('weights_num_epochs', 0)),
             'threshold': 0.5,  # probability threshold for binary classification, 0.5 default
             'print_model_info': True,
             'text_adam_learning_rate': 0.0002,  # 0.001 default
             'text_dense_l2_regularization': 0.01,  # output layer, 0.01 default
             'embeddings_trainable': True,
             'text_segment_max_length': 58,
             'text_early_stopping': True,
             'text_early_stopping_config': {'min_delta': 0.0005, 'patience': 5, 'weights_num_epochs': 1},
             'text_cnn_config': (16, (1, 2, 3, 4, 5, 6), 8),
             }


def prepare_ranlp_train_test_data(olid_data_filename, reddit_data_xml_corpus_filename):
    sys.stderr.write('Reading olid training data...\n')
    olid_data = read.read_olid_training_data(olid_data_filename)
    sys.stderr.write(' ... done.\n')
    train_data = [labeled_text[0] for labeled_text in olid_data]
    train_data_labels = [labeled_text[1] for labeled_text in olid_data]

    reddit_data = read.read_reddit_xml_corpus_texts(reddit_data_xml_corpus_filename)

    return train_data, train_data_labels, reddit_data


def train_test_cnn(training_data, train_gold_labels, test_data, nn_config, word_index, embedding_matrix):
    sys.stderr.write('Preparing data for NN...\n')
    training_data_vector = nn.index_data(word_index, training_data)
    test_data_vector = nn.index_data(word_index, test_data)
    label_index, train_gold_labels_vector, class_weights = nn.create_label_index(train_gold_labels)

    sequence_max_length = nn_config.get('text_segment_max_length', 58)
    train_data_sequences = keras.preprocessing.sequence.pad_sequences(
        training_data_vector, value=word_index["<PAD>"], padding='pre', maxlen=sequence_max_length)
    val_ratio_denominator = 10
    partial_x_train = train_data_sequences[round(len(train_data_sequences) / val_ratio_denominator):]
    partial_y_train = train_gold_labels_vector[round(len(train_gold_labels_vector) / val_ratio_denominator):]
    x_val = train_data_sequences[:round(len(train_data_sequences) / val_ratio_denominator)]
    y_val = train_gold_labels_vector[:round(len(train_gold_labels_vector) / val_ratio_denominator)]
    sys.stderr.write('... done.\n')

    sys.stderr.write('Preparing NN...\n')
    nn_model = nn.build_cnn(word_index, embedding_matrix, len(label_index), nn_config)
    sys.stderr.write('... done.\n')
    sys.stderr.write('Training...\n')
    nn.train_model(nn_model, partial_x_train, partial_y_train, x_val, y_val, class_weights, len(label_index), nn_config)
    sys.stderr.write('... done.\n')

    sys.stderr.write('Predicting...\n')
    test_data_sequences = keras.preprocessing.sequence.pad_sequences(
        test_data_vector, value=word_index["<PAD>"], padding='pre', maxlen=sequence_max_length)
    predictions = nn_model.predict(test_data_sequences)
    sys.stderr.write('... done.\n')
    return predictions


def predict_offensive_language(olid_data_filename, reddit_data_xml_corpus_filename, word_embeddings_path,
                               pickle_file_path):
    # prepare train and test data and load word embeddings
    try:
        sys.stderr.write('Loading data pickle...\n')
        loaded_data = pickle.load(open(pickle_file_path, mode='rb'))
        sys.stderr.write('...done.\n')
        word_index, embedding_matrix, training_data, train_gold_labels, test_data = loaded_data
    except FileNotFoundError:
        training_data, train_gold_labels, test_data = \
            prepare_ranlp_train_test_data(olid_data_filename, reddit_data_xml_corpus_filename)
        word_index, embedding_matrix = read.read_word_embeddings_random(word_embeddings_path, training_data)
        with open(pickle_file_path, mode='wb') as outfile:
            pickle.dump((word_index, embedding_matrix, training_data, train_gold_labels, test_data), outfile)

    # run NN
    predictions = train_test_cnn(training_data, train_gold_labels, test_data, NN_CONFIG, word_index, embedding_matrix)

    return predictions

