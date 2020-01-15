#!/usr/bin/python3

"""
@author Johannes Schäfer - IwiSt, University of Hildesheim
"""

import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow import keras


class EarlyStoppingModelCheckpoints(keras.callbacks.Callback):
    """Stop training and load saved model weights when a monitored quantity has stopped improving.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0.,
                 patience=0,
                 verbose=0,
                 weights_num_epochs=0,
                 mode='auto',
                 baseline=None):
        """
        :param monitor: quantity to be monitored.
        :param min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
                          change of less than min_delta, will count as no improvement.
        :param patience: number of epochs with no improvement after which training will be stopped.
        :param verbose: verbosity mode.
        :param weights_num_epochs: number of epochs with no improvement to load model weights from.
        :param mode: one of {auto, min, max}. In `min` mode, training will stop when the quantity monitored has stopped
                     decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing; in
                     `auto` mode, the direction is automatically inferred from the name of the monitored quantity.
        :param baseline: baseline value for the monitored quantity. Training will stop if the model doesn't show
                         improvement over the baseline.
        """
        super(EarlyStoppingModelCheckpoints, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.weights_num_epochs = weights_num_epochs
        self.best = None
        self.saved_model_weights = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.saved_model_weights = []
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.saved_model_weights = [self.model.get_weights()]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.set_weights(self.saved_model_weights[self.weights_num_epochs])
                self.model.stop_training = True
            else:
                self.saved_model_weights.append(self.model.get_weights())

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %d: early stopping, loading weights from model after epoch %d' %
                  (self.stopped_epoch + 1,
                   self.stopped_epoch + 1 + self.weights_num_epochs - len(self.saved_model_weights)))


def index_data(word_index, data):
    """Function to convert a dataset to indexed instances based on a given word index.

    :param word_index: given mapping of words to indices
    :param data: list of text instances
    :return: list of indexed instances
    """
    data_vector = []
    for instance in data:
        instance_vector = [word_index["<START>"]]
        for word in instance:
            instance_vector.append(word_index.get(word, word_index["<UNK>"]))
        data_vector.append(instance_vector)
    return data_vector


def create_label_index(train_gold_labels):
    """Create an index for labels (binary per default, otherwise fine grained using the given labels).

    :param train_gold_labels: list of training data labels
    :return: the label index mapping and vectors for the train and test labels
    """
    label_index = {True: 1, False: 0}
    try:
        train_gold_labels_vector = [label_index[label] for label in train_gold_labels]
    except KeyError:
        # fine-grained labels, categories
        labels = sorted(Counter(train_gold_labels).items(), key=lambda pair: pair[1])
        label_index = {label[0]: index for index, label in enumerate(labels)}
        train_gold_labels_vector = [label_index[label] for label in train_gold_labels]
    counted_labels = Counter(train_gold_labels)
    class_weights = {label: len(train_gold_labels) / float(freq) for label, freq in counted_labels.items()}
    return label_index, train_gold_labels_vector, class_weights


def build_cnn(word_index, embedding_matrix, num_labels, config):
    # Input Layer
    sequence_max_length = config.get('text_segment_max_length', 48)
    input_layer_dtype = 'int32'
    input_layer_shape = (sequence_max_length,)
    input_layer_name = 'Input_Text'
    input_layer = keras.layers.Input(shape=input_layer_shape, dtype=input_layer_dtype, name=input_layer_name)

    # Embedding Layer
    embedding_layer_output_dim = 200
    embedding_layer_input_length = sequence_max_length
    embedding_layer_name = 'Word_Embedding'
    embedding_layer_input_dim = len(word_index)
    embedding_initial_weights = [embedding_matrix] if len(embedding_matrix) > 1 else None
    embedding_weights_trainable = config.get('embeddings_trainable', True)
    embedding_layer = keras.layers.Embedding(embedding_layer_input_dim + 1,
                                             embedding_layer_output_dim,
                                             weights=embedding_initial_weights,
                                             input_length=embedding_layer_input_length,
                                             trainable=embedding_weights_trainable,
                                             name=embedding_layer_name,
                                             )(input_layer)

    # (parallel) CNN Layers
    num_filters, window_sizes, max_pool_output_size = config.get('text_cnn_config', (16, (1, 2, 3, 4, 5, 6), 8))
    max_pool_size = round(sequence_max_length / max_pool_output_size)

    conv_layers = []
    conv_layer_id = 0
    for window_size in window_sizes:
        conv_layer_id += 1
        conv_encoder = keras.layers.Conv1D(filters=num_filters,
                                           kernel_size=window_size,
                                           activation=tf.nn.relu,
                                           padding='same',
                                           name='Conv_' + str(conv_layer_id),
                                           )(embedding_layer)
        conv_encoder = keras.layers.MaxPool1D(pool_size=max_pool_size, name='MaxPool_' + str(conv_layer_id)
                                              )(conv_encoder)
        conv_encoder = keras.layers.Dropout(0.25)(conv_encoder)
        conv_layers.append(conv_encoder)

    if len(conv_layers) > 1:
        encoder_text = keras.layers.concatenate(conv_layers, axis=-1)
    else:
        encoder_text = conv_layers[0]

    encoder_text = keras.layers.Flatten(name='Flatten')(encoder_text)

    # Output Layer
    final_layer_activation = tf.nn.softmax if num_labels > 2 else tf.nn.sigmoid
    final_layer_output_dim = num_labels if num_labels > 2 else 1
    output_layer_l2_regularization = config.get('text_dense_l2_regularization', 0.01)
    output_layer = keras.layers.Dense(final_layer_output_dim,
                                      kernel_regularizer=keras.regularizers.l2(output_layer_l2_regularization),
                                      name='Output',
                                      activation=final_layer_activation)(encoder_text)

    # finalize NN-Model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    if num_labels > 2:
        model_loss = keras.losses.categorical_crossentropy
        model_metrics = [keras.metrics.categorical_accuracy, model_loss]
    else:
        model_loss = keras.losses.binary_crossentropy
        model_metrics = [keras.metrics.binary_accuracy, model_loss]

    adam_learning_rate = config.get('text_adam_learning_rate', 0.001)
    model.compile(optimizer=keras.optimizers.Adam(lr=adam_learning_rate), loss=model_loss, metrics=model_metrics)

    if config.get('print_model_info', False):
        model.summary()
        # keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)

    return model


def train_model(model, x_train, y_train, x_val, y_val, class_weights, num_labels, config):
    # prepare early stopping
    early_stopping_config = config.get('text_early_stopping_config',
                                       {'min_delta': 0.0005, 'patience': 5, 'weights_num_epochs': 1})
    monitored_value = 'val_categorical_crossentropy' if num_labels > 2 else 'val_binary_crossentropy'
    early_stopping_model_checkpoints = \
        EarlyStoppingModelCheckpoints(monitor=monitored_value,
                                      min_delta=early_stopping_config['min_delta'],
                                      patience=early_stopping_config['patience'],
                                      verbose=1 if config.get('print_model_info', False) else 0,
                                      weights_num_epochs=early_stopping_config['weights_num_epochs'],
                                      mode='auto')
    model_callbacks = [early_stopping_model_checkpoints]

    # train model
    num_epochs = 100  # max limit; early stopping should cause earlier break
    model.fit(x_train, y_train,
              epochs=num_epochs,
              batch_size=64,
              validation_data=(x_val, y_val) if len(y_val) else None,
              verbose=2 if config.get('print_model_info', False) else 0,
              callbacks=model_callbacks,
              class_weight=class_weights)




