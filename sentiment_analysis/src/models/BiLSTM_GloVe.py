import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from src.encoder import DataPreprocessor

DATA_PATH = '../data'

GLOVE_PATH = os.path.join(DATA_PATH, 'GloVe')
GLOVE_FN = 'glove.6B.50d.txt'
GLOVE_FILE_PATH = os.path.join(GLOVE_PATH, GLOVE_FN)


def get_glove_dictionary():
    dict_w2v = {}
    unk_words = []
    with open(GLOVE_FILE_PATH, "rb") as file:
        for line in file:
            tokens = line.split()
            word = tokens[0].decode('utf-8')
            vector = np.array(tokens[1:], dtype=np.float32)

            if vector.shape[0] == 50:
                dict_w2v[word] = vector
            else:
                unk_words.append(word)

    return dict_w2v, unk_words


def get_embedding_matrix(tfds_encoder, glove_w2v, embedding_dim):
    embedding_matrix = np.zeros((tfds_encoder.vocab_size, embedding_dim))

    unk_cnt = 0
    unk_set = set()
    for word in tfds_encoder.encoder.tokens:
        embedding_vector = glove_w2v.get(word)

        if embedding_vector is not None:
            tkn_id = tfds_encoder.encoder.encode(word)[0]
            embedding_matrix[tkn_id] = embedding_vector
        else:
            unk_cnt += 1
            unk_set.add(word)

    return embedding_matrix, unk_cnt, unk_set


def build_model_bilstm(vocab_size, embedding_dim, rnn_units, embedding_matrix, train_emb=False):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, mask_zero=True,
                  weights=[embedding_matrix], trainable=train_emb),
        Bidirectional(LSTM(rnn_units, return_sequences=True,
                           dropout=0.5)),
        Bidirectional(LSTM(rnn_units, dropout=0.25)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
    return model


def train_model(train, test, epochs, batch_size, callbacks):
    embedding_dim = 50
    rnn_units = 64

    tfds_encoder = DataPreprocessor()
    encoded_train, encoded_test = tfds_encoder.prepare_datasets(train, test)

    glove_w2v, _ = get_glove_dictionary()
    embedding_matrix, _, _ = get_embedding_matrix(tfds_encoder, glove_w2v, embedding_dim)

    model = build_model_bilstm(
        vocab_size=tfds_encoder.vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        embedding_matrix=embedding_matrix)
    model.summary()

    encoded_train_batched = encoded_train.batch(batch_size).prefetch(100)
    encoded_test_batched = encoded_test.batch(batch_size).prefetch(100)
    history = model.fit(encoded_train_batched,
                        epochs=epochs,
                        validation_data=encoded_test_batched,
                        callbacks=callbacks)

    return model, history
