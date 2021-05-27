import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from src.encoder import DataPreprocessor
from src.download_data import download_dataset
import os
from src.utils import save_model

DATA_PATH = '../data'
BATCH_SIZE = 100

GLOVE_PATH = os.path.join(DATA_PATH, 'GloVe')
GLOVE_FN = 'glove.6B.50d.txt'
GLOVE_FILE_PATH = os.path.join(GLOVE_PATH, GLOVE_FN)

MODELS_PATH = '../models'
MODEL_NAME = 'CNN' # ['CNN', 'BiLSTM_with_GloVe_embeddings', ]
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, MODEL_NAME)


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


def build_model_cnn(max_len, embedding_dims, vocab_size):

    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dims,
                  input_length=max_len, mask_zero=True),
        Conv1D(256, 3, padding='valid',
               activation='relu', strides=1),
        MaxPooling1D(2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
    return model


def preprocess_for_cnn(encoded_train, encoded_test):
    x_train, y_train = [], []
    for sample in encoded_train:
        x_train.append(sample[0].numpy())
        y_train.append(sample[1].numpy())

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_valid, y_valid = [], []
    for sample in encoded_test:
        x_valid.append(sample[0].numpy())
        y_valid.append(sample[1].numpy())

    x_valid = np.array(x_train)
    y_valid = np.array(y_train)

    return x_train, y_train, x_valid, y_valid


def main():
    imdb_train, imdb_test = download_dataset()

    tfds_encoder = DataPreprocessor()
    encoded_train, encoded_test = tfds_encoder.prepare_datasets(imdb_train, imdb_test)

    glove_w2v, _ = get_glove_dictionary()

    embedding_dim = 50
    embedding_matrix, _, _ = get_embedding_matrix(tfds_encoder, glove_w2v, embedding_dim)

    rnn_units = 64

    if MODEL_NAME == 'BiLSTM_with_GloVe_embeddings':
        model = build_model_bilstm(
            vocab_size=tfds_encoder.vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            embedding_matrix=embedding_matrix)
        model.summary()

        encoded_train_batched = encoded_train.batch(BATCH_SIZE).prefetch(100)
        history = model.fit(encoded_train_batched, epochs=1)

    elif MODEL_NAME == 'CNN':
        x_train, y_train, x_valid, y_valid = preprocess_for_cnn(encoded_train, encoded_test)

        model = build_model_cnn(max_len=150, embedding_dims=50, vocab_size=tfds_encoder.vocab_size)

        model.summary()
        history = model.fit(x_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=1,
                            validation_data=(x_valid, y_valid))

    save_model(model,
               history,
               MODEL_SAVE_PATH,
               MODEL_NAME)


if __name__ == '__main__':
    main()
