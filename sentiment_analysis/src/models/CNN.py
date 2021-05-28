import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from src.encoder import DataPreprocessor


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


def split_encoded(encoded_dataset):
    text, labels = [], []
    for sample in encoded_dataset:
        text.append(sample[0].numpy())
        labels.append(sample[1].numpy())

    text = np.array(text)
    labels = np.array(labels)

    return text, labels


def preprocess_for_cnn(encoded_train, encoded_test):
    x_train, y_train = split_encoded(encoded_train)
    x_valid, y_valid = split_encoded(encoded_test)

    return x_train, y_train, x_valid, y_valid


def train_model(train, test, epochs, batch_size, callbacks):

    tfds_encoder = DataPreprocessor()
    encoded_train, encoded_test = tfds_encoder.prepare_datasets(train, test)

    x_train, y_train, x_valid, y_valid = preprocess_for_cnn(encoded_train, encoded_test)

    model = build_model_cnn(max_len=150, embedding_dims=50, vocab_size=tfds_encoder.vocab_size)

    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_valid, y_valid),
                        callbacks=callbacks)

    return model, history
