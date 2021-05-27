import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.preprocessing import sequence


class DataPreprocessor:
    def __init__(self):
        self.encoder = None
        self.max_tokens = 0
        self.vocab_size = 0
        self.dict_w2v = {}
        self.unknown_words = []

    def set_encoder(self, train_dataset):
        tokenizer = tfds.deprecated.text.Tokenizer()
        vocabulary_set = set()

        max_tokens = 0
        for example, label in train_dataset:
            some_tokens = tokenizer.tokenize(example.numpy())
            if max_tokens < len(some_tokens):
                max_tokens = len(some_tokens)
            vocabulary_set.update(some_tokens)

        self.encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set, lowercase=True, tokenizer=tokenizer)
        self.max_tokens = max_tokens
        self.vocab_size = self.encoder.vocab_size

    def encode_pad_transform(self, sample):
        encoded = self.encoder.encode(sample.numpy())
        pad = sequence.pad_sequences([encoded], padding='post', maxlen=150)
        return np.array(pad[0], dtype=np.int64)

    def encode_tf_fn(self, sample, label):
        encoded = tf.py_function(self.encode_pad_transform,
                                 inp=[sample],
                                 Tout=(tf.int64))

        encoded.set_shape([None])
        label.set_shape([])
        return encoded, label

    def prepare_datasets(self, train_dataset, test_dataset):
        if self.encoder is None:
            self.set_encoder(train_dataset)

        encoded_train = train_dataset.map(self.encode_tf_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        encoded_test = test_dataset.map(self.encode_tf_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return encoded_train, encoded_test



