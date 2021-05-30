import nltk
import sys
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from tensorflow.keras.callbacks import Callback
import numpy as np
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.utils import to_categorical
import os
from src.utils import save_model

MODELS_PATH = '../models'
MODEL_NAME = 'BidirectionalGRU'  # ['RNN', 'BidirectionalRNN', 'BidirectionalGRU']
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
CHECKPOINTS_PATH = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)

BATCH_SIZE = 32


def to_matrix(lines, token_to_id, max_len=None, pad=0, dtype='int32', time_major=False):
    max_len = max_len or max(map(len, lines))
    matrix = np.empty([len(lines), max_len], dtype)
    matrix.fill(pad)

    for i in range(len(lines)):
        line_ix = list(map(token_to_id.__getitem__, lines[i]))[:max_len]
        matrix[i, :len(line_ix)] = line_ix

    return matrix.T if time_major else matrix


def generate_batches(sentences, word_to_id, tag_to_id, all_tags, batch_size=BATCH_SIZE, max_len=None, pad=0):
    assert isinstance(sentences, np.ndarray), "Make sure sentences is q numpy array"

    while True:
        indices = np.random.permutation(np.arange(len(sentences)))
        for start in range(0, len(indices) - 1, batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_words, batch_tags = [], []
            for sent in sentences[batch_indices]:
                words, tags = zip(*sent)
                batch_words.append(words)
                batch_tags.append(tags)

            batch_words = to_matrix(batch_words, word_to_id, max_len, pad)
            batch_tags = to_matrix(batch_tags, tag_to_id, max_len, pad)

            batch_tags_1hot = to_categorical(batch_tags, len(all_tags)).reshape(batch_tags.shape + (-1,))
            yield batch_words, batch_tags_1hot


def build_simple_rnn(all_words, all_tags):
    model = Sequential()
    model.add(L.InputLayer([None], dtype='int32'))
    model.add(L.Embedding(len(all_words), 50))
    model.add(L.SimpleRNN(64, return_sequences=True))

    stepwise_dense = L.Dense(len(all_tags), activation='softmax')
    stepwise_dense = L.TimeDistributed(stepwise_dense)

    model.add(stepwise_dense)

    return model


def build_bidirectional_rnn(all_words, all_tags):
    model = Sequential()

    model.add(L.InputLayer([None], dtype='int32'))
    model.add(L.Embedding(len(all_words), 50))
    model.add(L.Bidirectional(L.SimpleRNN(64, return_sequences=True)))

    stepwise_dense = L.Dense(len(all_tags), activation='softmax')
    stepwise_dense = L.TimeDistributed(stepwise_dense)
    model.add(stepwise_dense)
    return model


def build_bidirectional_gru(all_words, all_tags):
    model = Sequential()

    model.add(L.InputLayer([None], dtype='int32'))
    model.add(L.Embedding(len(all_words), 50))
    model.add(L.Bidirectional(L.GRU(128, return_sequences=True, activation='relu')))
    model.add(L.Dropout(0.5))
    model.add(L.Bidirectional(L.GRU(64, return_sequences=True, activation='relu')))
    model.add(L.Dropout(0.5))

    stepwise_dense = L.Dense(len(all_tags), activation='softmax')
    stepwise_dense = L.TimeDistributed(stepwise_dense)
    model.add(stepwise_dense)
    return model


def main():
    nltk.download('brown')
    nltk.download('universal_tagset')
    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    all_tags = ['#EOS#', '#UNK#', 'ADV', 'NOUN', 'ADP', 'PRON', 'DET', '.', 'PRT', 'VERB', 'X', 'NUM', 'CONJ', 'ADJ']

    data = np.array([[(word.lower(), tag) for word, tag in sentence] for sentence in data])

    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    word_counts = Counter()
    for sentence in data:
        words, tags = zip(*sentence)
        word_counts.update(words)

    all_words = ['#EOS#', '#UNK#'] + list(list(zip(*word_counts.most_common(10000)))[0])

    print("Coverage = %.5f" % (float(sum(word_counts[w] for w in all_words)) / sum(word_counts.values())))

    word_to_id = defaultdict(lambda: 1, {word: i for i, word in enumerate(all_words)})
    tag_to_id = {tag: i for i, tag in enumerate(all_tags)}

    def compute_test_accuracy(model):
        test_words, test_tags = zip(*[zip(*sentence) for sentence in test_data])
        test_words, test_tags = to_matrix(test_words, word_to_id), to_matrix(test_tags, tag_to_id)

        predicted_tag_probabilities = model.predict(test_words, verbose=1)
        predicted_tags = predicted_tag_probabilities.argmax(axis=-1)

        numerator = np.sum(np.logical_and((predicted_tags == test_tags), (test_words != 0)))
        denominator = np.sum(test_words != 0)
        return float(numerator) / denominator

    class EvaluateAccuracy(Callback):
        def on_epoch_end(self, epoch, logs=None):
            sys.stdout.flush()
            acc = compute_test_accuracy(self.model)
            print("\nValidation accuracy: %.5f\n" % acc)
            sys.stdout.flush()

    if MODEL_NAME == 'RNN':
        model = build_simple_rnn(all_words, all_tags)
    elif MODEL_NAME == 'BidirectionalRNN':
        model = build_bidirectional_rnn(all_words, all_tags)
    else:
        model = build_bidirectional_gru(all_words, all_tags)

    model.compile('adam', 'categorical_crossentropy')
    history = model.fit(generate_batches(train_data, word_to_id, tag_to_id, all_tags),
                        steps_per_epoch=len(train_data) / BATCH_SIZE,
                        callbacks=[EvaluateAccuracy()], epochs=5, )

    save_model(model,
               history,
               MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
