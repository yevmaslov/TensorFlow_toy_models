import numpy as np
import csv
import tensorflow as tf

EOS = '<EOS>'
UNK = '<UNK>'
PAD = '<PAD>'
MAX_LEN = 75

EMBEDDING_DIM = 256
RNN_UNITS = 1024


def get_token_list():
    chars = r"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'''/\|_@#$%ˆ&*˜'+-=()[]{}' ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chars = sorted(set(chars))
    chars = list(chars)
    chars.append(UNK)
    chars.append(EOS)
    chars.insert(0, PAD)
    return chars


def get_char2idx_dict(chars):
    char2idx = {u: i for i, u in enumerate(chars)}
    idx2char = np.array(chars)
    return char2idx, idx2char


def char_idx(c, char2idx):
    return char2idx.get(c, char2idx[UNK])


def read_and_tokenize(data_path, char2idx):
    data = []

    with open(data_path, "r", encoding="utf8") as file:
        lines = csv.reader(file, delimiter='\t')
        for line in lines:
            headline = line[0]
            converted = [char_idx(c, char2idx) for c in headline[:-1]]
            if len(converted) >= MAX_LEN:
                converted = converted[0:MAX_LEN - 1]
                converted.append(char2idx[EOS])
            else:
                converted.append(char2idx[EOS])
                remain = MAX_LEN - len(converted)
                if remain > 0:
                    for i in range(remain):
                        converted.append(char2idx[PAD])
            data.append(converted)
    np_data = np.array(data)
    return np_data


def build_model(vocab_size, batch_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  mask_zero=True,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(vocab_size)
    ])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    return model


def build_gen_model(vocab_size, batch_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def train_model(data_path, epochs, batch_size, callbacks, debug=False):
    chars = get_token_list()
    char2idx, idx2char = get_char2idx_dict(chars)
    data = read_and_tokenize(data_path, char2idx)
    if debug:
        data = data[:1000]
    data_in = data[:, :-1]
    data_out = data[:, 1:]

    x = tf.data.Dataset.from_tensor_slices((data_in, data_out))

    vocab_size = len(chars)

    x_train = x.shuffle(100000, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)

    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        rnn_units=RNN_UNITS,
        batch_size=batch_size)

    model.summary()

    history = model.fit(x_train, epochs=epochs,
                        callbacks=callbacks)

    return model, history, char2idx, idx2char


def restore_model(checkpoint_dir, batch_size=1):
    chars = get_token_list()

    vocab_size = len(chars)

    model = build_gen_model(vocab_size, batch_size)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    return model


def generate_text_greedy(model, start_string, temperature=0.7, num_generate=75):
    chars = get_token_list()
    char2idx, idx2char = get_char2idx_dict(chars)

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)
