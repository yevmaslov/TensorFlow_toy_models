from src.seq2seq import seq2seq as s2s
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import sequence
import os

MODELS_PATH = '../../models'

BATCH_SIZE = 1

art_max_len = 128
smry_max_len = 50


def greedy_search(encoder, decoder, article, tokenizer, units, start, end):

    tokens = tokenizer.encode(article)
    if len(tokens) > art_max_len:
        tokens = tokens[:art_max_len]

    inputs = sequence.pad_sequences([tokens], padding='post',
                                    maxlen=art_max_len).squeeze()
    inputs = tf.expand_dims(tf.convert_to_tensor(inputs), 0)

    summary = ''

    hidden = [tf.zeros((1, units)) for i in range(2)]  # BiRNN
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([start], 0)

    for t in range(smry_max_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()
        if predicted_id == end:
            return summary, article

        summary += tokenizer.decode([predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)

    return summary, article


def main():
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("../gigaword32k.enc")

    embedding_dim = 128
    units = 256

    start = tokenizer.vocab_size + 1
    end = tokenizer.vocab_size
    vocab_size = end + 2

    encoder = s2s.Encoder(vocab_size, embedding_dim, units,
                          BATCH_SIZE)
    decoder = s2s.Decoder(vocab_size, embedding_dim, units,
                          BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = os.path.join(MODELS_PATH, 'checkpoints_seq2seq')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    chkpt_status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    chkpt_status.assert_existing_objects_matched()

    while True:
        article = input('Enter article: ')
        if not article:
            break
        summary, _ = greedy_search(encoder, decoder, article, tokenizer, units, start, end)
        print('Predicted summary: ', summary)


if __name__ == '__main__':
    main()
