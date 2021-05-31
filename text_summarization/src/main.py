import tensorflow as tf
import tensorflow_datasets as tfds
import os
from src.models import seq2seq as s2s
import datetime
import time

DATA_PATH = '../data'
MODELS_PATH = '../models'
ENCODER_NAME = 'encoder'
DECODER_NAME = 'decoder'
ENCODER_SAVE_PATH = os.path.join(MODELS_PATH, ENCODER_NAME)
DECODER_SAVE_PATH = os.path.join(MODELS_PATH, DECODER_NAME)

BUFFER_SIZE = 1500000
BATCH_SIZE = 64
EPOCHS = 3

START = None
END = None
TOKENIZER = None

CHECKPOINT = None


def load_data():

    (ds_train, ds_val, ds_test), ds_info = tfds.load('gigaword',
                                                     split=['train', 'validation', 'test'],
                                                     shuffle_files=True,
                                                     as_supervised=True,
                                                     with_info=True,
                                                     )
    return ds_train, ds_val, ds_test


def get_tokenizer(data, file="gigaword32k.enc"):
    if os.path.exists(file+'.subwords'):
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(file)
    else:
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            ((art.numpy() + b" " + smm.numpy()) for art, smm in data),
            target_vocab_size=2**15
        )
        tokenizer.save_to_file(file)

    return tokenizer


# def encode(article, summary, start, end,
#            tokenizer, art_max_len=128, smry_max_len=50):
#
#     tokens = tokenizer.encode(article.numpy())
#     if len(tokens) > art_max_len:
#         tokens = tokens[:art_max_len]
#     art_enc = sequence.pad_sequences([tokens], padding='post',
#                                      maxlen=art_max_len).squeeze()
#
#     tokens = [start] + tokenizer.encode(summary.numpy())
#     if len(tokens) > smry_max_len:
#         tokens = tokens[:smry_max_len]
#     else:
#         tokens = tokens + [end]
#
#     smry_enc = sequence.pad_sequences([tokens], padding='post',
#                                       maxlen=smry_max_len).squeeze()
#     return art_enc, smry_enc
#
#
# def tf_encode(article, summary):
#     art_enc, smry_enc = tf.py_function(encode, [article, summary],
#                                        [tf.int64, tf.int64])
#     art_enc.set_shape([None])
#     smry_enc.set_shape([None])
#     return art_enc, smry_enc


@tf.function
def train_step(encoder, decoder, inp, targ, enc_hidden, optimizer, start, max_gradient_norm=5):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([start] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input,
                                                 dec_hidden, enc_output)

            loss += s2s.loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, max_gradient_norm)

    optimizer.apply_gradients(zip(clipped_gradients, variables))
    return batch_loss


# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#                     from_logits=False, reduction='none')
#
#
# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)
#
#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask
#
#     return tf.reduce_mean(loss_)


def main():
    print('Loading data...')
    train_ds, _, _ = load_data()
    print('Creating tokenizer (if not cached, may take a long time)...')
    tokenizer = get_tokenizer(train_ds)

    start = tokenizer.vocab_size + 1
    end = tokenizer.vocab_size
    vocab_size = end + 2

    train = train_ds.take(BUFFER_SIZE)  # 1.5M samples
    print("Dataset sample taken")
    train_dataset = train.map(s2s.tf_encode)
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE,
                                        drop_remainder=True)

    steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 128
    units = 256
    encoder = s2s.Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)
    decoder = s2s.Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=steps_per_epoch * (EPOCHS / 2),
        decay_rate=2,
        staircase=False)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    checkpoint_dir = os.path.join(DATA_PATH, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    if CHECKPOINT is not None:
        chkpt_status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        chkpt_status.assert_existing_objects_matched()
    else:
        print("Starting new training run from scratch")

    for epoch in range(EPOCHS):
        start_tm = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (art, smry)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(encoder, decoder,
                                    art, smry,
                                    enc_hidden,
                                    optimizer,
                                    start)
            total_loss += batch_loss
            if batch % 100 == 0:
                ts = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
                print('[{}] Epoch {} Batch {} Loss {:.6f}'.format(ts, epoch + 1,
                                                                  batch,
                                                                  batch_loss.numpy())
                      )

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_tm))

    encoder.summary()
    decoder.summary()


if __name__ == '__main__':
    main()
