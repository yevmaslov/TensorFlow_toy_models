import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import transformer as vt
import time
import datetime

DATA_PATH = '../data'
SAVE_PATH = os.path.join(DATA_PATH, 'features')


def load_image_feature(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8'))
    return img_tensor, cap


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_masks(inp, tar):
    inp_seq = tf.ones([inp.shape[0], inp.shape[1]])
    enc_padding_mask = vt.create_padding_mask(inp_seq)
    dec_padding_mask = vt.create_padding_mask(inp_seq)

    look_ahead_mask = vt.create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = vt.create_padding_mask(tar)

    combined_mask = tf.maximum(dec_target_padding_mask,look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


def main():
    inputs = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'),
                         header=None,
                         names=['caption', 'image'])

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('captions')

    lens = inputs['caption'].map(lambda x: len(x.split()))
    lens = inputs['caption'].map(
        lambda x: len(tokenizer.encode(x.lower()))
    )
    max_len = int(lens.quantile(0.99) + 1)

    start = '<s>'
    end = '</s>'
    inputs['tokenized'] = inputs['caption'].map(lambda x: start + x.lower().strip() + end)

    def tokenize_pad(x):
        x = tokenizer.encode(x)
        if len(x) < max_len:
            x = x + [0] * int(max_len - len(x))
        return x[:max_len]

    inputs['tokens'] = inputs.tokenized.map(lambda x: tokenize_pad(x))

    inputs['img_features'] = inputs['image'].map(lambda x:
                                                 os.path.join(SAVE_PATH,
                                                              x.split('/')[-1][:-3]+'npy'))

    captions = inputs.tokens.tolist()
    img_names = inputs.img_features.tolist()

    img_train, cap_train = img_names, captions

    dataset = tf.data.Dataset.from_tensor_slices((img_train,
                                                  cap_train))

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        load_image_feature, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    num_layers = 4
    d_model = 128
    dff = d_model * 4
    num_heads = 8

    target_vocab_size = tokenizer.vocab_size
    # already includes start/end tokens
    dropout_rate = 0.1
    EPOCHS = 1  # should see results in 4-10 epochs also
    transformer = vt.Transformer(num_layers, d_model, num_heads, dff,
                                 target_vocab_size,
                                 pe_input=49,  # 7x7 pixels
                                 pe_target=target_vocab_size,
                                 rate=dropout_rate,
                                 use_pe=False
                                 )

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    checkpoint_path = "../models/transformer"
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss,
                                  transformer.trainable_variables)

        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    buff_size = 1000
    batch_size = 64

    dataset = dataset.shuffle(buff_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for epoch in range(EPOCHS):
        start_tm = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        # inp -> images, tar -> caption
        for (batch, (inp, tar)) in enumerate(dataset):
            train_step(inp, tar)
            if batch % 100 == 0:
                ts = datetime.datetime.now().strftime(
                    "%d-%b-%Y (%H:%M:%S)")
                print('[{}] Epoch {} Batch {} Loss {:.6f} Accuracy {:.6f}'.format(ts, epoch + 1, batch,
                                                                                  train_loss.result(),
                                                                                  train_accuracy.result()))
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))
        print('Epoch {} Loss {:.6f} Accuracy {:.6f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(
            time.time() - start_tm))


if __name__ == '__main__':
    main()
