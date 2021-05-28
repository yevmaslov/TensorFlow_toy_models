import tensorflow as tf
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification
import numpy as np


def bert_encoder(review, tokenizer):
    txt = review.numpy().decode('utf-8')
    encoded = tokenizer.encode_plus(txt, add_special_tokens=True,
                                    max_length=150,
                                    truncation=True,
                                    pad_to_max_length=True,
                                    return_attention_mask=True,
                                    return_token_type_ids=True)

    return encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask']


def example_to_features(input_ids, attention_masks, token_type_ids, y):
    return {'input_ids': input_ids,
            'attention_mask': attention_masks,
            'token_type_ids': token_type_ids}, y


def preprocess(train, tokenizer):
    bert_train = [bert_encoder(r, tokenizer) for r, l in train]
    bert_lbl = [l for r, l in train]
    bert_train = np.array(bert_train)
    bert_lbl = tf.keras.utils.to_categorical(bert_lbl, num_classes=2)

    x_train, x_val, y_train, y_val = train_test_split(bert_train,
                                                      bert_lbl,
                                                      test_size=0.2,
                                                      random_state=42)

    tr_reviews, tr_segments, tr_masks = np.split(x_train, 3, axis=1)
    val_reviews, val_segments, val_masks = np.split(x_val, 3, axis=1)
    tr_reviews = tr_reviews.squeeze()
    tr_segments = tr_segments.squeeze()
    tr_masks = tr_masks.squeeze()
    val_reviews = val_reviews.squeeze()
    val_segments = val_segments.squeeze()
    val_masks = val_masks.squeeze()

    train_ds = tf.data.Dataset.from_tensor_slices((tr_reviews,
                                                   tr_masks, tr_segments, y_train)). \
        map(example_to_features).shuffle(100).batch(16)
    valid_ds = tf.data.Dataset.from_tensor_slices((val_reviews,
                                                   val_masks, val_segments, y_val)). \
        map(example_to_features).shuffle(100).batch(16)

    return train_ds, valid_ds


def build_model(bert_name):
    bert_model = TFBertForSequenceClassification.from_pretrained(bert_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    bert_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return bert_model


def train_model(train, test, epochs, batch_size, callbacks):
    bert_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_name,
                                              add_special_tokens=True,
                                              do_lower_case=False,
                                              max_length=150,
                                              truncation=True,
                                              pad_to_max_length=True)

    train_ds, valid_ds = preprocess(train, tokenizer)
    model = build_model(bert_name)
    model.summary()

    history = model.fit(train_ds,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=valid_ds,
                        callbacks=callbacks)

    return model, history
