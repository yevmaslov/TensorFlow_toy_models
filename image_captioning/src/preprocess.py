import os
import json
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow_datasets as tfds
import tqdm
import numpy as np

DATA_PATH = '../data'
SAVE_PATH = os.path.join(DATA_PATH, 'features')
TRAIN_LEN = 15000  # or None for full
VALID_LEN = 15000  # or None for full


def preprocess_json():
    annotations_path = os.path.join(DATA_PATH, 'annotations')
    train_captions_path = os.path.join(annotations_path, 'captions_train2014.json')
    valid_captions_path = os.path.join(annotations_path, 'captions_val2014.json')

    train_captions = json.load(open(train_captions_path, 'r'))
    valid_captions = json.load(open(valid_captions_path, 'r'))

    train_prefix = os.path.join(DATA_PATH, 'train2014/')
    valid_prefix = os.path.join(DATA_PATH, 'val2014/')

    valid_set = len(valid_captions['images'])
    train_set = len(train_captions['images'])

    if TRAIN_LEN:
        train_set = TRAIN_LEN
    if VALID_LEN:
        valid_set = VALID_LEN

    train_images = {x['id']: x['file_name'] for x in train_captions['images'][:train_set]}
    valid_images = {x['id']: x['file_name'] for x in valid_captions['images'][:valid_set]}
    true_valid_images = {x['id']: x['file_name'] for x in valid_captions['images'][valid_set:valid_set+5000]}

    data = list()
    errors = list()
    validation = list()

    for item in train_captions['annotations']:
        img_id = int(item['image_id'])
        if img_id in train_images:
            file_path = train_prefix + train_images[img_id]
            caption = item['caption']
            data.append((caption, file_path))
        else:
            errors.append(item)

    for item in valid_captions['annotations']:
        caption = item['caption']
        img_id = int(item['image_id'])
        if img_id in valid_images:
            file_path = valid_prefix + valid_images[img_id]
            data.append((caption, file_path))
        elif img_id in true_valid_images:
            file_path = valid_prefix + true_valid_images[img_id]
            validation.append((caption, file_path))
        else:
            errors.append(item)

    with open(os.path.join(DATA_PATH, 'train.csv'), 'w') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerows(data)

    with open(os.path.join(DATA_PATH, 'valid.csv'), 'w') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerows(validation)


def load_image(img_path, img_size=(224, 224)):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = preprocess_input(img)
    return img, img_path


def preprocess_images():
    inputs = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'),
                         header=None,
                         names=['caption', 'image'])

    unique_images = sorted(inputs['image'].unique())
    image_dataset = tf.data.Dataset.from_tensor_slices(unique_images)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    resnet50 = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    new_input = resnet50.input
    hidden_layer = resnet50.layers[-1].output

    features_extractor = tf.keras.Model(new_input, hidden_layer)

    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for img, path in tqdm.tqdm(image_dataset):
        batch_features = features_extractor(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for feat, p in zip(batch_features, path):
            file_path = p.numpy().decode('utf-8')
            file_path = os.path.join(SAVE_PATH, file_path.split('/')[-1][:-3] + 'npy')
            np.save(file_path, feat.numpy())


def preprocess_annotations():
    inputs = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'),
                         header=None,
                         names=['caption', 'image'])

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        inputs['caption'].map(lambda x: x.lower().strip()).tolist(),
        target_vocab_size=2**13,
        reserved_tokens=['<s>', '</s>'],
    )

    tokenizer.save_to_file('captions')


def main():
    print('JSON preprocessing...')
    preprocess_json()
    print('Images preprocessing...')
    preprocess_images()
    print('Captions preprocessing...')
    preprocess_annotations()


if __name__ == '__main__':
    main()
