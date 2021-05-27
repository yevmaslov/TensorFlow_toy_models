import os
import shutil
import tensorflow_datasets as tfds
import urllib
from zipfile import ZipFile

DATA_PATH = '../data'
GLOVE_PATH = os.path.join(DATA_PATH, 'GloVe')


def download_dataset():
    imdb_train, ds_info = tfds.load(name="imdb_reviews",
                                    split="train",
                                    with_info=True, as_supervised=True)

    imdb_test = tfds.load(name="imdb_reviews", split="test", as_supervised=True)

    return imdb_train, imdb_test


def download_glove():
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'

    if not os.path.isdir(GLOVE_PATH):
        os.mkdir(GLOVE_PATH)

    archive_file_path = os.path.join(GLOVE_PATH, 'glove.6B.zip')
    with urllib.request.urlopen(url) as response, open(archive_file_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    archive = ZipFile(archive_file_path)
    archive.extractall(GLOVE_PATH)


def main():
    print('Downloading dataset...')
    download_dataset()
    print('Downloading GloVe...')
    download_glove()


if __name__ == '__main__':
    main()
