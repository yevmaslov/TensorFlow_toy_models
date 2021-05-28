import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.download_data import download_dataset
import os
from src.utils import save_model

from src.models import BiLSTM_GloVe
from src.models import CNN
from src.models import BERT

DATA_PATH = '../data'
BATCH_SIZE = 100

MODELS_PATH = '../models'
MODEL_NAME = 'BERT'  # ['CNN', 'BiLSTM_with_GloVe_embeddings', 'BERT']
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
CHECKPOINTS_PATH = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)

DEBUG = True
EPOCHS = 10
if DEBUG:
    EPOCHS = 1


def main():
    imdb_train, imdb_test = download_dataset()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(filepath=CHECKPOINTS_PATH,
                        save_weights_only=True,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)
    ]

    model = None
    history = None
    if MODEL_NAME == 'BiLSTM_with_GloVe_embeddings':
        model, history = BiLSTM_GloVe.train_model(imdb_train, imdb_test, EPOCHS, BATCH_SIZE, callbacks)
    elif MODEL_NAME == 'CNN':
        model, history = CNN.train_model(imdb_train, imdb_test, EPOCHS, BATCH_SIZE, callbacks)
    elif MODEL_NAME == 'BERT':
        model, history = BERT.train_model(imdb_train, imdb_test, EPOCHS, BATCH_SIZE, callbacks)

    save_model(model,
               history,
               MODEL_SAVE_PATH,
               MODEL_NAME)


if __name__ == '__main__':
    main()
