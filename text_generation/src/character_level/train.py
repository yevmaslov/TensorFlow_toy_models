from src.download_data import download_dataset
import os
import tensorflow as tf
from src.character_level.models import RNN
from src.utils import save_model
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '../../data'
TSV_PATH = os.path.join(DATA_PATH, 'news-headlines.tsv')

MODELS_PATH = '../../models'
MODEL_NAME = 'RNN'  # ['RNN']
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, MODEL_NAME)

EPOCHS = 25
BATCH_SIZE = 256
DEBUG = True
if DEBUG:
    EPOCHS = 1


def main():
    if not os.path.isfile(TSV_PATH):
        download_dataset(TSV_PATH)

    checkpoint_prefix = os.path.join(MODEL_SAVE_PATH, "ckpt_{epoch}")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_prefix,
                                           monitor='loss',
                                           save_best_only=True)
    ]

    model = None
    history = None
    if MODEL_NAME == 'RNN':
        model, history, char2idx, idx2char = RNN.train_model(TSV_PATH, EPOCHS, BATCH_SIZE, callbacks, DEBUG)

    save_model(model,
               history,
               MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
