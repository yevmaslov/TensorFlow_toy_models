import os
from src.character_level.models import RNN

MODELS_PATH = '../../models'
MODEL_NAME = 'RNN'  # ['RNN']
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, MODEL_NAME)


def main():

    if MODEL_NAME == 'RNN':
        model, char2idx, idx2char = RNN.restore_model(checkpoint_dir=MODEL_SAVE_PATH)
        while True:
            input_sentence = input('Enter a beginning of sentence: ')
            if not input_sentence:
                break
            text = RNN.generate_text_greedy(model, input_sentence, char2idx, idx2char)
            print('RNN predictions: ', text)


if __name__ == '__main__':
    main()
