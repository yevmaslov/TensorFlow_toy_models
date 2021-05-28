import os
from src.character_level.models import RNN
from src.character_level.models import GPT2

MODELS_PATH = '../../models'
MODEL_NAME = 'GPT2'  # ['RNN', 'GPT2', ]
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, MODEL_NAME)


def prediction_loop(model, predict_function):
    while True:
        input_sentence = input('Enter a beginning of sentence: ')
        if not input_sentence:
            break
        text = predict_function(model, input_sentence)
        print('Model predictions: ', text)


def main():

    model = None
    predict = None
    if MODEL_NAME == 'RNN':
        model = RNN.restore_model(checkpoint_dir=MODEL_SAVE_PATH)
        predict = RNN.generate_text_greedy

    elif MODEL_NAME == 'GPT2':
        model = GPT2.get_pretrained_model()
        predict = GPT2.generate_text_greedy

    if model is not None:
        prediction_loop(model, predict)


if __name__ == '__main__':
    main()
