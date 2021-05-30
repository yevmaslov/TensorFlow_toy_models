import os
import json


def save_model(model,
               model_history,
               model_save_path):

    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    with open(os.path.join(model_save_path, 'model_history.json'), "w") as outfile:
        json.dump(model_history.history, outfile)

    model_json = model.to_json()
    with open(os.path.join(model_save_path, 'model.json'), "w") as json_file:
        json_file.write(model_json)

    model.save_weights(os.path.join(model_save_path, 'model_weights.json'))