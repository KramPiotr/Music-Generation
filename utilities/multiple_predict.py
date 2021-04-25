# Author: Piotr Kram
from utilities.predict import predict
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import itertools
from tqdm import tqdm
import os
import RNN_attention.model as attention_model
from utilities.midi_to_mp3.midi_to_mp3 import convert_database, get_db_path


def multi_predict(predict_params, experiment_params=None):
    if experiment_params is None:
        experiment_params = {}
    experiment_params.setdefault('init', [None, 'cfge', 'ttls'])
    experiment_params.setdefault('notes_temp', np.linspace(1, 4, 5))
    experiment_params.setdefault('duration_temp', np.linspace(1, 4, 5))

    arg_values = list(itertools.product(*experiment_params.values()))

    for arg_list in tqdm(arg_values):
        experiment_kwargs = {key: arg_list[i] for i, key in enumerate(experiment_params.keys())}
        predict(**predict_params, **experiment_kwargs)

def save_model_from_multihot():
    model_params = {
        "n_notes": 12,
        "n_durations": 19,
        "embed_size": 5,
        "rnn_units": 128,
        "dense_units": 128,
        "n_lstm": 2,
        "n_dense": 1,
        "drop1": 0.2,
        "drop2": 0.2,
        "use_attention": True,
        "learning_rate": 0.001,
        "optimizer": RMSprop
    }
    model = attention_model.create_network(**model_params)[0]
    model_dir = "../run/two_datasets_attention_hpc/21/weights/weights-improvement-11-3.5538.h5"
    model.load_weights(model_dir)
    model.save("../run/two_datasets_attention_hpc/21/final_model")

def generate_multihot():
    # np.random.seed(0)
    predict_params = {
        'section': "two_datasets_multihot",
        'version_id': 8,
        'dataset_version': 3,
    }
    experiment_params = {
        'init': [None, 'cfge', 'ttls'],
        'notes_temp': np.linspace(1, 10, 10),
        'duration_temp': np.linspace(0.5, 3, 5),
    }
    multi_predict(predict_params, experiment_params)
    os.chdir("./midi_to_mp3")
    convert_database(get_db_path("two_datasets_multihot", 8))

def save_model_from_hpc():
    model_params = {
        "n_notes": 439,
        "n_durations": 19,
        "embed_size": 5,
        "rnn_units": 128,
        "dense_units": 128,
        "n_lstm": 2,
        "n_dense": 1,
        "drop1": 0.2,
        "drop2": 0.2,
        "use_attention": True,
        "learning_rate": 0.001,
        "optimizer": RMSprop
    }
    model = attention_model.create_network(**model_params)[0]
    model_dir = "../run/two_datasets_attention_hpc/21/weights/weights-improvement-11-3.5538.h5"
    model.load_weights(model_dir)
    model.save("../run/two_datasets_attention_hpc/21/final_model")


def generate_attention_hpc():
    # np.random.seed(0)
    predict_params = {
        'section': "two_datasets_attention_hpc",
        'version_id': 21,
        'dataset_dir': "../run/two_datasets_attention/store",
        'dataset_version': 3,
    }
    experiment_params = {
        'init': [None, 'cfge', 'ttls'],
        'notes_temp': np.linspace(0, 2, 5),
        'duration_temp': np.linspace(0, 2, 5),
    }
    multi_predict(predict_params, experiment_params)
    os.chdir("./midi_to_mp3")
    convert_database(get_db_path("two_datasets_attention_hpc", 21))


if __name__ == "__main__":
    # np.random.seed(0)
    generate_multihot()
    # generate_attention_hpc()
