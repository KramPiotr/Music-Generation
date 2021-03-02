from tensorflow.keras.optimizers import RMSprop, Adam
import itertools
from tqdm import tqdm
from utilities.run import run
from utilities.utils import print_to_file
import numpy as np
from RNN_attention.model import create_network
from datetime import date
import random
import os
from functools import partial

dataset_and_model = "two_datasets_attention"

# model_param_space = {
#     "embed_size":  [2, 3, 5, 10, 30],
#     "rnn_units": [32, 64, 128, 256, 512],
#     "dense_units": [32, 64, 128, 256, 512],
#     "n_lstm": [1, 2, 3, 4],
#     "n_dense": [1, 2, 3, 4],
#     "drop1": [0, 0.2, 0.5, 0.7],
#     "drop2": [0, 0.2, 0.5, 0.7],
#     "use_attention": [True, False],
#     "learning_rate": [0.0001, 0.0003, 0.001, 0.003, 0.01],
#     "optimizer": [RMSprop, Adam]
# }

model_param_space = {
    "embed_size":  [5],
    "rnn_units": [64, 128],
    "dense_units": [64, 128],
    "n_lstm": [1, 2],
    "n_dense": [1, 2],
    "drop1": [0.2, 0.5],
    "drop2": [0.2, 0.5],
    "use_attention": [True, False],
    "learning_rate": [0.001],
    "optimizer": [RMSprop]
}

arg_values = list(itertools.product(*model_param_space.values()))
random.shuffle(arg_values)

log_dir = f"../run/{dataset_and_model}/optimisation"
os.makedirs(log_dir, exist_ok=True)

epochs = 200
patience = 10

with open(f"{log_dir}/results.txt", "w") as f:
    print_ = partial(print_to_file, file = f)
    print_(f"Model trained for {epochs} epochs and {patience} patience. Below scores mean the best validation losses achieved.\n")
    scores = []
    for arg_list in tqdm(arg_values):
        experiment_kwargs = dict(zip(model_param_space.keys(), arg_list))
        run_id, score = run(dataset_and_model, 2, create_network, experiment_kwargs, epochs = epochs, patience = patience, descr=f"Hyperparameter training {date.today().strftime('%d/%m/%Y')}")
        scores.append((score, run_id, experiment_kwargs))
    scores = sorted(scores, key=lambda x: x[0])
    for i, (score, run_id, kwargs) in enumerate(scores):
        print_(f'{f"{i+1}/{len(scores)} {run_id}":<28}{score}')
        print_("Model parameters:", indent_level = 1)
        for key, value in kwargs.items():
            print_(f"{key:<20}{value}", indent_level = 2)
        print_("")

