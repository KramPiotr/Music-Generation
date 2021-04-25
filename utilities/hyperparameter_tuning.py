# Author: Piotr Kram
from tensorflow.keras.optimizers import RMSprop, Adam
import itertools
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from utilities.run import run
from utilities.utils import print_to_file
import numpy as np
from RNN_attention.model import create_network
from datetime import date, datetime
import random
from functools import partial
import signal
import time

def log_results(*args):
    global scores, epochs, patience
    with open(f"{log_dir}results{datetime.now().strftime('%m%d%H%M')}.txt", "w") as f:
        print_ = partial(print_to_file, file = f)
        print_(f"Model trained for {epochs} epochs and {patience} patience. Below scores mean the best validation losses achieved.\n")
        scores = sorted(scores, key=lambda x: x[0])
        for i, (score, run_id, kwargs) in enumerate(scores):
            pos_id = f"{i+1}/{len(scores)} {run_id}"
            print_(f'{pos_id:<28}{score}')
            print_("Model parameters:", indent_level = 1)
            for key, value in kwargs.items():
                print_(f"{key:<20}{value}", indent_level = 2)
            print_("")
    sys.exit(1)

#
# dataset_and_model = "two_datasets_attention"
#
# model_param_space = {
#     "embed_size":  [5],
#     "rnn_units": [64, 128], #128
#     "dense_units": [64, 128], #128
#     "n_lstm": [1, 2], #2
#     "n_dense": [1, 2], #1
#     "drop1": [0.2, 0.5], #0.2
#     "drop2": [0.2, 0.5], #0.2
#     "use_attention": [True],
#     "learning_rate": [0.001],
#     "optimizer": [RMSprop]
# }

dataset_and_model = "two_datasets_attention"

model_param_space = {
    "embed_size":  [3],
    "rnn_units": [128, 256], #128
    "dense_units": [128, 256], #128
    "n_lstm": [1, 2], #2
    "n_dense": [1, 2], #1
    "drop1": [0, 0.2], #0.2
    "drop2": [0, 0.2], #0.2
    "use_attention": [True],
    "learning_rate": [0.001],
    "optimizer": [RMSprop]
}


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

arg_values = list(itertools.product(*model_param_space.values()))
random.shuffle(arg_values)

log_dir = f"..\\run\\{dataset_and_model}\\optimisation\\"
os.makedirs(log_dir, exist_ok=True)

epochs = 200
patience = 3


scores = []
for arg_list in tqdm(arg_values):
    experiment_kwargs = dict(zip(model_param_space.keys(), arg_list))
    descr = f"Hyperparameter training {date.today().strftime('%d/%m/%Y')}\n\nRun for parameters:\n"
    for key, value in experiment_kwargs.items():
        descr += f"    {key:<20}{value}\n"
    descr += "\n"
    print(descr)
    run_id, score = run(dataset_and_model, 0, create_network, experiment_kwargs, epochs = epochs, patience = patience, descr=descr)
    time.sleep(1)
    print(f"\nFinal loss achieved for the model with id {run_id} is: {score}")
    scores.append((score, run_id, experiment_kwargs))
    if len(scores) == 1:
        signal.signal(signal.SIGINT, log_results)
log_results()

