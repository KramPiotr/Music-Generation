from tensorflow.keras.optimizers import RMSprop, Adam
import itertools
from tqdm import tqdm
from utilities.run import run
import numpy as np
from RNN_attention.model import create_network
from datetime import date

#epochs

model_param_space = {
    "embed_size":  [2, 3, 5, 10, 30],
    "rnn_units": [32, 64, 128, 256, 512],
    "dense_units": [32, 64, 128, 256, 512],
    "n_lstm": [1, 2, 3, 4],
    "n_dense": [1, 2, 3, 4],
    "drop1": [0, 0.2, 0.5, 0.7],
    "drop2": [0, 0.2, 0.5, 0.7],
    "use_attention": [True, False],
    "learning_rate": [0.0001, 0.0003, 0.001, 0.003, 0.01],
    "optimizer": [RMSprop, Adam]
}

arg_values = list(itertools.product(model_param_space.values()))

for arg_list in tqdm(arg_values):
    experiment_kwargs = {key: arg_list[i] for i, key in enumerate(model_param_space.keys())}
    run("two_datasets_attention", 2, create_network, experiment_kwargs, epochs = 200, patience = 10, descr=f"Hyperparameter training {date.today().strftime('%d/%m/%Y')}")

