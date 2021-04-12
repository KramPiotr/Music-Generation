# Author: Piotr Kram
from utilities.predict import predict
import numpy as np
import itertools
from tqdm import tqdm


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


def generate_multihot():
    # np.random.seed(0)
    predict_params = {
        'section': "two_datasets_multihot",
        'version_id': 1,
        'dataset_version': 1,
    }
    multi_predict(predict_params)


def generate_attention_hpc():
    # np.random.seed(0)
    predict_params = {
        'section': "two_datasets_attention_hpc",
        'version_id': 21,
        'dataset_dir': "../run/two_datasets_attention/store",
        'dataset_version': 2,
    }
    multi_predict(predict_params)


if __name__ == "__main__":
    # np.random.seed(0)
    generate_attention_hpc()
