from RNN_attention.predict import predict
import numpy as np
import itertools
from tqdm import tqdm
inits = [None, 'cfge', 'ttls']
note_temps = np.linspace(0, 1, 5)
duration_temps = np.linspace(0, 1, 5)
weights_files = ['weights.h5']

product = itertools.product(inits, note_temps, duration_temps, weights_files)

for data in tqdm(product):
    predict(*data)