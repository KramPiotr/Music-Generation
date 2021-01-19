import numpy as np
from keras.utils import np_utils
import os
import pickle as pkl
from utilities.notes_utils import multi_hot_encoding_12_tones

def get_distinct(elements):
    # Get all pitch names
    element_names = sorted(set(elements))  # a lot of 0s for duration...
    n_elements = len(element_names)
    return (element_names, n_elements)


def create_lookups(element_names):
    # create dictionary to map notes and durations to integers
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))

    return (element_to_int, int_to_element)


def prepare_sequences(notes, durations, lookups, distincts, seq_len=32):
    """ Prepare the sequences used to train the Neural Network """

    duration_to_int, int_to_duration = lookups
    duration_names, n_durations = distincts
    encoded_notes = multi_hot_encoding_12_tones(notes)

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = encoded_notes[i:i + seq_len]
        notes_sequence_out = encoded_notes[i + seq_len]
        notes_network_input.append(notes_sequence_in)
        notes_network_output.append(notes_sequence_out)

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.array(notes_network_input)
    durations_network_input = np.array(durations_network_input)
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np.array(notes_network_output)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return network_input, network_output

def sample_with_temp(preds, temperature):
    if temperature == 0:
        return np.argmax(preds)
    else:
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)

def save_notes_and_durations(store_folder, notes, durations):
    os.makedirs(store_folder, exist_ok=True)
    with open(os.path.join(store_folder, 'notes'), 'wb') as f:
        pkl.dump(notes, f)
    with open(os.path.join(store_folder, 'durations'), 'wb') as f:
        pkl.dump(durations, f)

def retrieve_notes_and_durations(store_folder):
    notes_path = os.path.join(store_folder, "notes")
    dur_path = os.path.join(store_folder, "durations")
    if not os.path.exists(notes_path) or not os.path.exists(dur_path):
        return [], []
    with open(notes_path, 'rb') as f:
        notes = pkl.load(f)
    with open(dur_path, 'rb') as f:
        durations = pkl.load(f)
    return notes, durations

def retrieve_network_input_output(store_folder):
    in_path = os.path.join(store_folder, "network_input")
    out_path = os.path.join(store_folder, "network_output")
    if not os.path.exists(in_path) or not os.path.exists(out_path):
        raise Exception("Invalid path for the network input and output")
    with open(in_path, 'rb') as f:
        in_data = pkl.load(f)
    with open(out_path, 'rb') as f:
        out_data = pkl.load(f)
    return in_data, out_data

def color_list(kind, mod):
    color_map = np.random.rand(mod, 3)
    colors = [0]
    for i in range(1, len(kind)):
        if kind[i] == kind[i-1]:
            colors.append(colors[-1])
        else:
            colors.append((colors[-1] + 1)%mod)
    return [color_map[i] for i in colors]

##TODO Question how to sample with temperature from multihot encoding, przygotuj inne pytania na spotkanie

## Zrob jakies unit tests, np czy durations z rests sie zgadzaja
## uporzadkuj code base
