import numpy as np
from keras.utils import np_utils
import os
import pickle as pkl
from utilities.notes_utils import multi_hot_encoding_12_tones
import ntpath
import sys

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

def retrieve_distincts_and_lookups(store_folder):
    distincts_path = os.path.join(store_folder, "distincts")
    lookups_path = os.path.join(store_folder, "lookups")
    if not os.path.exists(distincts_path) or not os.path.exists(lookups_path):
        raise Exception("Invalid path for the distincts and lookups")
    with open(distincts_path, 'rb') as f:
        distincts = pkl.load(f)
    with open(lookups_path, 'rb') as f:
        lookups = pkl.load(f)
    return distincts, lookups

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

def extract_name_from_python_file():
    filename_splitted = ntpath.basename(sys.argv[0]).split(".")
    if len(filename_splitted) > 2:
        raise Exception("The filename contains '.'")
    return filename_splitted[0]

def normalize_array(arr):
  return arr/np.maximum(1, np.tile(np.expand_dims(np.sum(arr, axis = 1), axis=1), (1, arr.shape[1])))

def dump(dir, name, file):
    with open(os.path.join(dir, name), "wb") as f:
        pkl.dump(file, f)

def save_train_test_split(store_folder):
    in_, out_ = retrieve_network_input_output(store_folder)
    dataset_len = len(in_[0])
    trainable = np.ones(dataset_len)
    trainable[:int(0.2*dataset_len)] = 0
    trainable = trainable.astype(bool)
    np.random.shuffle(trainable)
    train_dir = os.path.join(store_folder, "train")
    test_dir = os.path.join(store_folder, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for suffix, ds in zip(("input", "output"), [in_, out_]):
        train_data = [data[trainable] for data in ds]
        test_data = [data[~trainable] for data in ds]
        name = "network_"+suffix
        dump(train_dir, name, train_data)
        dump(test_dir, name, test_data)

def print_to_file(str_, file):
    print(str_, file=file)
    print(str_)

##TODO Question how to sample with temperature from multihot encoding, przygotuj inne pytania na spotkanie

## Zrob jakies unit tests, np czy durations z rests sie zgadzaja
## uporzadkuj code base

if __name__ == "__main__":
    save_train_test_split("../run/two_datasets_attention/store/")