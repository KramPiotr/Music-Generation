import numpy as np
from keras.utils import np_utils
import os
import pickle as pkl
from utilities.utils import retrieve_notes_and_durations, save_notes_and_durations, get_distinct, create_lookups, save_train_test_split, retrieve_network_input_output, print_to_file
from utilities.midi_utils import translate_chord
from utilities.notes_utils import multi_hot_encoding_12_tones
from RNN_attention.model import create_network

def create_store(retrieve_folder, store_folder, seq_len=32):
    notes, durations = retrieve_notes_and_durations(retrieve_folder)
    str_notes = []
    for idx, n in enumerate(notes):
        if durations[idx] == 0:
            str_notes.append("S")
        elif n[0] == -1:
            str_notes.append("R")
        else:
            str_notes.append(translate_chord(n, "."))

    save_notes_and_durations(store_folder, str_notes, durations)
    process_notes_and_durations(str_notes, durations, store_folder, seq_len)
    save_train_test_split(store_folder)


def process_notes_and_durations(notes, durations, store_folder, seq_len):

    # get the distinct sets of notes and durations
    note_names, n_notes = get_distinct(notes)
    duration_names, n_durations = get_distinct(durations)
    distincts = [note_names, n_notes, duration_names, n_durations]

    with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
        pkl.dump(distincts, f)

    # make the lookup dictionaries for notes and dictionaries and save
    note_to_int, int_to_note = create_lookups(note_names)
    duration_to_int, int_to_duration = create_lookups(duration_names)
    lookups = [note_to_int, int_to_note, duration_to_int, int_to_duration]

    with open(os.path.join(store_folder, 'lookups'), 'wb') as f:
        pkl.dump(lookups, f)

    network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, seq_len)

    with open(os.path.join(store_folder, 'network_input'), 'wb') as f:
        pkl.dump(network_input, f)

    with open(os.path.join(store_folder, 'network_output'), 'wb') as f:
        pkl.dump(network_output, f)



def prepare_sequences(notes, durations, lookups, distincts, seq_len=32):
    """ Prepare the sequences used to train the Neural Network """

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []

    [note_to_int, _, duration_to_int, _] = lookups
    [_, n_notes, _, n_durations] = distincts

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - seq_len):
        notes_sequence_in = notes[i:i + seq_len]
        notes_sequence_out = notes[i + seq_len]
        notes_network_input.append([note_to_int[char] for char in notes_sequence_in])
        notes_network_output.append(note_to_int[notes_sequence_out])

        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])

    # reshape the input into a format compatible with LSTM layers
    notes_network_input = np.array(notes_network_input)
    durations_network_input = np.array(durations_network_input)
    network_input = [notes_network_input, durations_network_input]

    notes_network_output = np_utils.to_categorical(notes_network_output, num_classes=n_notes)
    durations_network_output = np_utils.to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]

    return network_input, network_output

def predict(version='00', weights_file="weights.h5", embed_size=100, rnn_units=256, use_attention=True):
    store_model_folder = "../run/two_datasets_attention/store"
    run_folder = f"../run/two_datasets_attention/{version}"

    test_folder = os.path.join(store_model_folder, "test")
    test_input, test_output = retrieve_network_input_output(test_folder)

    with open(os.path.join(store_model_folder, "distincts"), 'rb') as f:
        [_, n_notes, _, n_durations] = pkl.load(f)

    model, _ = create_network(n_notes, n_durations, embed_size=embed_size, rnn_units=rnn_units,
                                      use_attention=use_attention)
    if weights_file:
        weights_source = f"../run/two_datasets_attention/00/weights/{weights_file}"
        model.load_weights(weights_source)

    # for i in range(2):
    #     test_input[i] = test_input[i][:100]
    #     test_output[i] = test_output[i][:100]

    with open(os.path.join(run_folder, f"test_results_{weights_file}.txt"), "w") as f:
        print_to_file(f"Evaluation using a test set containing {len(test_input[0])} sequences", f)
        results = model.evaluate(test_input, test_output, batch_size=32)
        for n, r in zip(model.metrics_names, results):
            print_to_file(f"{n:>13}: {r:.4f}", f)

if __name__ == "__main__":
    predict('00', None)