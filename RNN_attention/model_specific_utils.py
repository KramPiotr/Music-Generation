import numpy as np
from keras.utils import np_utils
import os
import pickle as pkl
from utilities.utils import retrieve_notes_and_durations, save_notes_and_durations, get_distinct, create_lookups
from utilities.midi_utils import translate_chord
from utilities.notes_utils import multi_hot_encoding_12_tones

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