# -*- coding: future_fstrings -*-
import os
import pickle as pkl
from utilities.utils import create_lookups, get_distinct, prepare_sequences, retrieve_notes_and_durations, save_notes_and_durations, save_train_test_split
from utilities.notes_utils import only_chroma

def separate_parts(notes, durations):
    start_seq = []
    starts = []
    for i, (n, d) in enumerate(zip(notes, durations)):
        if d == 0 and n[0] == -1:
            if len(start_seq) > 0 and start_seq[-1][1] == i-1:
                start_seq[-1][1] += 1
            else:
                start_seq.append([i, i])
                starts.append(i)

    for seq in start_seq:
        if seq[1] - seq[0] != 31:
            raise Exception(f"invalid start sequence: {seq}")

    part_by_note = [0]
    for i in range(1, len(notes)):
        next = part_by_note[-1] + 1
        if next < len(starts) and i == starts[next]:
            part_by_note.append(next)
        else:
            part_by_note.append(next - 1)

    return(starts, part_by_note)


def preprocess(notes, durations):
    _, part_by_note = separate_parts(notes, durations)
    n_parts = part_by_note[-1] + 1
    valid_part = [True] * n_parts

    for i, d in enumerate(durations):
        if (d == 0 and notes[i][0] != -1) or d > 3:
            valid_part[part_by_note[i]] = False

    notes = only_chroma(notes)

    for i, n in enumerate(notes):
        if len(n) > 4:
            valid_part[part_by_note[i]] = False

    notes = [n for (i, n) in enumerate(notes) if valid_part[part_by_note[i]]]
    durations = [d for (i, d) in enumerate(durations) if valid_part[part_by_note[i]]]
    return notes, durations

def single_part_preprocess(notes, durations):
    valid = True
    for i, d in enumerate(durations):
        if (d == 0 and notes[i][0] != -1) or d > 3:
            valid = False

    notes = only_chroma(notes)

    for i, n in enumerate(notes):
        if len(n) > 4:
            valid = False

    return (notes, durations) if valid else ([], [])

def process_notes_and_durations(notes, durations, seq_len, store_folder): #TODO move to model specific for multihot
    # get the distinct sets of notes and durations
    duration_names, n_durations = get_distinct(durations)
    distincts = [duration_names, n_durations]

    with open(os.path.join(store_folder, 'distincts'), 'wb') as f:
        pkl.dump(distincts, f)

    # make the lookup dictionaries for notes and dictionaries and save
    duration_to_int, int_to_duration = create_lookups(duration_names)
    lookups = [duration_to_int, int_to_duration]

    with open(os.path.join(store_folder, 'lookups'), 'wb') as f:
        pkl.dump(lookups, f)

    network_input, network_output = prepare_sequences(notes, durations, lookups, distincts, seq_len)

    with open(os.path.join(store_folder, 'network_input'), 'wb') as f:
        pkl.dump(network_input, f)

    with open(os.path.join(store_folder, 'network_output'), 'wb') as f:
        pkl.dump(network_output, f)


if __name__ == "__main__":
    pass
    # for no in ("1", "0"):
    #     save_train_test_split(save_dir+no)